import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import signal
import sys
import tarfile
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import requests
import yaml
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from PIL import Image


# ========== Utils ==========
def now_epoch_ms() -> int:
    return int(time.time() * 1000)


def hour_bucket(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H")


def ts_for_filename(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def ahash_8x8_hex(img: Image.Image) -> str:
    g = img.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << (63 - i)
    return f"{bits:016x}"


def safe_rmtree(p: Path):
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def prune_empty_parents(p: Path, stop_at: Path):
    """Remove empty parent directories up to stop_at"""
    try:
        cur = p
        stop_at = stop_at.resolve()
        while True:
            if not cur.exists():
                cur = cur.parent
            elif cur.is_dir() and not any(cur.iterdir()):
                if cur.resolve() == stop_at:
                    break
                cur.rmdir()
                cur = cur.parent
            else:
                break
    except Exception:
        pass


# ========== Circuit Breaker ==========
class CameraCircuitBreaker:
    """Circuit breaker for cameras that consistently fail"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_failure_times: Dict[str, float] = {}

    def record_failure(self, cam_id: str):
        """Record a failure for a camera"""
        self.failure_counts[cam_id] += 1
        self.last_failure_times[cam_id] = time.time()

    def record_success(self, cam_id: str):
        """Record a success - reset failure count"""
        self.failure_counts[cam_id] = 0
        if cam_id in self.last_failure_times:
            del self.last_failure_times[cam_id]

    def should_skip(self, cam_id: str) -> bool:
        """Check if camera should be skipped due to circuit breaker"""
        if cam_id not in self.failure_counts:
            return False

        if self.failure_counts[cam_id] >= self.failure_threshold:
            if cam_id in self.last_failure_times:
                time_since_failure = time.time() - self.last_failure_times[cam_id]
                if time_since_failure < self.timeout:
                    return True
                else:
                    # Reset after timeout
                    self.failure_counts[cam_id] = 0
                    del self.last_failure_times[cam_id]
        return False


# ========== Drive Uploader ==========
class DriveUploader:
    """Fast parallel uploader with correct folder structure"""

    def __init__(
        self,
        client_secret_json: str,
        token_json: str,
        root_folder_id: str,
        logger: logging.Logger,
    ):
        self.client_secret_json = client_secret_json
        self.token_json = token_json
        self.root_folder_id = root_folder_id
        self.logger = logger
        self._session: Optional[AuthorizedSession] = None
        self._folder_cache: Dict[str, str] = {}

    def _ensure_session(self):
        if self._session:
            return
        scopes = ["https://www.googleapis.com/auth/drive"]
        creds = None
        if os.path.exists(self.token_json):
            creds = Credentials.from_authorized_user_file(self.token_json, scopes)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_json, scopes=scopes
                )
                creds = flow.run_local_server(port=0)
            Path(self.token_json).write_text(creds.to_json(), encoding="utf-8")
        self._session = AuthorizedSession(creds)

    def _get_or_create_folder_structure(
        self, date_folder: str, cam_folder_name: str
    ) -> str:
        """Get or create folder structure: YYYYMMDD/cam_001_camid under root_folder_id"""
        self._ensure_session()
        cache_key = f"{date_folder}/{cam_folder_name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        # Step 1: Get or create date folder under root
        q_date = (
            f"mimeType='application/vnd.google-apps.folder' and trashed=false and "
            f"name='{date_folder}' and '{self.root_folder_id}' in parents"
        )
        r_date = self._session.get(
            "https://www.googleapis.com/drive/v3/files",
            params={"q": q_date, "fields": "files(id,name)", "pageSize": 1},
        )
        r_date.raise_for_status()
        files_date = r_date.json().get("files", [])

        if files_date:
            date_id = files_date[0]["id"]
        else:
            meta_date = {
                "name": date_folder,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [self.root_folder_id],
            }
            r_date = self._session.post(
                "https://www.googleapis.com/drive/v3/files",
                headers={"Content-Type": "application/json"},
                data=json.dumps(meta_date),
            )
            r_date.raise_for_status()
            date_id = r_date.json()["id"]

        # Step 2: Get or create camera folder under date folder
        q_cam = (
            f"mimeType='application/vnd.google-apps.folder' and trashed=false and "
            f"name='{cam_folder_name}' and '{date_id}' in parents"
        )
        r_cam = self._session.get(
            "https://www.googleapis.com/drive/v3/files",
            params={"q": q_cam, "fields": "files(id,name)", "pageSize": 1},
        )
        r_cam.raise_for_status()
        files_cam = r_cam.json().get("files", [])

        if files_cam:
            cam_id = files_cam[0]["id"]
        else:
            meta_cam = {
                "name": cam_folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [date_id],
            }
            r_cam = self._session.post(
                "https://www.googleapis.com/drive/v3/files",
                headers={"Content-Type": "application/json"},
                data=json.dumps(meta_cam),
            )
            r_cam.raise_for_status()
            cam_id = r_cam.json()["id"]

        self._folder_cache[cache_key] = cam_id
        return cam_id

    def upload_archive(
        self,
        archive_path: Path,
        date_folder: str,
        cam_folder_name: str,
        max_retries: int = 2,
    ) -> bool:
        """Upload single archive with optimized speed, under YYYYMMDD/cam_001_camid"""
        for attempt in range(max_retries + 1):
            try:
                folder_id = self._get_or_create_folder_structure(
                    date_folder, cam_folder_name
                )

                metadata = {"name": archive_path.name, "parents": [folder_id]}
                meta_bytes = json.dumps(metadata).encode("utf-8")
                boundary = f"===============BOUNDARY{int(time.time()*1000)}=="
                delimiter = f"--{boundary}\r\n"
                close_delim = f"--{boundary}--\r\n"

                body = io.BytesIO()
                body.write(delimiter.encode())
                body.write(b"Content-Type: application/json; charset=UTF-8\r\n\r\n")
                body.write(meta_bytes)
                body.write(b"\r\n")
                body.write(delimiter.encode())
                body.write(b"Content-Type: application/x-tar\r\n\r\n")

                with open(archive_path, "rb") as f:
                    shutil.copyfileobj(f, body)
                body.write(b"\r\n")
                body.write(close_delim.encode())

                r = self._session.post(
                    "https://www.googleapis.com/upload/drive/v3/files",
                    params={"uploadType": "multipart"},
                    headers={"Content-Type": f"multipart/related; boundary={boundary}"},
                    data=body.getvalue(),
                    timeout=25,  # Fast timeout
                )
                r.raise_for_status()
                return True

            except Exception as e:
                if attempt >= max_retries:
                    self.logger.error(
                        f"Failed to upload {archive_path.name} after {max_retries + 1} attempts: {e}"
                    )
                    return False
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        return False


# ========== Crawler ==========
class CameraCrawler:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.logger = logging.getLogger("crawler")
        self.logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        self.logger.addHandler(h)

        # Validate required configuration
        self._validate_config(cfg)

        self.endpoint_tmpl = cfg["endpoint_template"]
        self.data_dir = Path(cfg["data_dir"])
        self.archive_dir = Path(cfg["archive_dir"])

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Timing configuration
        self.crawl_interval_minutes = cfg.get("crawl_interval_minutes", 5)
        self.sweep_interval = self.crawl_interval_minutes * 60

        # Batch processing
        self.batch_size = cfg.get("batch_size", 50)
        self.batch_delay = cfg.get("batch_delay", 3)

        # HTTP settings
        self.http_timeout = cfg.get("http_timeout", 8)
        self.http_retries = cfg.get("http_retries", 1)
        self.concurrency_limit = cfg.get("concurrency_limit", 50)

        # Error detection - made less strict
        self.use_error_detection = all(
            k in cfg for k in ["na_size", "na_sha256", "na_ahash_8x8"]
        )
        if self.use_error_detection:
            self.na_size = tuple(cfg["na_size"])
            self.na_sha256 = cfg["na_sha256"].lower()
            self.na_ahash = cfg["na_ahash_8x8"].lower()

        self.blacklist_file = Path(cfg["blacklist_file"])
        self.camera_ids_file = Path(cfg["camera_ids_file"])

        self.stop_event = asyncio.Event()
        self.last_upload_hour = None
        self.current_hour = None

        # Upload thread management
        self._upload_thread = None
        self._current_batch_task = None

        # Circuit breaker for failing cameras
        self.circuit_breaker = CameraCircuitBreaker(
            failure_threshold=cfg.get("circuit_breaker_threshold", 5),
            timeout=cfg.get("circuit_breaker_timeout", 300),
        )

        self.uploader = DriveUploader(
            client_secret_json=cfg["oauth_client_secret"],
            token_json=cfg["oauth_token_file"],
            root_folder_id=cfg["drive_folder_id"],
            logger=self.logger,
        )

        self.blacklist: Set[str] = set()
        self.active_cams: List[Tuple[int, str]] = []

        # Enhanced metrics
        self.stats = {
            "total_crawled": 0,
            "errors": 0,
            "uploaded": 0,
            "blacklisted": 0,
            "circuit_breaker_skips": 0,
            "errors_by_type": defaultdict(int),
            "start_time": time.time(),
        }

    def _validate_config(self, cfg: Dict):
        """Validate required configuration keys"""
        required_keys = [
            "endpoint_template",
            "data_dir",
            "archive_dir",
            "oauth_client_secret",
            "oauth_token_file",
            "drive_folder_id",
            "blacklist_file",
            "camera_ids_file",
        ]
        missing_keys = [key for key in required_keys if key not in cfg]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    def load_camera_ids(self):
        """Load camera IDs from CSV file with index,cam_id format"""
        if not self.camera_ids_file.exists():
            self.logger.error(f"Camera IDs file not found: {self.camera_ids_file}")
            return

        cams = []
        try:
            with open(self.camera_ids_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (
                        "index" in row
                        and "cam_id" in row
                        and row["index"].strip()
                        and row["cam_id"].strip()
                    ):
                        try:
                            index = int(row["index"].strip())
                            cam_id = row["cam_id"].strip()
                            cams.append((index, cam_id))
                        except (ValueError, KeyError):
                            continue
        except Exception as e:
            self.logger.error(f"Failed to load camera IDs: {e}")
            return

        cams.sort(key=lambda x: x[0])
        self.active_cams = cams
        self.logger.info(f"Loaded {len(cams)} camera IDs from CSV")

    def load_blacklist(self):
        if self.blacklist_file.exists():
            self.blacklist = {
                l.strip()
                for l in self.blacklist_file.read_text(encoding="utf-8").splitlines()
                if l.strip()
            }
        else:
            self.blacklist = set()
        self.logger.info(f"Loaded blacklist: {len(self.blacklist)} cameras")

    def append_blacklist(self, cam_id: str):
        if cam_id in self.blacklist:
            return
        self.blacklist.add(cam_id)
        self.stats["blacklisted"] += 1
        with open(self.blacklist_file, "a", encoding="utf-8") as f:
            f.write(cam_id + "\n")

    def _is_error_image(self, raw: bytes) -> bool:
        """Improved error detection - less strict to avoid false positives"""
        # First check size - very small is definitely error
        if len(raw) < 100:  # Less than 100 bytes is definitely error
            return True

        # If no error detection config, only check if it's a valid image
        if not self.use_error_detection:
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    # Basic validation - must be reasonable size
                    if im.size[0] < 50 or im.size[1] < 50:  # Very small images
                        return True
                    return False  # Valid image
            except Exception:
                return True  # Invalid image

        # Check specific error patterns
        sha = hashlib.sha256(raw).hexdigest().lower()
        if sha == self.na_sha256:
            return True

        try:
            with Image.open(io.BytesIO(raw)) as im:
                # Check specific error image size
                if im.size == self.na_size:
                    return True
                # Check ahash for error image pattern
                if ahash_8x8_hex(im).lower() == self.na_ahash:
                    return True
                # Less strict size check - only very small images
                if im.size[0] < 50 or im.size[1] < 50:
                    return True
                # Less strict single color check - completely uniform images only
                pixels = list(im.convert("RGB").getdata())
                if len(set(pixels)) == 1:  # Completely uniform color
                    return True
        except Exception:
            # If can't open as image, it's an error
            return True
        return False

    async def _fetch_one(
        self, session: aiohttp.ClientSession, cam_index: int, cam_id: str
    ) -> Tuple[int, str, Optional[bytes], Optional[str]]:
        """Fetch one camera image with enhanced error tracking"""
        # Check circuit breaker
        if self.circuit_breaker.should_skip(cam_id):
            self.stats["circuit_breaker_skips"] += 1
            return cam_index, cam_id, None, "circuit_breaker"

        url = self.endpoint_tmpl.format(cam_id=cam_id, epoch_ms=now_epoch_ms())
        error_type = None

        for attempt in range(self.http_retries + 1):
            try:
                async with session.get(url, timeout=self.http_timeout) as resp:
                    if resp.status != 200:
                        error_type = f"http_{resp.status}"
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message="non-200",
                            headers=resp.headers,
                        )
                    data = await resp.read()
                    self.circuit_breaker.record_success(cam_id)
                    return cam_index, cam_id, data, None
            except asyncio.TimeoutError:
                error_type = "timeout"
            except aiohttp.ClientError:
                error_type = "client_error"
            except Exception as e:
                error_type = "unknown"

            if attempt >= self.http_retries:
                self.stats["errors"] += 1
                self.stats["errors_by_type"][error_type] += 1
                self.circuit_breaker.record_failure(cam_id)
                return cam_index, cam_id, None, error_type
            await asyncio.sleep(0.1 * (attempt + 1))

        return cam_index, cam_id, None, error_type

    def _save_original_image(
        self, cam_index: int, cam_id: str, raw: bytes, dt: datetime
    ):
        """Save original image with unique naming and improved error detection"""
        if self._is_error_image(raw):
            self.append_blacklist(cam_id)
            self.logger.warning(
                f"Blacklisted cam_{cam_index:03d}_{cam_id} (error image, size: {len(raw)} bytes)"
            )
            return

        try:
            # Use microsecond timestamp + UUID for guaranteed uniqueness
            microsec_ts = dt.strftime("%Y%m%d_%H%M%S_%f")
            unique_id = str(uuid.uuid4())[:8]

            # Create structured path: data/cam_001_camid/20240315_14/20240315_142301_123456_abc12345.jpg
            cam_folder = f"cam_{cam_index:03d}_{cam_id}"
            hour_dir = self.data_dir / cam_folder / hour_bucket(dt)

            # Unique filename with microsecond timestamp and UUID
            filename = f"{microsec_ts}_{unique_id}.jpg"
            img_path = hour_dir / filename

            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(raw)
            self.stats["total_crawled"] += 1

        except Exception as e:
            self.logger.exception(f"Save failed for cam_{cam_index:03d}_{cam_id}: {e}")
            self.stats["errors"] += 1
            self.stats["errors_by_type"]["save_error"] += 1

    async def _crawl_batch(
        self, session: aiohttp.ClientSession, cams: List[Tuple[int, str]], dt: datetime
    ):
        """Crawl a batch of cameras with enhanced error tracking"""
        # Filter out blacklisted cameras at batch level
        active_cams = [
            (idx, cam_id) for idx, cam_id in cams if cam_id not in self.blacklist
        ]

        tasks = [self._fetch_one(session, idx, cam_id) for idx, cam_id in active_cams]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = 0
        for r in results:
            if isinstance(r, Exception):
                self.stats["errors"] += 1
                self.stats["errors_by_type"]["exception"] += 1
            else:
                cam_index, cam_id, raw, error_type = r
                if raw:
                    self._save_original_image(cam_index, cam_id, raw, dt)
                    processed += 1
                elif error_type:
                    self.logger.debug(
                        f"Failed to fetch cam_{cam_index:03d}_{cam_id}: {error_type}"
                    )

        return processed

    async def _crawl_sweep(self, cams: List[Tuple[int, str]]):
        """Crawl all cameras in batches with small delays"""
        connector = aiohttp.TCPConnector(limit=self.concurrency_limit)
        headers = {
            "Accept": "image/*",
            "User-Agent": "HCMC-Traffic-CamCrawler/2.1",
            "Cache-Control": "no-cache",
        }

        dt = datetime.now()
        total_processed = 0

        async with aiohttp.ClientSession(
            connector=connector, headers=headers
        ) as session:
            # Split cameras into batches
            for i in range(0, len(cams), self.batch_size):
                batch = cams[i : i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(cams) + self.batch_size - 1) // self.batch_size

                self.logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} cameras)"
                )

                # Store current batch task for graceful shutdown
                self._current_batch_task = asyncio.create_task(
                    self._crawl_batch(session, batch, dt)
                )
                processed = await self._current_batch_task
                total_processed += processed

                # Small delay between batches
                if i + self.batch_size < len(cams) and not self.stop_event.is_set():
                    try:
                        await asyncio.wait_for(
                            self.stop_event.wait(), timeout=self.batch_delay
                        )
                        self.logger.info("Batch processing cancelled")
                        break
                    except asyncio.TimeoutError:
                        pass

        self.logger.info(
            f"Sweep completed: {total_processed}/{len(cams)} cameras processed"
        )

    def _archive_hour_data(self, hour_folder: str):
        """Archive all cameras from a specific hour folder"""
        archives_created = 0

        # Check if data directory exists
        if not self.data_dir.exists():
            return archives_created

        try:
            for cam_dir in self.data_dir.iterdir():
                if not cam_dir.is_dir():
                    continue

                cam_folder_name = cam_dir.name  # cam_001_camid format
                hour_dir = cam_dir / hour_folder

                if not hour_dir.exists() or not hour_dir.is_dir():
                    continue

                # Get all images from this hour
                jpgs = sorted(hour_dir.glob("*.jpg"))
                if not jpgs:
                    continue

                # Create archive: cam_001_camid_20240315_14.tar
                tar_path = self.archive_dir / f"{cam_folder_name}_{hour_folder}.tar"
                if tar_path.exists():
                    continue

                try:
                    with tarfile.open(tar_path, "w") as tar:
                        for jpg_path in jpgs:
                            tar.add(jpg_path, arcname=jpg_path.name)

                    archives_created += 1
                    self.logger.info(f"Archived {tar_path.name} ({len(jpgs)} images)")

                    # Clean up hour directory after successful archive
                    safe_rmtree(hour_dir)
                    prune_empty_parents(hour_dir.parent, stop_at=self.data_dir)

                except Exception as e:
                    self.logger.error(
                        f"Failed to archive {cam_folder_name}/{hour_folder}: {e}"
                    )
        except Exception as e:
            self.logger.error(f"Error accessing data directory: {e}")

        return archives_created

    def _upload_archives(self):
        """Upload all pending archives with fast parallel processing"""
        archives = sorted(self.archive_dir.glob("*.tar"))
        if not archives:
            return

        uploaded = 0
        failed = 0

        import concurrent.futures

        batch_size = 8
        self.logger.info(
            f"Starting upload of {len(archives)} archives (batches of {batch_size})"
        )

        for i in range(0, len(archives), batch_size):
            batch = archives[i : i + batch_size]

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_size
            ) as executor:
                futures = []

                for archive_path in batch:
                    # Extract cam_folder_name and date from filename: cam_001_camid_20240101_14.tar
                    parts = archive_path.stem.split("_")
                    if len(parts) >= 4:
                        cam_folder_name = "_".join(parts[:-2])  # cam_001_camid
                        date_folder = parts[-2]  # YYYYMMDD
                        hour = parts[-1]  # HH
                        future = executor.submit(
                            self.uploader.upload_archive,
                            archive_path,
                            cam_folder_name,
                            date_folder,
                        )
                        futures.append((future, archive_path, cam_folder_name))
                    else:
                        self.logger.warning(
                            f"Invalid archive name format: {archive_path.name}"
                        )
                        continue

                # Collect results
                for future, archive_path, cam_folder_name in futures:
                    try:
                        success = future.result(timeout=30)  # 30s timeout per file
                        if success:
                            archive_path.unlink()
                            uploaded += 1
                        else:
                            failed += 1
                            self.logger.warning(
                                f"Failed to upload: {archive_path.name}"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Upload timeout/error for {archive_path.name}: {e}"
                        )
                        failed += 1

            # Progress log every 5 batches
            batch_num = i // batch_size + 1
            total_batches = (len(archives) + batch_size - 1) // batch_size
            if batch_num % 5 == 0 or batch_num == total_batches:
                self.logger.info(
                    f"Upload progress: batch {batch_num}/{total_batches}, {uploaded} completed"
                )

        self.stats["uploaded"] += uploaded
        self.logger.info(f"Upload completed: {uploaded} success, {failed} failed")

    def _start_background_upload(self, archives_count: int):
        """Start upload in background thread - completely non-blocking"""
        if self._upload_thread and self._upload_thread.is_alive():
            self.logger.info("Upload thread already running - skipping")
            return

        self._upload_thread = threading.Thread(
            target=self._upload_archives, daemon=True
        )
        self._upload_thread.start()
        self.logger.info(
            f"Started background upload thread for {archives_count} archives - crawling continues"
        )

    def _log_stats(self):
        """Log enhanced statistics"""
        runtime = time.time() - self.stats["start_time"]
        active_count = len(self.active_cams) - self.stats["blacklisted"]

        self.logger.info(
            f"Runtime: {runtime:.1f}s | "
            f"Crawled: {self.stats['total_crawled']} | "
            f"Errors: {self.stats['errors']} | "
            f"Blacklisted: {self.stats['blacklisted']} | "
            f"Uploaded: {self.stats['uploaded']} | "
            f"CB Skips: {self.stats['circuit_breaker_skips']} | "
            f"Active: {active_count}"
        )

        if self.stats["errors_by_type"]:
            error_summary = ", ".join(
                [
                    f"{error_type}: {count}"
                    for error_type, count in self.stats["errors_by_type"].items()
                ]
            )
            self.logger.info(f"Error breakdown: {error_summary}")

    async def graceful_shutdown(self):
        """Perform graceful shutdown"""
        self.logger.info("Initiating graceful shutdown...")
        self.stop_event.set()

        # Wait for current batch to complete
        if self._current_batch_task and not self._current_batch_task.done():
            self.logger.info("Waiting for current batch to complete...")
            try:
                await asyncio.wait_for(self._current_batch_task, timeout=30)
            except asyncio.TimeoutError:
                self.logger.warning("Current batch did not complete in time")

        # Archive current hour data
        if self.last_upload_hour:
            self.logger.info(f"Archiving current hour data: {self.last_upload_hour}")
            archives_created = self._archive_hour_data(self.last_upload_hour)
            if archives_created > 0:
                self.logger.info(f"Created {archives_created} archives during shutdown")

    async def run_forever(self):
        self.load_blacklist()
        self.load_camera_ids()

        # Filter out blacklisted cameras
        active_cams = [
            (idx, cam_id)
            for idx, cam_id in self.active_cams
            if cam_id not in self.blacklist
        ]
        self.logger.info(f"Starting crawler with {len(active_cams)} active cameras")
        self.logger.info(f"Crawl interval: {self.crawl_interval_minutes} minutes")
        self.logger.info(
            f"Batch size: {self.batch_size}, Concurrency: {self.concurrency_limit}"
        )
        if self.use_error_detection:
            self.logger.info("Error detection enabled")
        else:
            self.logger.info("Error detection disabled - accepting all valid images")

        # Signal handling
        loop = asyncio.get_running_loop()
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig:
                try:
                    loop.add_signal_handler(
                        sig, lambda: asyncio.create_task(self.graceful_shutdown())
                    )
                except NotImplementedError:
                    pass

        try:
            while not self.stop_event.is_set():
                start_time = time.time()
                current_hour = hour_bucket(datetime.now())

                # Check if we moved to a new hour - archive and upload previous hour
                if self.last_upload_hour and self.last_upload_hour != current_hour:
                    self.logger.info(
                        f"New hour detected: {current_hour} (was {self.last_upload_hour})"
                    )

                    # Archive quickly
                    archives_created = self._archive_hour_data(self.last_upload_hour)
                    if archives_created > 0:
                        # Start upload in background thread - NO BLOCKING
                        self._start_background_upload(archives_created)
                    else:
                        self.logger.info(
                            f"No data to archive for hour {self.last_upload_hour}"
                        )

                self.last_upload_hour = current_hour

                # Reload blacklist periodically to catch new additions
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    old_count = len(self.blacklist)
                    self.load_blacklist()
                    if len(self.blacklist) != old_count:
                        self.logger.info(
                            f"Blacklist updated: {len(self.blacklist)} cameras"
                        )

                # Crawl cameras for current hour - THIS HAPPENS IMMEDIATELY
                self.logger.info(f"Starting sweep for hour {current_hour}")
                await self._crawl_sweep(active_cams)

                elapsed = time.time() - start_time
                self._log_stats()

                # Wait for next sweep
                wait_time = max(0.0, self.sweep_interval - elapsed)
                if wait_time > 0:
                    self.logger.info(f"Waiting {wait_time:.1f}s until next sweep")
                    try:
                        await asyncio.wait_for(
                            self.stop_event.wait(), timeout=wait_time
                        )
                        break  # Stop event was set
                    except asyncio.TimeoutError:
                        pass

        except KeyboardInterrupt:
            self.logger.info("Ctrl+C detected - starting graceful shutdown")
            await self.graceful_shutdown()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            await self.graceful_shutdown()
        finally:
            # Final cleanup
            self.logger.info("Performing final cleanup...")

            # Wait for upload thread to complete
            if self._upload_thread and self._upload_thread.is_alive():
                self.logger.info("Waiting for background upload to complete...")
                self._upload_thread.join(timeout=60)
                if self._upload_thread.is_alive():
                    self.logger.warning("Upload thread did not complete in time")

            # Final upload of any remaining archives
            remaining_archives = list(self.archive_dir.glob("*.tar"))
            if remaining_archives:
                self.logger.info(
                    f"Final upload of {len(remaining_archives)} remaining archives"
                )
                self._upload_archives()

            self._log_stats()
            self.logger.info("Crawler stopped cleanly")


# ========== Entry ==========
def main():
    cfg_path = os.environ.get("CRAWLER_CONFIG", "config.yaml")
    if not Path(cfg_path).exists():
        print(f"Missing config file: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load config file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        crawler = CameraCrawler(cfg)
        asyncio.run(crawler.run_forever())
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Stopped by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
