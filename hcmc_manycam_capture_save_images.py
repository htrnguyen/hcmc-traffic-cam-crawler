import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import threading
import subprocess
import logging
from logging.handlers import TimedRotatingFileHandler
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import signal
import platform

import httpx
from PIL import Image

# ================= Helpers ENV =================
def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v.strip() == "":
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if v.strip() == "":
        return default
    try:
        return float(v.strip())
    except Exception:
        return default

# ================= CẤU HÌNH =================
# Crawl & network
INTERVAL_SEC = env_int("INTERVAL_SEC", 60)
CONCURRENCY = env_int("CONCURRENCY", 200)
TIMEOUT_S = env_float("TIMEOUT_S", 8.0)
RETRIES = env_int("RETRIES", 2)

# Staging + logging
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "staging_images"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / "capture.log"

# Rclone & Drive
RCLONE_EXE = os.environ.get("RCLONE_EXE", "rclone")  
RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrv")
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "")

# Rclone tuning
RCLONE_MIN_AGE_S = env_int("RCLONE_MIN_AGE_S", 30) 
RCLONE_PERIOD_S = env_int("RCLONE_PERIOD_S", 60)  
RCLONE_TRANSFERS = env_int("RCLONE_TRANSFERS", 32)
RCLONE_CHECKERS = env_int("RCLONE_CHECKERS", 32)
RCLONE_BUFFER_SIZE = os.environ.get("RCLONE_BUFFER_SIZE", "32M")
RCLONE_CHUNK_SIZE = os.environ.get("RCLONE_CHUNK_SIZE", "64M")
RCLONE_NO_TRAVERSE = os.environ.get("RCLONE_NO_TRAVERSE", "1") 

# Ảnh/đầu cuối
IMG_BASE = "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx"
GOOD_SIZE = (512, 288)
NA_SIZE = (284, 177)
NA_SHA256 = "1ef51d905a2e9d42678f4a37d5d54ecf9407dfd683ef39607782ddd64ab1aed5"
NA_AHASH_8x8 = "ffe3cbc3c3e781ff"

# JPEG output
JPEG_QUALITY = 85
JPEG_OPTIMIZE = True
JPEG_PROGRESSIVE = True

# ================= VN TIMEZONE =================
try:
    from zoneinfo import ZoneInfo

    VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    VN_TZ = timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")


def now_vn() -> datetime:
    return datetime.now(VN_TZ)


# ================= LOGGING =================
logger = logging.getLogger("capture")
logger.setLevel(logging.INFO)


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = TimedRotatingFileHandler(
        filename=str(LOG_FILE),
        when="midnight",
        backupCount=14,
        encoding="utf-8",
        utc=False,
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)


# ================= UTILS =================
def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_image_url(cam_id: str) -> str:
    t = int(time.time() * 1000) 
    return f"{IMG_BASE}?id={cam_id}&t={t}"


def read_image_meta(b: bytes) -> Optional[Tuple[int, int, str]]:
    try:
        with Image.open(BytesIO(b)) as im:
            im.load()
            return im.size[0], im.size[1], (im.format or "").upper()
    except Exception:
        return None


def ahash_8x8_hex(b: bytes) -> Optional[str]:
    try:
        with Image.open(BytesIO(b)) as im:
            g = im.convert("L").resize((8, 8), Image.LANCZOS)
            pixels = list(g.getdata())
            avg = sum(pixels) / 64.0
            bits = 0
            for p in pixels:
                bits = (bits << 1) | (1 if p >= avg else 0)
            return f"{bits:016x}"
    except Exception:
        return None


def is_not_available(w: int, h: int, sha: str, ahash: Optional[str]) -> bool:
    return (w, h) == NA_SIZE or sha == NA_SHA256 or (ahash == NA_AHASH_8x8)


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def safe_iso_for_name(dt: datetime) -> str:
    s = dt.isoformat(timespec="milliseconds")
    return s.replace(":", "-")


def decide_ext(fmt: str) -> str:
    f = (fmt or "").upper()
    if f in ("JPG", "JPEG"):
        return ".jpg"
    elif f == "PNG":
        return ".png"
    elif f == "GIF":
        return ".gif"
    else:
        return ".jpg"


def maybe_convert_to_jpeg(original_bytes: bytes, fmt: str) -> Tuple[bytes, str]:
    f = (fmt or "").upper()
    if f in ("JPG", "JPEG"):
        return original_bytes, ".jpg"
    try:
        with Image.open(BytesIO(original_bytes)) as im:
            rgb = im.convert("RGB")
            buf = BytesIO()
            rgb.save(
                buf,
                format="JPEG",
                quality=JPEG_QUALITY,
                optimize=JPEG_OPTIMIZE,
                progressive=JPEG_PROGRESSIVE,
            )
            return buf.getvalue(), ".jpg"
    except Exception:
        return original_bytes, decide_ext(fmt)


def parse_http_date_utc(date_str: str) -> Optional[datetime]:
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


class _IgnoreCtrlC:
    """Tắt tạm SIGINT trong with-block."""

    def __enter__(self):
        try:
            self._old = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            self._old = None

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._old is not None:
                signal.signal(signal.SIGINT, self._old)
        except Exception:
            pass


def _rclone_run_safe(args: list[str]) -> subprocess.CompletedProcess:
    """
    Chạy rclone cách ly Ctrl+C:
    - POSIX: start_new_session=True (ngăn SIGINT truyền sang child)
    - Windows: CREATE_NEW_PROCESS_GROUP (ngăn Ctrl+C truyền sang child)
    """
    kwargs = dict(capture_output=True, text=True)
    if _is_windows():
        creationflags = 0
        try:
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        except Exception:
            pass

        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    return subprocess.run(args, **kwargs)


# ================= KẾT QUẢ 1 CAM =================
@dataclass
class CamResult:
    cam_id: str
    status: str  # 'ok', 'dup', 'na', 'bad_size', 'fetch', 'meta', 'error'


# ================= HTTP FETCH (trả cả thời điểm) =================
async def fetch_bytes(client: httpx.AsyncClient, url: str):
    """
    Trả (bytes|None, headers|None, t0_utc, t1_utc) để tính midpoint + skew.
    """
    delay = 0.2
    for _ in range(RETRIES + 1):
        t0 = datetime.now(timezone.utc)
        try:
            r = await client.get(url, timeout=TIMEOUT_S)
            t1 = datetime.now(timezone.utc)
            if r.status_code == 200 and r.content:
                return r.content, r.headers, t0, t1
        except Exception:
            t1 = datetime.now(timezone.utc)
        await asyncio.sleep(delay)
        delay *= 2
    nowu = datetime.now(timezone.utc)
    return None, None, nowu, nowu


# ================= LƯU ẢNH =================
def save_image_bytes_atomic(cam_id: str, ts_vn: datetime, ext: str, img_bytes: bytes):
    # <OUT_ROOT>/<cam_id>/<YYYY-MM-DD>/<ISO>_<cam_id>.<ext>
    day = ts_vn.strftime("%Y-%m-%d")
    ts_iso = safe_iso_for_name(ts_vn)
    rel_name = f"{ts_iso}_{cam_id}.{ext.lstrip('.')}"
    out_dir = OUT_ROOT / cam_id / day
    out_path = out_dir / rel_name
    if out_path.exists():
        return out_path  
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    ensure_parent(out_path)
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    return out_path


# ================= RCLONE =================
def _rclone_move_args(min_age_s: int) -> List[str]:
    args = [
        RCLONE_EXE,
        "move",
        str(OUT_ROOT),
        f"{RCLONE_REMOTE}:",
        "--drive-root-folder-id",
        GDRIVE_FOLDER_ID,
        "--transfers",
        str(RCLONE_TRANSFERS),
        "--checkers",
        str(RCLONE_CHECKERS),
        "--buffer-size",
        RCLONE_BUFFER_SIZE,
        "--drive-chunk-size",
        RCLONE_CHUNK_SIZE,
        "--min-age",
        f"{min_age_s}s",
        "--ignore-existing",
        "--create-empty-src-dirs",
        "--delete-empty-src-dirs",
        "--retries",
        "6",
        "--low-level-retries",
        "20",
        "--log-level",
        "INFO",
    ]
    if RCLONE_NO_TRAVERSE in ("1", "true", "TRUE", "on", "yes"):
        args.append("--no-traverse")
    return args


def rclone_move_once(label: str, min_age_s: int) -> Tuple[int, int, int]:
    """
    Chạy rclone move một lần (cách ly Ctrl+C).
    Trả (uploaded, deleted, return_code).
    CHỈ log nếu có thay đổi (uploaded+deleted>0) hoặc có lỗi.
    Luôn kèm thời gian thực thi (s).
    """
    if not GDRIVE_FOLDER_ID.strip():
        logger.error("[SYNC][%s] ERROR (missing GDRIVE_FOLDER_ID env)", label)
        return (0, 0, 1)

    t0 = time.perf_counter()
    try:
        out = _rclone_run_safe(_rclone_move_args(min_age_s))
    except FileNotFoundError:
        logger.error("[SYNC][%s] ERROR (rclone not found: %s)", label, RCLONE_EXE)
        return (0, 0, 127)
    dur = time.perf_counter() - t0

    text = (out.stdout or "") + (("\n" + out.stderr) if out.stderr else "")
    uploaded = text.count("Copied (new)")
    deleted = text.count("Deleted")

    if out.returncode != 0:
        logger.error("[SYNC][%s] ERROR (exit=%s) | %.2fs", label, out.returncode, dur)
    else:
        if (uploaded + deleted) > 0:
            logger.info(
                "[SYNC][%s] Ok (Uploaded=%d, Deleted=%d) | %.2fs",
                label,
                uploaded,
                deleted,
                dur,
            )

    return (uploaded, deleted, out.returncode)


def start_uploader_thread(stop_event: threading.Event):
    def loop():
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        while not stop_event.wait(RCLONE_PERIOD_S):
            try:
                rclone_move_once("Loop", RCLONE_MIN_AGE_S)
            except Exception:
                logger.error("[SYNC][Loop] ERROR (unexpected exception)")

    th = threading.Thread(target=loop, name="rclone-uploader", daemon=True)
    th.start()
    return th


# ================= XỬ LÝ 1 CAM =================
@dataclass
class Cam:
    cam_id: str


async def process_one(
    cam: Dict[str, Any], client: httpx.AsyncClient, last_sha: Dict[str, str]
) -> CamResult:
    cam_id = cam.get("cam_id") or ""
    try:
        url = build_image_url(cam_id)
        b, headers, t0_utc, t1_utc = await fetch_bytes(client, url)
        if not b:
            return CamResult(cam_id, "fetch")

        # Midpoint + hiệu chỉnh theo server Date (nếu có)
        mid_utc = t0_utc + (t1_utc - t0_utc) / 2
        ts_server_utc = None
        if headers and "Date" in headers:
            ts_server_utc = parse_http_date_utc(headers["Date"])
        if ts_server_utc is not None:
            skew = ts_server_utc - t1_utc  # server - local
            mid_utc = mid_utc + skew
        ts_vn = mid_utc.astimezone(VN_TZ)

        meta = read_image_meta(b)
        if not meta:
            return CamResult(cam_id, "meta")
        w, h, fmt = meta

        sha = sha256_hex(b)
        ah = ahash_8x8_hex(b)

        if is_not_available(w, h, sha, ah):
            return CamResult(cam_id, "na")
        if (w, h) != GOOD_SIZE:
            return CamResult(cam_id, "bad_size")
        if last_sha.get(cam_id) == sha:
            return CamResult(cam_id, "dup")

        out_bytes, out_ext = maybe_convert_to_jpeg(b, fmt)
        save_image_bytes_atomic(cam_id, ts_vn, out_ext, out_bytes)
        last_sha[cam_id] = sha
        return CamResult(cam_id, "ok")
    except Exception:
        return CamResult(cam_id, "error")


# ================= VÒNG CHẠY CHÍNH =================
async def run_loop(cams_json_path: Path):
    cams_raw = json.loads(cams_json_path.read_text(encoding="utf-8"))
    cams: List[Dict[str, Any]] = [c for c in cams_raw if c.get("cam_id")]
    if not cams:
        logger.error("Không load được camera.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    last_sha: Dict[str, str] = {}

    limits = httpx.Limits(
        max_connections=CONCURRENCY * 2, max_keepalive_connections=CONCURRENCY
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        "Origin": "https://giaothong.hochiminhcity.gov.vn",
        "Referer": "https://giaothong.hochiminhcity.gov.vn/",
    }

    async with httpx.AsyncClient(
        http2=True, headers=headers, limits=limits, timeout=None
    ) as client:
        sem = asyncio.Semaphore(CONCURRENCY)
        stop_event = threading.Event()
        up_thread = start_uploader_thread(stop_event)

        try:
            while True:
                tick_start = time.perf_counter()

                async def task(cam: Dict[str, Any]):
                    async with sem:
                        # jitter 5–30 ms để tránh bắn cùng lúc
                        await asyncio.sleep(
                            0.005 + (hash(cam.get("cam_id")) % 25) / 1000.0
                        )
                        return await process_one(cam, client, last_sha)

                results: List[CamResult] = await asyncio.gather(
                    *[task(c) for c in cams], return_exceptions=False
                )

                total = len(results)
                ok_count = sum(1 for r in results if r.status in ("ok", "dup"))
                elapsed = time.perf_counter() - tick_start
                logger.info("[OK] : %d/%d | %.2fs", ok_count, total, elapsed)

                failed_ids = [
                    r.cam_id for r in results if r.status not in ("ok", "dup")
                ]
                if failed_ids:
                    logger.warning("[ERROR] : %s", "; ".join(failed_ids))

                await asyncio.sleep(max(0.0, INTERVAL_SEC - elapsed))
        finally:
            # 1) dừng uploader loop
            try:
                stop_event.set()
            except Exception:
                pass

            # 2) chờ uploader kết thúc để tránh rclone song song
            try:
                if up_thread.is_alive():
                    up_thread.join(timeout=10.0)
            except Exception:
                pass

            # 3) Final flush: bỏ qua Ctrl+C trong pha shutdown để rclone không bị cắt giữa chừng
            with _IgnoreCtrlC():
                # Final-1: theo min-age cấu hình (thận trọng)
                try:
                    rclone_move_once("Final-1", RCLONE_MIN_AGE_S)
                except Exception:
                    pass

                # Final-2: min-age=0 (lúc này không còn ghi file)
                try:
                    rclone_move_once("Final-2", 0)
                except Exception:
                    pass

                # 4) Nếu vẫn còn file trong OUT_ROOT (do mạng nghẽn/quota), retry thêm 2 lần nhẹ
                def _has_files(root: Path) -> bool:
                    try:
                        for _p in root.rglob("*"):
                            if _p.is_file():
                                return True
                        return False
                    except Exception:
                        return True  # bảo thủ: coi như còn

                retries = 2
                while retries > 0 and _has_files(OUT_ROOT):
                    time.sleep(2)  # nghỉ ngắn
                    try:
                        rclone_move_once(f"Final-Retry-{3 - retries}", 0)
                    except Exception:
                        pass
                    retries -= 1


# ================= MAIN =================
def main():
    os.environ.setdefault("PYTHONUTF8", "1")
    for s in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        try:
            if s and hasattr(s, "reconfigure"):
                s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass

    setup_logging()

    if not GDRIVE_FOLDER_ID.strip():
        logger.error(
            "Missing env GDRIVE_FOLDER_ID. Set it via environment or .env file."
        )
        raise SystemExit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-file", required=True)
    args = ap.parse_args()

    cams_path = Path(args.chunk_file)
    try:
        asyncio.run(run_loop(cams_path))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: shutting down gracefully...")


if __name__ == "__main__":
    main()
