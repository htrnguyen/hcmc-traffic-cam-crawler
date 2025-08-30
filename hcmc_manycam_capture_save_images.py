#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, asyncio, json, logging, platform, shutil, subprocess, sys, time, threading, hashlib
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image
from dotenv import dotenv_values

# ========= Load config from .env =========
CFG = dotenv_values(".env")


def must_get(k: str) -> str:
    v = (CFG.get(k) or "").strip()
    if not v:
        raise RuntimeError(f"Missing config in .env: {k}")
    return v


def as_int(k: str) -> int:
    return int(must_get(k))


def as_float(k: str) -> float:
    return float(must_get(k))


def parse_size(s: str) -> Tuple[int, int]:
    s = s.lower().strip()
    if "x" not in s:
        raise RuntimeError(f"Invalid NA_SIZE, expect WxH: {s}")
    w, h = s.split("x")
    return int(w), int(h)


# ---- Timezone ----
try:
    from zoneinfo import ZoneInfo

    VN_TZ = ZoneInfo(must_get("TIMEZONE"))  # e.g. Asia/Ho_Chi_Minh
except Exception:
    VN_TZ = timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")

# ---- Crawl ----
INTERVAL_SEC = as_int("INTERVAL_SEC")
CONCURRENCY = as_int("CONCURRENCY")
TIMEOUT_S = as_float("TIMEOUT_S")
RETRIES = as_int("RETRIES")

# ---- Paths ----
OUT_ROOT = Path(must_get("OUT_ROOT"))
LOG_DIR = Path(must_get("LOG_DIR"))
LOG_FILE = LOG_DIR / "capture.log"

# ---- Upload mode ----
UPLOAD_MODE = (
    (CFG.get("UPLOAD_MODE") or "files_from").strip().lower()
)  # "all" | "files_from"
UPLOAD_LIST = CFG.get("UPLOAD_LIST") or (LOG_DIR / "upload_now.txt").as_posix()
PRIME_FULL_MOVE = (CFG.get("PRIME_FULL_MOVE") or "1").lower() in (
    "1",
    "true",
    "on",
    "yes",
)

# ---- Snapshot endpoint & NA fingerprint ----
IMG_BASE = must_get("IMG_BASE")
NA_SIZE = parse_size(must_get("NA_SIZE"))
NA_SHA256 = must_get("NA_SHA256")
NA_AHASH = must_get("NA_AHASH")

# ---- Blacklist settings ----
BLACKLIST_MODE = (
    (CFG.get("BLACKLIST_MODE") or "na_immediate").strip().lower()
)  # off | na_immediate | na_after_streak
BLACKLIST_STREAK = int((CFG.get("BLACKLIST_STREAK") or "3").strip())
BLACKLIST_TTL_DAYS = int((CFG.get("BLACKLIST_TTL_DAYS") or "7").strip())
BLACKLIST_FILE = (
    CFG.get("BLACKLIST_FILE") or (LOG_DIR / "blacklist.json").as_posix()
).strip()
BLACKLIST_DELETE_LOCAL = (CFG.get("BLACKLIST_DELETE_LOCAL") or "1").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)

# ---- Rclone (validated when --rclone on) ----
RCLONE_EXE = (CFG.get("RCLONE_EXE") or "").strip()
RCLONE_REMOTE = (CFG.get("RCLONE_REMOTE") or "").strip()
GDRIVE_FOLDER_ID = (CFG.get("GDRIVE_FOLDER_ID") or "").strip()
RCLONE_MIN_AGE_S = int((CFG.get("RCLONE_MIN_AGE_S") or "10").strip())
RCLONE_TRANSFERS = int((CFG.get("RCLONE_TRANSFERS") or "16").strip())
RCLONE_CHECKERS = int((CFG.get("RCLONE_CHECKERS") or "16").strip())
RCLONE_BUFFER_SIZE = (CFG.get("RCLONE_BUFFER_SIZE") or "8M").strip()
RCLONE_CHUNK_SIZE = (CFG.get("RCLONE_CHUNK_SIZE") or "32M").strip()
RCLONE_NO_TRAVERSE = (CFG.get("RCLONE_NO_TRAVERSE") or "1").strip()

# ========= Logging =========
logger = logging.getLogger("capture")
logger.setLevel(logging.INFO)


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = TimedRotatingFileHandler(
        str(LOG_FILE), when="midnight", backupCount=14, encoding="utf-8", utc=False
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)


# ========= Utils =========
def now_vn() -> datetime:
    return datetime.now(VN_TZ)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_image_url(cam_id: str) -> str:
    return f"{IMG_BASE}?id={cam_id}&t={int(time.time()*1000)}"


def read_image_meta(b: bytes) -> Optional[Tuple[int, int, str]]:
    with Image.open(BytesIO(b)) as im:
        im.load()
        return im.size[0], im.size[1], (im.format or "").upper()


def ahash_8x8_hex(b: bytes) -> Optional[str]:
    with Image.open(BytesIO(b)) as im:
        g = im.convert("L").resize((8, 8), Image.LANCZOS)
        pixels = list(g.getdata())
        avg = sum(pixels) / 64.0
        bits = 0
        for p in pixels:
            bits = (bits << 1) | (1 if p >= avg else 0)
        return f"{bits:016x}"


def is_not_available(w: int, h: int, sha: str, ahash: Optional[str]) -> bool:
    return (w, h) == NA_SIZE or sha == NA_SHA256 or (ahash == NA_AHASH)


def safe_iso_for_name(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace(":", "-")


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def parse_http_date_utc(date_str: str) -> Optional[datetime]:
    dt = parsedate_to_datetime(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ========= Save (atomic) + enqueue (event-driven) =========
# Queue item: (relative_posix_path, enqueued_epoch_seconds)
_UPLOAD_Q: deque[Tuple[str, float]] = deque()
_UPLOAD_LOCK = threading.Lock()
_UPLOAD_COND = threading.Condition(_UPLOAD_LOCK)  # báo thức uploader


def save_image_bytes_atomic(cam_id: str, ts_vn: datetime, img_bytes: bytes) -> Path:
    day = ts_vn.strftime("%Y-%m-%d")
    ts_iso = safe_iso_for_name(ts_vn)
    out_dir = OUT_ROOT / cam_id / day
    out_path = out_dir / f"{ts_iso}_{cam_id}.jpg"
    if out_path.exists():
        return out_path
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    ensure_parent(out_path)
    with open(tmp, "wb") as f:
        f.write(img_bytes)
        f.flush()
    tmp.replace(out_path)
    # enqueue + đánh thức uploader
    try:
        rel = out_path.relative_to(OUT_ROOT).as_posix()
    except ValueError:
        rel = out_path.name
    with _UPLOAD_COND:
        _UPLOAD_Q.append((rel, time.time()))
        _UPLOAD_COND.notify()  # có file mới -> báo uploader
    return out_path


def _queue_size() -> int:
    with _UPLOAD_LOCK:
        return len(_UPLOAD_Q)


# ========= Blacklist state =========
_blacklist: Dict[str, float] = {}  # cam_id -> expire_epoch (0 = perm)
_BL_LOCK = threading.Lock()
na_streak: Dict[str, int] = {}


def _bl_load():
    p = Path(BLACKLIST_FILE)
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    now = time.time()
    cleaned: Dict[str, float] = {}
    for cid, exp in obj.items():
        try:
            e = float(exp)
        except Exception:
            continue
        if e == 0 or e > now:
            cleaned[str(cid)] = e
    with _BL_LOCK:
        _blacklist.clear()
        _blacklist.update(cleaned)


def _bl_save():
    p = Path(BLACKLIST_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_blacklist, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(p)


def _bl_add(cam_id: str):
    if BLACKLIST_MODE == "off":
        return
    exp = 0.0
    if BLACKLIST_TTL_DAYS > 0:
        exp = time.time() + BLACKLIST_TTL_DAYS * 86400
    with _BL_LOCK:
        if cam_id in _blacklist:
            return
        _blacklist[cam_id] = exp
        _bl_save()
    if BLACKLIST_DELETE_LOCAL:
        shutil.rmtree(OUT_ROOT / cam_id, ignore_errors=True)
    ttl_txt = f"{BLACKLIST_TTL_DAYS}d" if BLACKLIST_TTL_DAYS > 0 else "∞"
    logger.warning("[BLACKLIST] add %s (ttl=%s)", cam_id, ttl_txt)


# ========= Rclone =========
def _validate_rclone_config():
    if not RCLONE_EXE:
        raise RuntimeError("Missing RCLONE_EXE in .env")
    if not RCLONE_REMOTE:
        raise RuntimeError("Missing RCLONE_REMOTE in .env")
    if not GDRIVE_FOLDER_ID:
        raise RuntimeError("Missing GDRIVE_FOLDER_ID in .env")


def _rclone_run_safe(args: List[str]) -> subprocess.CompletedProcess:
    kwargs = dict(capture_output=True, text=True)
    if platform.system().lower().startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return subprocess.run(args, **kwargs)


def _args_common_all(min_age_s: int) -> List[str]:
    # dùng cho move toàn thư mục (có --min-age)
    args = [
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
        "--retries",
        "6",
        "--low-level-retries",
        "20",
        "--log-level",
        "INFO",
        "--fast-list",
        "--delete-empty-src-dirs",
    ]
    if RCLONE_NO_TRAVERSE.lower() in ("1", "true", "on", "yes"):
        args.append("--no-traverse")
    return args


def _args_common_no_filter() -> List[str]:
    # dùng cho files_from (KHÔNG filter)
    args = [
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
        "--retries",
        "6",
        "--low-level-retries",
        "20",
        "--log-level",
        "INFO",
        "--delete-empty-src-dirs",
    ]
    if RCLONE_NO_TRAVERSE.lower() in ("1", "true", "on", "yes"):
        args.append("--no-traverse")
    return args


def rclone_move_all(label: str, min_age_s: int) -> Tuple[int, int, int]:
    args = [RCLONE_EXE, "move", str(OUT_ROOT), f"{RCLONE_REMOTE}:"]
    args += _args_common_all(min_age_s)
    t0 = time.perf_counter()
    out = _rclone_run_safe(args)
    dur = time.perf_counter() - t0
    text = (out.stdout or "") + (("\n" + out.stderr) if out.stderr else "")
    uploaded, deleted = text.count("Copied (new)"), text.count("Deleted")
    if out.returncode != 0:
        tail = text.strip()[-400:]
        logger.error(
            "[SYNC][%s] ERROR (exit=%s) | %.2fs | %s", label, out.returncode, dur, tail
        )
    elif (uploaded + deleted) > 0:
        logger.info(
            "[SYNC][%s] Ok (Uploaded=%d, Deleted=%d) | %.2fs",
            label,
            uploaded,
            deleted,
            dur,
        )
    else:
        logger.info("[SYNC][%s] NoChange | %.2fs", label, dur)
    return uploaded, deleted, out.returncode


def rclone_move_list(rel_paths: List[str]) -> Tuple[int, int, int]:
    # Ghi danh sách (dùng POSIX path mọi nền tảng để tránh edge trên Windows)
    filt: List[str] = []
    for rp in rel_paths:
        src = OUT_ROOT / rp.replace("\\", "/")
        if not src.is_file():
            continue
        filt.append(rp.replace("\\", "/"))  # luôn dùng '/' trong list

    if not filt:
        return (0, 0, 0)

    lst_path = Path(UPLOAD_LIST)
    lst_path.parent.mkdir(parents=True, exist_ok=True)
    lst_path.write_text("\n".join(filt), encoding="utf-8")

    args = [RCLONE_EXE, "move", str(OUT_ROOT), f"{RCLONE_REMOTE}:"]
    args += ["--files-from-raw", str(lst_path)]
    args += _args_common_no_filter()  # KHÔNG filter
    args += [
        "--tpslimit",
        "4",
        "--tpslimit-burst",
        "8",
        "--drive-pacer-min-sleep",
        "100ms",
        "--drive-pacer-burst",
        "200",
        "--timeout",
        "5m",
        "--contimeout",
        "30s",
    ]
    t0 = time.perf_counter()
    out = _rclone_run_safe(args)
    dur = time.perf_counter() - t0

    text = (out.stdout or "") + (("\n" + out.stderr) if out.stderr else "")
    uploaded, deleted = text.count("Copied (new)"), text.count("Deleted")

    if out.returncode != 0:
        tail = text.strip()[-500:]
        logger.error("[SYNC][List] ERROR (exit=%s) | %s", out.returncode, tail)
        if out.returncode == 1:
            # requeue lại các file còn tồn tại
            with _UPLOAD_COND:
                now = time.time()
                for rp in filt:
                    rp_posix = rp.replace("\\", "/")
                    if (OUT_ROOT / rp_posix).is_file():
                        _UPLOAD_Q.appendleft((rp_posix, now))
                _UPLOAD_COND.notify()
        return (uploaded, deleted, out.returncode)

    if (uploaded + deleted) > 0:
        logger.info(
            "[SYNC][List] Ok (Uploaded=%d, Deleted=%d) | %.2fs", uploaded, deleted, dur
        )
    else:
        logger.info("[SYNC][List] NoChange (%d files listed) | %.2fs", len(filt), dur)
    return (uploaded, deleted, 0)


# ========= Uploader thread (event-driven, không rớt file trẻ) =========
def _compute_next_ready_delay(min_age_s: int) -> float:
    """Trả về số giây cần chờ để có ít nhất 1 file trong queue đủ tuổi; 0 nếu đã có."""
    now = time.time()
    with _UPLOAD_LOCK:
        if not _UPLOAD_Q:
            return None  # Không có file -> chờ vô hạn (đến khi notify)
        # tuổi nhỏ nhất còn thiếu để đạt min_age
        shortest = min(max(0.0, min_age_s - (now - enq)) for _, enq in _UPLOAD_Q)
        return shortest


def _dequeue_ready_paths(min_age_s: int) -> List[str]:
    """Lấy các đường dẫn đã đủ tuổi theo enqueued_time; phần còn lại trả về queue."""
    now = time.time()
    ready: List[str] = []
    not_ready: List[Tuple[str, float]] = []
    with _UPLOAD_LOCK:
        while _UPLOAD_Q:
            rp, enq = _UPLOAD_Q.popleft()
            src = OUT_ROOT / rp.replace("\\", "/")
            if not src.is_file():
                continue
            if (now - enq) < min_age_s:
                not_ready.append((rp, enq))
            else:
                ready.append(rp)
        for item in reversed(not_ready):
            _UPLOAD_Q.appendleft(item)
    return ready


def start_uploader_thread(stop_event: threading.Event) -> threading.Thread:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    busy = threading.Lock()

    def loop():
        if UPLOAD_MODE == "files_from" and PRIME_FULL_MOVE:
            try:
                rclone_move_all("Prime", RCLONE_MIN_AGE_S)
            except Exception:
                logger.warning("[SYNC][Prime] skipped")

        while True:
            if stop_event.is_set():
                break

            # Tính thời gian chờ tới khi có file đủ tuổi; nếu None -> chờ đến khi có notify
            delay = _compute_next_ready_delay(RCLONE_MIN_AGE_S)
            with _UPLOAD_COND:
                if delay is None:
                    _UPLOAD_COND.wait()  # chờ đến khi có file mới hoặc stop
                elif delay > 0:
                    _UPLOAD_COND.wait(timeout=delay)

            if stop_event.is_set():
                break

            if not busy.acquire(blocking=False):
                continue
            try:
                if UPLOAD_MODE == "files_from":
                    batch = _dequeue_ready_paths(RCLONE_MIN_AGE_S)
                    if batch:
                        rclone_move_list(batch)
                else:
                    # mode "all": đẩy cả OUT_ROOT theo chu kỳ file-ready (ít dùng)
                    rclone_move_all("Loop", RCLONE_MIN_AGE_S)
            finally:
                busy.release()

    th = threading.Thread(target=loop, name="rclone-uploader", daemon=True)
    th.start()
    return th


# ========= Per-cam =========
@dataclass
class CamResult:
    cam_id: str
    status: str  # 'ok','dup','na','fetch','meta','error'


async def fetch_bytes(client: httpx.AsyncClient, url: str):
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


async def process_one(
    cam: Dict[str, Any], client: httpx.AsyncClient, last_sha: Dict[str, str]
) -> CamResult:
    cam_id = (cam.get("cam_id") or "").strip()
    if not cam_id:
        return CamResult("", "error")

    b, headers, t0, t1 = await fetch_bytes(client, build_image_url(cam_id))
    if not b:
        return CamResult(cam_id, "fetch")

    # timestamp midpoint + skew theo server Date
    mid_utc = t0 + (t1 - t0) / 2
    if headers and "Date" in headers:
        ts_server_utc = parse_http_date_utc(headers["Date"])
        if ts_server_utc:
            mid_utc = mid_utc + (ts_server_utc - t1)
    ts_vn = mid_utc.astimezone(VN_TZ)

    m = read_image_meta(b)
    if not m:
        return CamResult(cam_id, "meta")
    w, h, fmt = m
    sha = sha256_hex(b)
    ah = ahash_8x8_hex(b)

    if is_not_available(w, h, sha, ah):
        return CamResult(cam_id, "na")
    if last_sha.get(cam_id) == sha:
        return CamResult(cam_id, "dup")

    save_image_bytes_atomic(cam_id, ts_vn, b)
    last_sha[cam_id] = sha
    return CamResult(cam_id, "ok")


# ========= Main loop =========
async def run_loop(cams_json_path: Path, use_rclone: bool):
    cams_raw = json.loads(cams_json_path.read_text(encoding="utf-8"))
    cams: List[Dict[str, Any]] = [c for c in cams_raw if c.get("cam_id")]
    if not cams:
        logger.error("Không load được camera từ chunk.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _bl_load()
    last_sha: Dict[str, str] = {}

    limits = httpx.Limits(
        max_connections=CONCURRENCY * 2, max_keepalive_connections=CONCURRENCY
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "image/jpeg,image/*;q=0.8,*/*;q=0.5",
        "Origin": "https://giaothong.hochiminhcity.gov.vn",
        "Referer": "https://giaothong.hochiminhcity.gov.vn/",
    }

    async with httpx.AsyncClient(
        http2=True, headers=headers, limits=limits, timeout=None
    ) as client:
        sem = asyncio.Semaphore(CONCURRENCY)
        stop_evt = threading.Event()
        up_thread = start_uploader_thread(stop_evt) if use_rclone else None

        try:
            while True:
                tick = time.perf_counter()
                now_ts = time.time()

                # lọc blacklist (tôn trọng TTL)
                active_cams: List[Dict[str, Any]] = []
                bl_skipped: List[str] = []
                with _BL_LOCK:
                    for c in cams:
                        cid = (c.get("cam_id") or "").strip()
                        if not cid:
                            continue
                        exp = _blacklist.get(cid)
                        if exp is None:
                            active_cams.append(c)
                        else:
                            if exp == 0 or exp > now_ts:
                                bl_skipped.append(cid)
                            else:
                                _blacklist.pop(cid, None)
                                _bl_save()
                                active_cams.append(c)

                async def task(c: Dict[str, Any]):
                    async with sem:
                        await asyncio.sleep(
                            0.005 + (hash(c.get("cam_id")) % 25) / 1000.0
                        )
                        try:
                            return await process_one(c, client, last_sha)
                        except Exception:
                            return CamResult((c.get("cam_id") or "").strip(), "error")

                results: List[CamResult] = await asyncio.gather(
                    *[task(c) for c in active_cams], return_exceptions=False
                )

                ok_dup = sum(1 for r in results if r.status in ("ok", "dup"))
                dup_cnt = sum(1 for r in results if r.status == "dup")
                na_ids = [r.cam_id for r in results if r.status == "na"]
                err_ids = [
                    r.cam_id for r in results if r.status not in ("ok", "dup", "na")
                ]
                dur = time.perf_counter() - tick

                if na_ids:
                    logger.warning("[NA] : %s", "; ".join(na_ids))
                    for cid in na_ids:
                        if BLACKLIST_MODE == "na_immediate":
                            _bl_add(cid)
                        elif BLACKLIST_MODE == "na_after_streak":
                            s = na_streak.get(cid, 0) + 1
                            na_streak[cid] = s
                            if s >= BLACKLIST_STREAK:
                                _bl_add(cid)

                if err_ids:
                    logger.warning("[ERROR] : %s", "; ".join(err_ids))

                logger.info(
                    "[ROUND] ok+dup=%d/%d (dup=%d) | na=%d | bl_skip=%d | err=%d | queue=%d | %.2fs",
                    ok_dup,
                    len(cams),
                    dup_cnt,
                    len(na_ids),
                    len(bl_skipped),
                    len(err_ids),
                    _queue_size(),
                    dur,
                )

                # Chạy theo INTERVAL_SEC như trước (crawl lặp), uploader là event-driven nên vẫn sync ngay khi đủ tuổi
                try:
                    await asyncio.sleep(max(0.0, INTERVAL_SEC - dur))
                except asyncio.CancelledError:
                    logger.info("Cancel during sleep; breaking loop...")
                    break

        except asyncio.CancelledError:
            logger.info("Cancellation received; exiting loop...")

        finally:
            if use_rclone and up_thread:
                stop_evt.set()
                with _UPLOAD_COND:
                    _UPLOAD_COND.notify_all()
                if up_thread.is_alive():
                    up_thread.join(timeout=10.0)
                # flush batch còn lại (min_age=0 để lấy hết)
                batch = _dequeue_ready_paths(0)
                if batch:
                    rclone_move_list(batch)
                # dọn tổng
                rclone_move_all("Final-1", RCLONE_MIN_AGE_S)
                rclone_move_all("Final-2", 0)

                # retry nếu còn file local
                def _has_files(root: Path) -> bool:
                    return any(p.is_file() for p in root.rglob("*"))

                retries = 2
                while retries > 0 and _has_files(OUT_ROOT):
                    time.sleep(2)
                    rclone_move_all(f"Final-Retry-{3-retries}", 0)
                    retries -= 1
            _bl_save()


# ========= CLI =========
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    setup_logging()

    ap = argparse.ArgumentParser(
        description="HCMC traffic snapshot crawler (.jpg 512x288)"
    )
    ap.add_argument(
        "--chunk-file", required=True, help="JSON chứa danh sách cam (key cam_id)"
    )
    ap.add_argument(
        "--rclone", choices=["on", "off"], default="on", help="Bật/tắt đồng bộ rclone"
    )
    args = ap.parse_args()

    use_rclone = args.rclone == "on"
    if use_rclone:
        if not (RCLONE_EXE and RCLONE_REMOTE and GDRIVE_FOLDER_ID):
            raise SystemExit(
                "Missing rclone config in .env (RCLONE_EXE/RCLONE_REMOTE/GDRIVE_FOLDER_ID)"
            )

    cams_path = Path(args.chunk_file)
    try:
        asyncio.run(run_loop(cams_path, use_rclone=use_rclone))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: shutting down gracefully...")
    except asyncio.CancelledError:
        logger.info("Cancelled; exiting cleanly.")


if __name__ == "__main__":
    main()
