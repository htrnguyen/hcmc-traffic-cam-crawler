"""
Crawler tải ảnh từ camera giao thông TPHCM.
Lưu ảnh dạng BLOB vào SQLite theo cam/ngày.
Đồng bộ thời điểm trong mỗi vòng chụp, xử lý song song.
Phát hiện & blacklist cam lỗi/NA liên tiếp.
Sync files ngày hôm trước lên Google Drive và xóa local.
"""

import argparse
import asyncio
import hashlib
import logging
import shutil
import sqlite3
import time
from datetime import datetime
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
import yaml
from PIL import Image

from sync_daily import DailySync, RcloneConfig

try:
    from zoneinfo import ZoneInfo

    VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
except ImportError:
    from datetime import timedelta, timezone

    VN_TZ = timezone(timedelta(hours=7))


def setup_logging(config):
    """Setup rotating log files by date"""
    log_dir = Path(config["LOG_DIR"])
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("crawler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Daily rotating file handler
    today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    log_file = log_dir / f"crawler_{today}.log"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger, str(log_file)


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ids():
    ids = []
    with open("camera_ids.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and line not in ids:
                ids.append(line)
    return ids


def load_blacklist(filepath):
    blacklist = set()
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    blacklist.add(line)
    except FileNotFoundError:
        pass
    return blacklist


def cleanup_camera_folder(cam_id, config):
    """Remove camera folder when blacklisted"""
    try:
        cam_path = Path(config["OUT_ROOT"]) / cam_id
        if cam_path.exists():
            shutil.rmtree(cam_path)
            return True
    except Exception:
        pass
    return False


def save_blacklist(filepath, blacklist):
    with open(filepath, "w") as f:
        for cam_id in sorted(blacklist):
            f.write(f"{cam_id}\n")


def is_na(img_bytes, config):
    det = config["image_detection"]

    # Check size
    if len(img_bytes) < det["min_file_size"]:
        return True

    # Check SHA256
    sha256_hash = hashlib.sha256(img_bytes).hexdigest()
    if sha256_hash == det["na_sha256"]:
        return True

    # Check dimensions and aHash
    try:
        img = Image.open(BytesIO(img_bytes))
        if [img.width, img.height] == det["na_size"]:
            # Calculate aHash 8x8
            img_gray = img.convert("L").resize((8, 8), Image.LANCZOS)
            pixels = list(img_gray.getdata())
            avg = sum(pixels) // len(pixels)
            bits = ["1" if p >= avg else "0" for p in pixels]
            ahash = hex(int("".join(bits), 2))[2:].zfill(16)
            if ahash == det["na_ahash_8x8"]:
                return True
    except Exception:
        pass

    return False


def compress_image(img_bytes):
    """Compress image if it's too large, otherwise return as-is"""
    try:
        if len(img_bytes) > 500_000:  # > 500KB
            img = Image.open(BytesIO(img_bytes))
            if img.format in ["JPEG", "JPG"]:
                # Re-compress JPEG with lower quality
                output = BytesIO()
                img.save(output, format="JPEG", quality=75, optimize=True)
                return output.getvalue()
            elif img.format == "PNG":
                # Convert PNG to JPEG for smaller size
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                output = BytesIO()
                img.save(output, format="JPEG", quality=80, optimize=True)
                return output.getvalue()
        return img_bytes
    except Exception:
        return img_bytes


def open_db_for(cam_id, date_str, config):
    db_dir = Path(config["OUT_ROOT"]) / cam_id
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / f"{date_str}.sqlite"
    conn = sqlite3.connect(str(db_path))

    # PRAGMA optimizations for SQLite
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
    conn.execute("PRAGMA page_size=4096")
    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")

    # Create schema - use BLOB for images
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS frames (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id     TEXT    NOT NULL,
            ts_vn_iso  TEXT    NOT NULL,
            epoch_ms   INTEGER NOT NULL,
            url        TEXT    NOT NULL,
            status     INTEGER NOT NULL,
            img_blob   BLOB    NOT NULL,
            img_size   INTEGER NOT NULL
        )
    """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_frames_cam_epoch
        ON frames(cam_id, epoch_ms)
    """
    )
    conn.commit()
    return conn


def insert_frame(conn, cam_id, ts_vn_iso, epoch_ms, url, status, img_blob):
    conn.execute(
        """
        INSERT OR IGNORE INTO frames (cam_id, ts_vn_iso, epoch_ms, url, status, img_blob, img_size)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (cam_id, ts_vn_iso, epoch_ms, url, status, img_blob, len(img_blob)),
    )
    conn.commit()


async def fetch_one(client, semaphore, cam_id, config):
    """Fetch one camera with individual timestamp"""
    # Each request gets its own timestamp
    now_vn = datetime.now(VN_TZ)
    epoch_ms = int(now_vn.timestamp() * 1000)

    # Add small random variation
    import random

    epoch_ms += random.randint(0, 999)

    url = f"https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={cam_id}&t={epoch_ms}"
    ts_vn_iso = now_vn.strftime("%Y-%m-%d_%H-%M-%S")

    async with semaphore:
        for attempt in range(config["RETRIES"] + 1):
            try:
                # Add headers to mimic browser
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
                    "Cache-Control": "no-cache",
                    "Referer": "https://giaothong.hochiminhcity.gov.vn/",
                }

                resp = await client.get(
                    url,
                    timeout=config["TIMEOUT_S"],
                    headers=headers,
                    follow_redirects=True,
                )

                if resp.status_code == 200:
                    img_bytes = resp.content

                    # Quick validation
                    if len(img_bytes) < 100:
                        if attempt == config["RETRIES"]:
                            return (
                                cam_id,
                                "err",
                                None,
                                url,
                                resp.status_code,
                                epoch_ms,
                                ts_vn_iso,
                            )
                        await asyncio.sleep(0.3 * (attempt + 1))
                        continue

                    if is_na(img_bytes, config):
                        return (
                            cam_id,
                            "na",
                            None,
                            url,
                            resp.status_code,
                            epoch_ms,
                            ts_vn_iso,
                        )

                    # Compress image if needed
                    compressed_img = compress_image(img_bytes)
                    return (
                        cam_id,
                        "ok",
                        compressed_img,
                        url,
                        resp.status_code,
                        epoch_ms,
                        ts_vn_iso,
                    )
                else:
                    if attempt == config["RETRIES"]:
                        return (
                            cam_id,
                            "err",
                            None,
                            url,
                            resp.status_code,
                            epoch_ms,
                            ts_vn_iso,
                        )
                    await asyncio.sleep(0.3 * (attempt + 1))

            except asyncio.CancelledError:
                # Handle graceful shutdown
                return cam_id, "cancelled", None, url, 0, epoch_ms, ts_vn_iso
            except Exception:
                if attempt == config["RETRIES"]:
                    return cam_id, "err", None, url, 0, epoch_ms, ts_vn_iso
                await asyncio.sleep(0.3 * (attempt + 1))


async def run_round(
    client, active_ids, config, logger, error_streaks, blacklist, current_date
):
    now_vn = datetime.now(VN_TZ)
    new_date = now_vn.strftime("%Y-%m-%d")

    # Check if date changed
    if new_date != current_date:
        logger.info(f"Date changed from {current_date} to {new_date}")
        return new_date

    semaphore = asyncio.Semaphore(config["CONCURRENCY"])
    start_time = time.time()

    # Send all requests concurrently with individual timestamps
    tasks = [fetch_one(client, semaphore, cam_id, config) for cam_id in active_ids]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info("Round cancelled - shutting down gracefully")
        return current_date

    stats = {"ok": 0, "na": 0, "err": 0, "blacklisted": 0, "cancelled": 0}
    new_blacklisted = []
    cleaned_folders = []

    for result in results:
        if isinstance(result, Exception):
            stats["err"] += 1
            continue

        cam_id, status, img_data, url, http_status, epoch_ms, ts_vn_iso = result

        if status == "cancelled":
            stats["cancelled"] += 1
            continue
        elif status == "ok":
            # Save to database
            date_str = new_date
            conn = open_db_for(cam_id, date_str, config)
            insert_frame(conn, cam_id, ts_vn_iso, epoch_ms, url, http_status, img_data)
            conn.close()

            error_streaks[cam_id] = 0
            stats["ok"] += 1

        elif status in ["na", "err"]:
            # Immediate blacklist for NA or error
            if cam_id not in blacklist:
                blacklist.add(cam_id)
                new_blacklisted.append(cam_id)
                stats["blacklisted"] += 1

                # Clean up folder
                if cleanup_camera_folder(cam_id, config):
                    cleaned_folders.append(cam_id)

            if status == "na":
                stats["na"] += 1
            else:
                stats["err"] += 1

        # Progress update
        processed = stats["ok"] + stats["na"] + stats["err"] + stats["cancelled"]
        print(
            f"\r{processed}/{len(active_ids)} | OK:{stats['ok']} NA:{stats['na']} ERR:{stats['err']}",
            end="",
            flush=True,
        )

    print()  # New line

    # Update blacklist file
    if new_blacklisted:
        save_blacklist(config["BLACKLIST_FILE"], blacklist)
        logger.warning(
            f"Blacklisted {len(new_blacklisted)} cameras, cleaned {len(cleaned_folders)} folders"
        )

    elapsed = time.time() - start_time
    total_processed = stats["ok"] + stats["na"] + stats["err"] + stats["cancelled"]
    rate = total_processed / elapsed if elapsed > 0 else 0

    # Compact summary
    summary = f"TOTAL:{len(active_ids)} OK:{stats['ok']} NA:{stats['na']} ERR:{stats['err']} BL+:{stats['blacklisted']} | {elapsed:.1f}s ({rate:.1f}/s)"
    print(f"{now_vn.strftime('%H:%M:%S')} {summary}")
    logger.info(summary)

    return current_date


async def main():
    """Main crawler function với sync tối ưu"""
    config = load_config()

    # Load data
    all_ids = load_ids()
    blacklist = load_blacklist(config["BLACKLIST_FILE"])
    error_streaks = {}

    current_date = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    logger, log_file = setup_logging(config)

    print(f"🚀 CRAWLER STARTED")
    print(f"📅 Date: {current_date}")
    print(f"📷 Cameras: {len(all_ids)} total, {len(blacklist)} blacklisted")
    print(f"📝 Logging to: {log_file}")
    print(
        f"⚙️ Concurrency: {config['CONCURRENCY']}, Interval: {config['INTERVAL_SEC']}s"
    )
    print("-" * 60)

    logger.info(
        f"CRAWLER STARTED - Date:{current_date} Total:{len(all_ids)} Blacklisted:{len(blacklist)}"
    )

    # === Daily rclone sync (VN timezone) ===
    rclone_cfg = config.get("rclone") or {}
    sync_cfg = config.get("sync") or {}

    # Log rclone ra logs/rclone_out.log
    rclone_log = Path(config["LOG_DIR"]) / f"rclone_out.log"
    rclone_extra = ["--log-file", str(rclone_log)]

    # Khởi tạo DailySync với logger
    daily_sync = DailySync(
        src_dir=config["OUT_ROOT"],
        rclone=RcloneConfig(
            exe=rclone_cfg.get("exe", "rclone"),
            remote=rclone_cfg.get("remote", "gdrv"),
            drive_folder_id=rclone_cfg.get("drive_folder_id", ""),
            mode=rclone_cfg.get("mode", "copy"),
            extra_args=rclone_extra,
        ),
        sync_at=sync_cfg.get("at", "00:05"),
        tz_name=sync_cfg.get("tz", "Asia/Ho_Chi_Minh"),
        logger=logger,
    )

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=300),
        timeout=httpx.Timeout(config["TIMEOUT_S"], connect=3.0),
        verify=False,
    ) as client:
        try:
            round_count = 0
            while True:
                active_ids = [cid for cid in all_ids if cid not in blacklist]

                if not active_ids:
                    print("⚠️ No active cameras remaining!")
                    logger.info("All cameras blacklisted - stopping")
                    break

                round_count += 1
                print(f"\n🔄 Round #{round_count} - Active: {len(active_ids)}")

                # Show sync status
                sync_status = daily_sync.get_status()
                if sync_status["sync_running"]:
                    print("🔄 [SYNC] Running in background...")

                # Run round and check for date change
                result_date = await run_round(
                    client,
                    active_ids,
                    config,
                    logger,
                    error_streaks,
                    blacklist,
                    current_date,
                )

                if result_date != current_date:
                    # Date changed - setup new logger
                    current_date = result_date
                    logger, log_file = setup_logging(config)
                    print(f"\n📅 NEW DAY: {current_date}")
                    print(f"📝 New log file: {log_file}")
                    logger.info(
                        f"NEW DAY STARTED - Date:{current_date} Active:{len(active_ids)}"
                    )
                    round_count = 0

                    # Update logger cho DailySync
                    daily_sync.logger = logger

                # Gọi tick() trước khi ngủ (rất nhẹ; chỉ chạy thật khi đến giờ & chưa chạy hôm nay)
                try:
                    if daily_sync.tick():
                        print("🚀 [SYNC] Started background sync for yesterday's files")
                        logger.info(
                            "[SYNC] Started background sync for yesterday's files"
                        )
                except Exception as e:
                    print(f"[SYNC] tick error: {e}")
                    logger.error(f"[SYNC] tick error: {e}")

                # Ngủ giữa các vòng; nếu Ctrl+C trong lúc ngủ -> thoát êm
                try:
                    await asyncio.sleep(config["INTERVAL_SEC"])
                except asyncio.CancelledError:
                    break

        except KeyboardInterrupt:
            print(f"\n⏹️ STOPPING CRAWLER...")

            # Cancel any running tasks
            tasks = [
                task
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()

            # Wait a bit for cleanup
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            print(f"🛑 CRAWLER STOPPED SAFELY")
            logger.info("CRAWLER STOPPED - KeyboardInterrupt")
            return 0


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Exited by Ctrl+C (clean).")
