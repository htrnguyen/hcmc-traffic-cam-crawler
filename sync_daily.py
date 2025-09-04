# sync_daily.py
from __future__ import annotations
import subprocess, sys, shutil
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from zoneinfo import ZoneInfo  # Py3.9+
except Exception:
    ZoneInfo = None


def _tz(name: str):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass

    # Fallback +07:00
    class _FixedTZ:
        def utcoffset(self, dt):
            return timedelta(hours=7)

        def tzname(self, dt):
            return name

        def dst(self, dt):
            return timedelta(0)

    return _FixedTZ()


@dataclass
class RcloneConfig:
    exe: str
    remote: str
    drive_folder_id: str
    mode: str = "copy"  # "copy" an toàn, "sync" phản chiếu
    extra_args: list[str] | None = None


class DailySync:
    """
    Sync files từ ngày hôm trước lên Google Drive và xóa local sau khi thành công.
    Chạy song song với crawler để không interrupt việc thu thập dữ liệu.
    """

    def __init__(
        self,
        src_dir,
        rclone: RcloneConfig,
        sync_at="00:05",
        tz_name="Asia/Ho_Chi_Minh",
        logger=None,
    ):
        self.src_dir = Path(src_dir)
        self.rclone = rclone
        self.sync_at = sync_at
        self.tz = _tz(tz_name)
        self.logger = logger or logging.getLogger(__name__)
        self._last_run_local_date: str | None = None
        self._sync_running = False
        self._sync_lock = threading.Lock()

        h, m = map(int, sync_at.split(":"))
        self._sync_time = time(hour=h, minute=m)

    def _now_local(self) -> datetime:
        return datetime.now(self.tz)

    def _today_key(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    def _yesterday_key(self, dt: datetime) -> str:
        yesterday = dt - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")

    def _target_dt_today(self, dt: datetime) -> datetime:
        return datetime(
            dt.year,
            dt.month,
            dt.day,
            self._sync_time.hour,
            self._sync_time.minute,
            tzinfo=self.tz,
        )

    def should_run_now(self) -> bool:
        with self._sync_lock:
            if self._sync_running:
                return False

            now = self._now_local()
            today = self._today_key(now)

            # Chỉ chạy 1 lần mỗi ngày
            if self._last_run_local_date == today:
                return False

            return now >= self._target_dt_today(now)

    def tick(self) -> bool:
        """Non-blocking check và chạy sync trong thread riêng"""
        if not self.should_run_now():
            return False

        # Chạy sync trong background thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self._async_sync)

        return True

    def force_run(self) -> int:
        """Force run sync ngay lập tức (blocking)"""
        return self._run_sync_with_cleanup()

    def _async_sync(self):
        """Chạy sync trong background thread"""
        with self._sync_lock:
            self._sync_running = True

        try:
            rc = self._run_sync_with_cleanup()
            if rc == 0:
                self._last_run_local_date = self._today_key(self._now_local())
        finally:
            with self._sync_lock:
                self._sync_running = False

    def _run_sync_with_cleanup(self) -> int:
        """Sync files ngày hôm trước và cleanup local files"""
        now = self._now_local()
        yesterday = self._yesterday_key(now)

        # Tìm tất cả thư mục camera có file ngày hôm trước
        yesterday_files = []
        camera_dirs = []

        for cam_dir in self.src_dir.iterdir():
            if cam_dir.is_dir():
                sqlite_file = cam_dir / f"{yesterday}.sqlite"
                if sqlite_file.exists():
                    yesterday_files.append(sqlite_file)
                    camera_dirs.append(cam_dir)

        if not yesterday_files:
            self.logger.info(f"[SYNC] No files found for {yesterday}")
            return 0

        self.logger.info(f"[SYNC] Found {len(yesterday_files)} files for {yesterday}")

        # Sync từng camera directory riêng biệt để tối ưu tốc độ
        success_count = 0
        failed_dirs = []

        for cam_dir in camera_dirs:
            sqlite_file = cam_dir / f"{yesterday}.sqlite"
            if self._sync_single_camera(cam_dir, sqlite_file, yesterday):
                success_count += 1
            else:
                failed_dirs.append(cam_dir.name)

        if failed_dirs:
            self.logger.error(
                f"[SYNC] Failed to sync {len(failed_dirs)} cameras: {failed_dirs[:10]}..."
            )
            return 1
        else:
            self.logger.info(
                f"[SYNC] Successfully synced and cleaned {success_count} cameras for {yesterday}"
            )
            return 0

    def _sync_single_camera(
        self, cam_dir: Path, sqlite_file: Path, date_str: str
    ) -> bool:
        """Sync một camera directory và xóa file local nếu thành công"""
        try:
            # Tạo temporary directory để sync chỉ file cụ thể
            temp_structure = Path(f"temp_sync_{cam_dir.name}_{date_str}")
            temp_cam_dir = temp_structure / cam_dir.name
            temp_cam_dir.mkdir(parents=True, exist_ok=True)

            # Copy file sqlite vào temp structure
            temp_file = temp_cam_dir / sqlite_file.name
            shutil.copy2(sqlite_file, temp_file)

            # Sync temp directory
            rc = self._run_rclone_sync(str(temp_structure))

            if rc == 0:
                # Sync thành công -> xóa file gốc
                sqlite_file.unlink()
                self.logger.info(
                    f"[SYNC] ✓ {cam_dir.name}/{sqlite_file.name} -> deleted local"
                )

                # Xóa temp directory
                shutil.rmtree(temp_structure, ignore_errors=True)
                return True
            else:
                self.logger.error(
                    f"[SYNC] ✗ Failed to sync {cam_dir.name}/{sqlite_file.name}"
                )
                shutil.rmtree(temp_structure, ignore_errors=True)
                return False

        except Exception as e:
            self.logger.error(f"[SYNC] Error syncing {cam_dir.name}: {e}")
            return False

    def _run_rclone_sync(self, src_path: str) -> int:
        """Chạy rclone với config tối ưu cho tốc độ (kept for compatibility)"""
        return self._run_rclone_sync_sequential(src_path, 1, 1)

    def get_status(self) -> dict:
        """Trả về trạng thái hiện tại của sync"""
        with self._sync_lock:
            return {
                "sync_running": self._sync_running,
                "last_run_date": self._last_run_local_date,
                "next_sync_time": self.sync_at,
                "timezone": str(self.tz),
            }
