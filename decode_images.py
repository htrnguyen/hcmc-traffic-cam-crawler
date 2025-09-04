#!/usr/bin/env python3
"""
Utility script để decode ảnh từ SQLite database
"""

import sqlite3
import sys
from pathlib import Path


def list_cameras(out_root="out"):
    """Liệt kê tất cả camera có dữ liệu"""
    out_path = Path(out_root)
    if not out_path.exists():
        print(f"Directory {out_root} không tồn tại")
        return []
    
    cameras = []
    for cam_dir in out_path.iterdir():
        if cam_dir.is_dir():
            db_files = list(cam_dir.glob("*.sqlite"))
            if db_files:
                cameras.append((cam_dir.name, len(db_files)))
    
    return cameras


def list_dates(cam_id, out_root="out"):
    """Liệt kê các ngày có dữ liệu của camera"""
    cam_path = Path(out_root) / cam_id
    if not cam_path.exists():
        return []
    
    dates = []
    for db_file in cam_path.glob("*.sqlite"):
        dates.append(db_file.stem)  # filename without .sqlite
    
    return sorted(dates)


def export_images(cam_id, date_str, output_dir=None, out_root="out", limit=None):
    """Export ảnh từ database ra file"""
    db_path = Path(out_root) / cam_id / f"{date_str}.sqlite"
    if not db_path.exists():
        print(f"Database {db_path} không tồn tại")
        return
    
    if output_dir is None:
        output_dir = f"exported_{cam_id}_{date_str}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    
    query = "SELECT ts_vn_iso, img_blob, img_size FROM frames WHERE cam_id=? ORDER BY epoch_ms"
    params = [cam_id]
    
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    
    cursor = conn.execute(query, params)
    
    count = 0
    total_size = 0
    
    for row in cursor:
        ts_vn_iso, img_blob, img_size = row
        
        # Detect image format from header
        if img_blob[:3] == b'\xff\xd8\xff':
            ext = 'jpg'
        elif img_blob[:8] == b'\x89PNG\r\n\x1a\n':
            ext = 'png'
        elif img_blob[:6] in (b'GIF87a', b'GIF89a'):
            ext = 'gif'
        else:
            ext = 'jpg'  # default
        
        filename = f"{ts_vn_iso}.{ext}"
        filepath = output_path / filename
        
        with open(filepath, 'wb') as f:
            f.write(img_blob)
        
        count += 1
        total_size += img_size
        
        if count % 10 == 0:
            print(f"Exported {count} images...")
    
    conn.close()
    
    print(f"✓ Exported {count} images to {output_path}")
    print(f"  Total size: {total_size // 1024:.1f} KB")


def show_stats(cam_id, date_str, out_root="out"):
    """Hiển thị thống kê database"""
    db_path = Path(out_root) / cam_id / f"{date_str}.sqlite"
    if not db_path.exists():
        print(f"Database {db_path} không tồn tại")
        return
    
    conn = sqlite3.connect(str(db_path))
    
    # Get basic stats
    cursor = conn.execute("SELECT COUNT(*), MIN(epoch_ms), MAX(epoch_ms), AVG(img_size), SUM(img_size) FROM frames WHERE cam_id=?", (cam_id,))
    count, min_epoch, max_epoch, avg_size, total_size = cursor.fetchone()
    
    if count == 0:
        print("Không có dữ liệu")
        return
    
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
    except ImportError:
        from datetime import timezone, timedelta
        VN_TZ = timezone(timedelta(hours=7))
    
    start_time = datetime.fromtimestamp(min_epoch/1000, VN_TZ)
    end_time = datetime.fromtimestamp(max_epoch/1000, VN_TZ)
    
    print(f"📊 Stats for {cam_id} on {date_str}:")
    print(f"  Images: {count}")
    print(f"  Time range: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
    print(f"  Avg size: {avg_size:.0f} bytes ({avg_size/1024:.1f} KB)")
    print(f"  Total size: {total_size/1024:.1f} KB ({total_size/1024/1024:.1f} MB)")
    print(f"  DB file size: {db_path.stat().st_size/1024:.1f} KB")
    
    # Get hourly distribution
    cursor = conn.execute("""
        SELECT substr(ts_vn_iso, 12, 2) as hour, COUNT(*) 
        FROM frames WHERE cam_id=? 
        GROUP BY hour ORDER BY hour
    """, (cam_id,))
    
    print("  Hourly distribution:")
    for hour, cnt in cursor:
        print(f"    {hour}h: {cnt} images")
    
    conn.close()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python decode_images.py list                     # List all cameras")
        print("  python decode_images.py dates <cam_id>           # List dates for camera")
        print("  python decode_images.py stats <cam_id> <date>    # Show statistics")
        print("  python decode_images.py export <cam_id> <date> [output_dir] [limit]")
        print()
        print("Examples:")
        print("  python decode_images.py list")
        print("  python decode_images.py dates 59ca2d9d02eb490011a0a3f0")
        print("  python decode_images.py stats 59ca2d9d02eb490011a0a3f0 2025-09-01")
        print("  python decode_images.py export 59ca2d9d02eb490011a0a3f0 2025-09-01")
        print("  python decode_images.py export 59ca2d9d02eb490011a0a3f0 2025-09-01 my_images 10")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        cameras = list_cameras()
        if not cameras:
            print("Không tìm thấy camera nào")
        else:
            print("📷 Available cameras:")
            for cam_id, db_count in cameras:
                print(f"  {cam_id} ({db_count} days)")
    
    elif command == "dates":
        if len(sys.argv) < 3:
            print("Usage: python decode_images.py dates <cam_id>")
            return
        
        cam_id = sys.argv[2]
        dates = list_dates(cam_id)
        if not dates:
            print(f"Không tìm thấy dữ liệu cho camera {cam_id}")
        else:
            print(f"📅 Available dates for {cam_id}:")
            for date in dates:
                print(f"  {date}")
    
    elif command == "stats":
        if len(sys.argv) < 4:
            print("Usage: python decode_images.py stats <cam_id> <date>")
            return
        
        cam_id = sys.argv[2]
        date_str = sys.argv[3]
        show_stats(cam_id, date_str)
    
    elif command == "export":
        if len(sys.argv) < 4:
            print("Usage: python decode_images.py export <cam_id> <date> [output_dir] [limit]")
            return
        
        cam_id = sys.argv[2]
        date_str = sys.argv[3]
        output_dir = sys.argv[4] if len(sys.argv) > 4 else None
        limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
        
        export_images(cam_id, date_str, output_dir, limit=limit)
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()