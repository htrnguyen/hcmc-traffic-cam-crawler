# HCMC Traffic Camera Crawler

Hệ thống thu thập hình ảnh từ camera giao thông công cộng, lưu trữ và phân tích dữ liệu.

## Tính năng

-   Thu thập hình ảnh từ nhiều camera đồng thời
-   Xử lý ảnh và loại bỏ duplicate
-   Logging chi tiết
-   Hỗ trợ cloud storage sync

## Cấu trúc dự án

```
├── hcmc_manycam_capture_save_images.py    # Script chính
├── requirements.txt                        # Dependencies Python
├── camera_catalog/                        # Dữ liệu camera
├── camera_catalog_chunks/                 # Phân chia camera thành chunks
├── scripts/                               # Tools tiền xử lý
├── rclone-v1.71.0-windows-amd64/         # Binary rclone
├── staging_images/
└── logs/
```

## Cài đặt

### Requirements

```bash
pip install -r requirements.txt
```

### Cấu hình Cloud Storage (tùy chọn)

Nếu muốn sync lên cloud storage, cần cài đặt rclone:

**Download rclone**: https://rclone.org/downloads/ (chọn Windows version)

```bash
# Cấu hình rclone remote (tên remote tùy ý, ví dụ: gdrv)
cd rclone-v1.71.0-windows-amd64
.\rclone.exe config create gdrv drive scope=drive

# Test kết nối
.\rclone.exe lsd gdrv:
```

**Lưu ý**: Tên remote có thể tùy ý (ví dụ: `gdrv`, `myremote`, `gdrive`...), nhưng phải nhất quán trong file `.env`.

### Environment Variables

Copy file `.env.example` thành `.env` và cập nhật các giá trị:

```bash
cp .env.example .env
# Hoặc trên Windows:
copy .env.example .env
```

Sau đó chỉnh sửa file `.env` với thông tin của bạn:

```bash
# Cloud storage - Bắt buộc nếu muốn sync lên Drive
# Lưu ý: RCLONE_REMOTE phải khớp với tên remote đã tạo ở bước trước
RCLONE_REMOTE=gdrv
GDRIVE_FOLDER_ID=your_actual_folder_id_here
RCLONE_EXE=.\rclone-v1.71.0-windows-amd64\rclone.exe

# Tuning - Có thể giữ mặc định
INTERVAL_SEC=60
CONCURRENCY=200
OUT_ROOT=staging_images
```

## Sử dụng

### Chạy crawler

```bash
python hcmc_manycam_capture_save_images.py --chunk-file camera_catalog_chunks/cams_chunk_000.json
```

### Preprocessing (nếu cần cập nhật data)

```bash
cd scripts
python parse_folder_ajax_response_full.py
python filter_imagehandler_clean.py
python build_chunks_from_clean.py
```

### Chạy multiple chunks (tùy chọn)

```bash
# Terminal 1
python hcmc_manycam_capture_save_images.py --chunk-file camera_catalog_chunks/cams_chunk_000.json

# Terminal 2
python hcmc_manycam_capture_save_images.py --chunk-file camera_catalog_chunks/cams_chunk_001.json
```

## Logs và Monitoring

-   **Console output**: Real-time status
-   **Log files**: `logs/capture.log` (auto-rotation)

```
2025-08-28 15:30:45 INFO [OK] : 95/100 | 12.34s
2025-08-28 15:30:45 WARNING [ERROR] : cam123; cam456
```

## Tính năng

-   **Async processing**: HTTP/2 với connection pooling
-   **Duplicate detection**: SHA256 hash comparison
-   **Error resilience**: Auto retry và graceful shutdown
-   **Memory efficient**: Stream processing
-   **Production logging**: Rotation và structured format

## License

MIT License - Dành cho mục đích nghiên cứu và giáo dục.
