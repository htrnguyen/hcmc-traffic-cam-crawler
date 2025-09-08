# Bộ thu thập ảnh giao thông HCMC

Hệ thống tự động thu thập, xử lý và lưu trữ ảnh từ các camera giao thông tại TP.HCM, phục vụ nghiên cứu và phân tích dữ liệu.

## Tính năng chính

-   Thu thập ảnh song song, tốc độ cao
-   Tự động lọc ảnh lỗi, camera lỗi sẽ bị đưa vào blacklist
-   Tạo thư mục lưu trữ theo cấu trúc: Ngày (YYYYMMDD) → Camera → Giờ
-   Tự động nén ảnh theo từng giờ và upload lên Google Drive
-   Hỗ trợ dừng an toàn, retry khi gặp lỗi, ghi log đầy đủ

## Cấu trúc dữ liệu trên Google Drive

```
Google Drive Root/
├── 20240301/                                   # Thư mục ngày (YYYYMMDD)
│   ├── cam_001_ABC123/                         # Thư mục camera
│   │   ├── cam_001_ABC123_20240301_00.tar      # File nén giờ 00
│   │   ├── cam_001_ABC123_20240301_01.tar      # File nén giờ 01
│   │   └── ...                                 # 24 file nén mỗi ngày
│   └── cam_002_DEF456/                         # Camera khác
└── 20240302/
	└── cam_001_ABC123/
```

## Hướng dẫn sử dụng

### 1. Cài đặt

```bash
git clone https://github.com/htrnguyen/hcmc-traffic-cam-crawler.git
cd hcmc-traffic-cam-crawler
pip install -r requirements.txt
```

### 2. Thiết lập Google Drive

1. Tạo Google Cloud Project, bật Drive API
2. Tải file OAuth2 credentials (.json)
3. Tạo thư mục gốc trên Google Drive và lấy ID

### 3. Danh sách camera

Tạo file `camera_ids.csv`:

```csv
index,cam_id
1,ABC123
2,DEF456
3,GHI789
```

### 5. Chạy chương trình

```bash
python crawler.py
```