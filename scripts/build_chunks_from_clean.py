import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Iterable, Tuple, Optional
from collections import defaultdict

# ====== CẤU HÌNH ======
INPUT_PATH = Path("../camera_catalog/camera_catalog_imagehandler_clean.json")
OUTPUT_DIR = Path("../camera_catalog_chunks")
CHUNK_SIZE = 600
STRATEGY = "district_interleave"  
SEED = "stable-seed-2025-08-25" 


# ====== TIỆN ÍCH ======
def stable_hash_int(s: str, seed: str) -> int:
    h = hashlib.sha256(f"{seed}|{s}".encode("utf-8")).hexdigest()
    # dùng 64-bit đầu để có số nguyên ổn định
    return int(h[:16], 16)


def chunks(seq: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def unique_by(seq: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for x in seq:
        k = x.get(key)
        if not k:  # bỏ record thiếu key quan trọng
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def light_cam(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Chỉ giữ trường cần dùng cho runner/giám sát."""
    return {
        "cam_id": rec.get("cam_id"),
        "code": rec.get("code"),
        "title": rec.get("title"),
        "district": rec.get("district"),
        "image_url": rec.get(
            "image_url"
        ),  # có thể bỏ và build ở runtime bằng cam_id + t=
    }


# ====== CHIẾN LƯỢC PHÂN PHỐI ======
def strategy_plain(cams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # giữ nguyên thứ tự input
    return cams


def strategy_hash_shard(cams: List[Dict[str, Any]], seed: str) -> List[Dict[str, Any]]:
    # sắp xếp ổn định bằng hash(cam_id)
    def key(rec: Dict[str, Any]) -> Tuple[int, str]:
        cam_id = rec.get("cam_id") or ""
        return (stable_hash_int(cam_id, seed), cam_id)

    return sorted(cams, key=key)


def strategy_district_interleave(
    cams: List[Dict[str, Any]], seed: str
) -> List[Dict[str, Any]]:
    """
    1) Gom theo district (None -> "UNKNOWN").
    2) Sắp xếp từng nhóm theo hash(cam_id) để ổn định.
    3) Interleave (round-robin) giữa các nhóm để mỗi chunk có mix nhiều quận.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in cams:
        d = rec.get("district") or "UNKNOWN"
        groups[str(d)].append(rec)

    # sort nhóm theo tên để ổn định vòng lặp
    district_keys = sorted(groups.keys())

    # sort từng group theo hash(cam_id) => ổn định nhưng "ngẫu nhiên"
    for d in district_keys:
        groups[d].sort(
            key=lambda r: (
                stable_hash_int(r.get("cam_id") or "", seed),
                r.get("cam_id") or "",
            )
        )

    # interleave
    # lấy lần lượt phần tử 0 từ mọi nhóm, rồi phần tử 1, ...
    out: List[Dict[str, Any]] = []
    # tính max chiều dài nhóm
    max_len = max(len(groups[d]) for d in district_keys) if district_keys else 0
    for i in range(max_len):
        for d in district_keys:
            if i < len(groups[d]):
                out.append(groups[d][i])
    return out


# ====== MAIN ======
def main():
    if not INPUT_PATH.exists():
        print(f"[ERR] Input not found: {INPUT_PATH}")
        return

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    # lọc phần tử hợp lệ + unique cam_id
    raw_cams = [c for c in data if isinstance(c, dict) and c.get("cam_id")]
    cams = unique_by(raw_cams, key="cam_id")

    total = len(cams)
    print(f"[INFO] Total cams (unique by cam_id): {total}")

    # chọn chiến lược
    if STRATEGY == "district_interleave":
        ordered = strategy_district_interleave(cams, SEED)
    elif STRATEGY == "hash_shard":
        ordered = strategy_hash_shard(cams, SEED)
    else:
        ordered = strategy_plain(cams)

    # rút gọn trường & chia chunk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    light_ordered = [light_cam(r) for r in ordered]

    count_files = 0
    for idx, block in enumerate(chunks(light_ordered, CHUNK_SIZE)):
        out_path = OUTPUT_DIR / f"cams_chunk_{idx:03d}.json"
        out_path.write_text(
            json.dumps(block, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] {out_path}  ({len(block)} cams)")
        count_files += 1

    # thống kê nhanh theo district
    dist_count: Dict[str, int] = defaultdict(int)
    for r in light_ordered:
        d = str(r.get("district") or "UNKNOWN")
        dist_count[d] += 1
    dist_stats = sorted(dist_count.items(), key=lambda x: (-x[1], x[0]))

    index = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": str(INPUT_PATH),
        "chunk_size": CHUNK_SIZE,
        "strategy": STRATEGY,
        "seed": SEED,
        "total_cameras": total,
        "total_files": count_files,
        "district_stats": [{"district": k, "count": v} for k, v in dist_stats],
    }
    (OUTPUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] Đã tạo {count_files} file trong thư mục {OUTPUT_DIR}/")
    print(f"[OK] Ghi chỉ mục: {OUTPUT_DIR/'index.json'}")


if __name__ == "__main__":
    main()
