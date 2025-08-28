import io
import json
import time
import hashlib
import asyncio
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from playwright.async_api import (
    async_playwright,
    APIRequestContext,
    TimeoutError as PWTimeout,
)

# ====== ĐƯỜNG DẪN ======
INPUT_JSON = Path("../camera_catalog/camera_catalog_imagehandler.json")
OUTPUT_JSON = Path("../camera_catalog/camera_catalog_imagehandler_clean.json")

# ====== DOMAIN & MẪU NHẬN DIỆN ======
BASE_IMG = "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx"
NA_SIZE = (284, 177)
NA_SHA256 = "1ef51d905a2e9d42678f4a37d5d54ecf9407dfd683ef39607782ddd64ab1aed5"
NA_AHASH_8x8 = "ffe3cbc3c3e781ff"
GOOD_SIZE = (512, 288)

# ====== CẤU HÌNH ======
CONCURRENCY = 12  # số tác vụ song song
REQ_TIMEOUT_MS = 12000  # timeout mỗi request (ms)
RETRIES = 2  # số lần retry thêm
PROGRESS_EVERY = 20  # in tiến độ mỗi N cam


# ====== TIỆN ÍCH ======
def now_ms() -> int:
    return int(time.time() * 1000)


def build_imagehandler_url(cam_id: str) -> str:
    q = urllib.parse.urlencode({"id": cam_id, "t": str(now_ms())})
    return f"{BASE_IMG}?{q}"


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ahash_8x8_hex(img: Image.Image) -> str:
    g = img.convert("L").resize((8, 8), Image.LANCZOS)
    pixels = list(g.getdata())
    avg = sum(pixels) / 64.0
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p >= avg else 0)
    return f"{bits:016x}"


def load_image_and_info(b: bytes) -> Optional[Tuple[Image.Image, Tuple[int, int]]]:
    try:
        img = Image.open(io.BytesIO(b))
        img.load()
        return img, img.size
    except Exception:
        return None


def is_not_available(img_size: Tuple[int, int], sha256: str, ahash: str) -> bool:
    return (img_size == NA_SIZE) or (sha256 == NA_SHA256) or (ahash == NA_AHASH_8x8)


def is_good_size(img_size: Tuple[int, int]) -> bool:
    return img_size == GOOD_SIZE


# ====== FETCH BẰNG PLAYWRIGHT ======
async def fetch_bytes(ctx: APIRequestContext, url: str) -> Optional[bytes]:
    try:
        resp = await ctx.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
                "Origin": "https://giaothong.hochiminhcity.gov.vn",
                "Referer": "https://giaothong.hochiminhcity.gov.vn/",
            },
            timeout=REQ_TIMEOUT_MS,
        )
        if resp.ok:
            b = await resp.body()
            if b:
                return b
    except PWTimeout:
        pass
    except Exception:
        pass
    return None


async def fetch_with_retry(ctx: APIRequestContext, cam_id: str) -> Optional[bytes]:
    attempts = 1 + max(0, RETRIES)
    for _ in range(attempts):
        b = await fetch_bytes(ctx, build_imagehandler_url(cam_id))
        if b:
            return b
        await asyncio.sleep(0.2)
    return None


# ====== XỬ LÝ 1 CAMERA ======
async def process_one(
    ctx: APIRequestContext, rec: Dict[str, Any], sem: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    async with sem:
        cam_id = (rec.get("cam_id") or "").strip()
        if not cam_id:
            return None

        b = await fetch_with_retry(ctx, cam_id)
        if not b:
            return None

        sha = sha256_hex(b)
        li = load_image_and_info(b)
        if not li:
            return None

        img, size = li
        ah = ahash_8x8_hex(img)

        if is_not_available(size, sha, ah):
            return None
        if not is_good_size(size):
            return None

        return {
            **rec,
            "image_meta": {
                "w": size[0],
                "h": size[1],
                "sha256": sha,
                "ahash_8x8": ah,
                "checked_at_ms": now_ms(),
                "checked_url": build_imagehandler_url(cam_id),
            },
        }


# ====== MAIN (ASYNC) ======
async def async_main() -> int:
    if not INPUT_JSON.exists():
        print(f"[ERR] Input not found: {INPUT_JSON}")
        return 2

    try:
        rows: List[Dict[str, Any]] = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERR] Failed to parse JSON: {e}")
        return 2

    total = len(rows)
    print(f"[INFO] Loaded {total} records from {INPUT_JSON}")

    kept: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async with async_playwright() as pw:
        request_ctx = await pw.request.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            ignore_https_errors=False,
        )

        tasks = [
            asyncio.create_task(process_one(request_ctx, rec, sem)) for rec in rows
        ]

        done = 0
        for coro in asyncio.as_completed(tasks):
            try:
                r = await coro
                if r is not None:
                    kept.append(r)
            except Exception:
                pass
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"[..] processed {done}/{total} -> kept={len(kept)}")

        await request_ctx.dispose()

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[DONE] Filtering ImageHandler snapshots (Playwright)")
    print(f"  total records : {total}")
    print(f"  kept good     : {len(kept)}")
    print(f"[OK] Output     : {OUTPUT_JSON}")
    return 0


def main():
    # >>> FIX CHÍNH Ở ĐÂY: dùng Proactor trên Windows để hỗ trợ subprocess <<<
    import sys, asyncio as _asyncio

    if sys.platform.startswith("win"):
        try:
            _asyncio.set_event_loop_policy(_asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            # Fallback (hiếm khi cần)
            pass

    rc = asyncio.run(async_main())
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
