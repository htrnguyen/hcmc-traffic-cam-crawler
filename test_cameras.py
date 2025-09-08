#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import hashlib
import io
import time
from pathlib import Path

import requests
from PIL import Image


def now_epoch_ms():
    return int(time.time() * 1000)


def ahash_8x8_hex(img: Image.Image) -> str:
    g = img.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << (63 - i)
    return f"{bits:016x}"


def test_camera(cam_id: str, endpoint_template: str, error_config: dict) -> dict:
    """Test a single camera and return result"""
    url = endpoint_template.format(cam_id=cam_id, epoch_ms=now_epoch_ms())

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "Accept": "image/*",
                "User-Agent": "HCMC-Traffic-CamTest/1.0",
                "Cache-Control": "no-cache",
            },
        )

        if response.status_code != 200:
            return {
                "cam_id": cam_id,
                "status": "HTTP_ERROR",
                "error": f"HTTP {response.status_code}",
                "size": 0,
            }

        raw = response.content
        size_kb = len(raw) / 1024

        # Check if error image
        is_error = False
        error_reason = None

        # Size check
        if len(raw) < 1000:
            is_error = True
            error_reason = "TOO_SMALL"

        # SHA256 check
        if not is_error:
            sha = hashlib.sha256(raw).hexdigest().lower()
            if sha == error_config["na_sha256"].lower():
                is_error = True
                error_reason = "ERROR_SHA256"

        # Image checks
        if not is_error:
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    # Size check
                    if im.size == tuple(error_config["na_size"]):
                        is_error = True
                        error_reason = "ERROR_SIZE"
                    # AHash check
                    elif (
                        ahash_8x8_hex(im).lower()
                        == error_config["na_ahash_8x8"].lower()
                    ):
                        is_error = True
                        error_reason = "ERROR_AHASH"
                    else:
                        # Valid image
                        return {
                            "cam_id": cam_id,
                            "status": "OK",
                            "size_kb": round(size_kb, 1),
                            "dimensions": f"{im.size[0]}x{im.size[1]}",
                            "mode": im.mode,
                        }
            except Exception as e:
                is_error = True
                error_reason = f"IMAGE_ERROR: {str(e)}"

        return {
            "cam_id": cam_id,
            "status": "ERROR_IMAGE",
            "error": error_reason,
            "size_kb": round(size_kb, 1),
        }

    except Exception as e:
        return {"cam_id": cam_id, "status": "NETWORK_ERROR", "error": str(e), "size": 0}


def test_cameras_from_csv(csv_file: str = "camera_ids.csv", sample_size: int = 10):
    """Test a sample of cameras from CSV file"""

    # Load config
    try:
        import yaml

        config_path = Path("config.yaml")
        if config_path.exists():
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            print("config.yaml not found, using default endpoint")
            cfg = {
                "endpoint_template": "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={cam_id}&t={epoch_ms}",
                "na_size": [284, 177],
                "na_sha256": "1ef51d905a2e9d42678f4a37d5d54ecf9407dfd683ef39607782ddd64ab1aed5",
                "na_ahash_8x8": "ffe3cbc3c3e781ff",
            }
    except ImportError:
        print("PyYAML not installed, using default config")
        cfg = {
            "endpoint_template": "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={cam_id}&t={epoch_ms}",
            "na_size": [284, 177],
            "na_sha256": "1ef51d905a2e9d42678f4a37d5d54ecf9407dfd683ef39607782ddd64ab1aed5",
            "na_ahash_8x8": "ffe3cbc3c3e781ff",
        }

    # Load cameras
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Camera CSV file not found: {csv_file}")
        return

    cameras = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "cam_id" in row:
                    cam_id = row["cam_id"].strip()
                    if cam_id and cam_id != "cam_id":
                        cameras.append(cam_id)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not cameras:
        print("No cameras found in CSV")
        return

    # Test sample
    import random

    test_cameras = (
        cameras[:sample_size]
        if sample_size >= len(cameras)
        else random.sample(cameras, sample_size)
    )

    print(f"Testing {len(test_cameras)} cameras from {len(cameras)} total...")
    print(f"Endpoint: {cfg['endpoint_template']}")
    print("-" * 80)

    ok_count = 0
    error_count = 0
    network_error_count = 0

    for i, cam_id in enumerate(test_cameras, 1):
        print(f"[{i:2d}/{len(test_cameras)}] Testing {cam_id}... ", end="", flush=True)

        result = test_camera(cam_id, cfg["endpoint_template"], cfg)

        if result["status"] == "OK":
            print(
                f"✓ OK ({result['size_kb']}KB, {result['dimensions']}, {result['mode']})"
            )
            ok_count += 1
        elif result["status"] == "ERROR_IMAGE":
            print(f"✗ ERROR_IMAGE ({result['error']}, {result['size_kb']}KB)")
            error_count += 1
        else:
            print(f"✗ {result['status']} ({result['error']})")
            network_error_count += 1

        # Small delay
        time.sleep(0.5)

    print("-" * 80)
    print(
        f"Results: {ok_count} OK, {error_count} error images, {network_error_count} network errors"
    )

    if ok_count > 0:
        print(f"✓ {ok_count}/{len(test_cameras)} cameras working properly")
    else:
        print("⚠ No working cameras found - check config or network")


def test_single_camera(cam_id: str):
    """Test a single camera"""
    # Load config
    try:
        import yaml

        config_path = Path("config.yaml")
        if config_path.exists():
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            cfg = {
                "endpoint_template": "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={cam_id}&t={epoch_ms}",
                "na_size": [284, 177],
                "na_sha256": "1ef51d905a2e9d42678f4a37d5d54ecf9407dfd683ef39607782ddd64ab1aed5",
                "na_ahash_8x8": "ffe3cbc3c3e781ff",
            }
    except ImportError:
        cfg = {
            "endpoint_template": "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={cam_id}&t={epoch_ms}",
            "na_size": [284, 177],
            "na_sha256": "1ef51d905a2e9d42678f4a37d5d54ecf9407dfd683ef39607782ddd64ab1aed5",
            "na_ahash_8x8": "ffe3cbc3c3e781ff",
        }

    print(f"Testing camera: {cam_id}")
    print(
        f"URL: {cfg['endpoint_template'].format(cam_id=cam_id, epoch_ms=now_epoch_ms())}"
    )

    result = test_camera(cam_id, cfg["endpoint_template"], cfg)

    if result["status"] == "OK":
        print(
            f"✓ Camera working: {result['size_kb']}KB, {result['dimensions']}, {result['mode']}"
        )
    else:
        print(
            f"✗ Camera failed: {result['status']} - {result.get('error', 'Unknown error')}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            # Test single camera
            if len(sys.argv) > 2:
                test_single_camera(sys.argv[2])
            else:
                print("Usage: python test_cameras.py single <cam_id>")
        else:
            # Test with custom sample size
            try:
                sample_size = int(sys.argv[1])
                test_cameras_from_csv(sample_size=sample_size)
            except ValueError:
                print(
                    "Usage: python test_cameras.py [sample_size] or python test_cameras.py single <cam_id>"
                )
    else:
        # Default: test 10 cameras
        test_cameras_from_csv()
