import json
import re
import time
import hashlib
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# ====== ĐƯỜNG DẪN ======
INPUT_PATH = "../FolderAjax_Response.json"
OUTPUT_JSON = "../camera_catalog/camera_catalog_imagehandler.json"

# ====== Mapping ======
FRIENDLY_MAP = {
    "CamId": "cam_id",
    "Code": "code",
    "CamType": "cam_type",
    "Disctrict": "district",
    "District": "district",
    "CamStatus": "status",
    "PTZ": "ptz",
    "Angle": "angle",
    "ManagementUnit": "management_unit",
    "Title": "title",
    "Name": "name",
    "Address": "address",
    "CamLocation": "cam_location",
    "Longitude": "lon",
    "Latitude": "lat",
    "Long": "lon",
    "Lat": "lat",
    "X": "x",
    "Y": "y",
}

# ====== Mẫu nhận diện "media-like" ======
MEDIA_PATTERNS = re.compile(
    r"(snapshot|hls|m3u8|rtsp|stream|video|mp4|jpeg|jpg|gif)", re.IGNORECASE
)

# ====== Regex fallback ======
NODE_ANCHOR_RE = re.compile(r'"Layer"\s*:\s*"CAMERA"', re.IGNORECASE)
TITLE_RE = re.compile(r'"Title"\s*:\s*"([^"]*)"', re.IGNORECASE)
NAME_RE = re.compile(r'"Name"\s*:\s*"([^"]*)"', re.IGNORECASE)
PATH_RE = re.compile(r'"Path"\s*:\s*"([^"]*)"', re.IGNORECASE)
TYPE_RE = re.compile(r'"Type"\s*:\s*"([^"]*)"', re.IGNORECASE)
CREATED_RE = re.compile(r'"Created"\s*:\s*"([^"]*)"', re.IGNORECASE)
MODIFIED_RE = re.compile(r'"Modified"\s*:\s*"([^"]*)"', re.IGNORECASE)
PROPS_BLOCK_RE = re.compile(r'"Properties"\s*:\s*\[(.*?)\]', re.DOTALL | re.IGNORECASE)
PROP_OBJ_RE = re.compile(r"\{(.*?)\}", re.DOTALL)
KVP_NAME_RE = re.compile(r'"Name"\s*:\s*"([^"]*)"', re.IGNORECASE)
KVP_DISPLAYNAME_RE = re.compile(r'"DisplayName"\s*:\s*"([^"]*)"', re.IGNORECASE)
KVP_VALUE_RE = re.compile(
    r'"Value"\s*:\s*(?P<val>"[^"]*"|true|false|null|-?\d+(?:\.\d+)?)', re.IGNORECASE
)
KVP_DEFAULT_RE = re.compile(
    r'"DefaultValue"\s*:\s*(?P<val>"[^"]*"|true|false|null|-?\d+(?:\.\d+)?)',
    re.IGNORECASE,
)


# ====== Tiện ích ======
def try_load_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _json_dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_of_obj(obj: Any) -> str:
    h = hashlib.sha256(_json_dumps_canonical(obj).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def parse_dotnet_date(s: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Chuyển '/Date(1693200000000)/' -> (iso8601_utc, epoch_ms).
    Trả về (iso, ms). Nếu không hợp lệ, trả (None, None).
    """
    if not s:
        return None, None
    m = re.search(r"/Date\((\-?\d+)\)/", s)
    if not m:
        return None, None
    try:
        ms = int(m.group(1))
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.isoformat(), ms
    except Exception:
        return None, None


def to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vs = v.strip().lower()
        if vs in ("true", "1", "yes"):
            return True
        if vs in ("false", "0", "no"):
            return False
    return None


def to_int(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        vs = v.strip()
        try:
            return int(float(vs))
        except Exception:
            return None
    return None


def to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        vs = v.strip()
        try:
            return float(vs)
        except Exception:
            return None
    return None


def choose_loc_raw(
    props_norm: Dict[str, Any], node_title: Optional[str], code: Optional[str]
) -> Optional[str]:
    return props_norm.get("cam_location") or node_title or code


# ==== CHUẨN HOÁ LINK ẢNH (ImageHandler) ====
def build_image_link(
    cam_id: Optional[str], loc_raw: Optional[str]
) -> Dict[str, Optional[str]]:
    """
    Trả về link ảnh dạng ImageHandler.ashx dựa trên cam_id.
    - loc_raw/loc_encoded: giữ thêm nhãn thuận tiện (không bắt buộc khi dùng ImageHandler)
    - image_url: https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=<cam_id>&t=<epoch_ms>
    """
    loc_encoded = urllib.parse.quote(str(loc_raw), safe="") if loc_raw else None
    if not cam_id:
        return {"loc_raw": loc_raw, "loc_encoded": loc_encoded, "image_url": None}

    t_ms = int(time.time() * 1000)
    image_url = (
        "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx"
        f"?id={cam_id}&t={t_ms}"
    )
    return {"loc_raw": loc_raw, "loc_encoded": loc_encoded, "image_url": image_url}


def walk(obj: Any):
    """Duyệt toàn cây JSON."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


# ====== Build properties_raw từ list Properties (GIỮ NGUYÊN) ======
def build_properties_raw_from_list(prop_list: Any) -> Dict[str, Dict[str, Any]]:
    """
    Trả về dict: Name -> {Value, DefaultValue, DisplayName, ... any other fields}
    Không xoá / không lọc gì cả.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(prop_list, list):
        return out
    for p in prop_list:
        if not isinstance(p, dict):
            continue
        name = (p.get("Name") or "").strip()
        if not name:
            name = f"__NO_NAME__:{hashlib.sha1(_json_dumps_canonical(p).encode('utf-8')).hexdigest()[:10]}"
        entry = {k: v for k, v in p.items() if k != "Name"}
        out[name] = entry
    return out


def normalize_properties(properties_raw: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sinh properties_norm: map tên thân thiện, ép kiểu nhẹ (ptz->bool, angle->int, lat/lon->float),
    KHÔNG xoá bất kỳ key nào trong properties_raw. Chỉ 'đọc' để map.
    """
    norm: Dict[str, Any] = {}

    for raw_key, entry in (properties_raw or {}).items():
        val = None
        if isinstance(entry, dict):
            if "Value" in entry:
                val = entry.get("Value")
            elif "DefaultValue" in entry:
                val = entry.get("DefaultValue")

        friendly = FRIENDLY_MAP.get(raw_key)
        if friendly:
            if friendly == "ptz":
                b = to_bool(val)
                norm[friendly] = b if b is not None else val
            elif friendly == "angle":
                i = to_int(val)
                norm[friendly] = i if i is not None else val
            elif friendly in ("lat", "lon", "x", "y"):
                f = to_float(val)
                norm[friendly] = f if f is not None else val
            else:
                norm[friendly] = val

    # Hợp nhất district nếu ghi sai chính tả Disctrict
    if "district" not in norm:
        for raw_key, entry in (properties_raw or {}).items():
            if raw_key.lower() in ("district", "disctrict"):
                val = entry.get("Value") if isinstance(entry, dict) else None
                if val:
                    norm["district"] = val
                    break

    return norm


def detect_media_flags(properties_raw: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    flags = {
        "has_snapshot_url": False,
        "has_hls": False,
        "has_rtsp": False,
        "has_image_like": False,
    }
    for k, entry in (properties_raw or {}).items():
        key_hit = bool(MEDIA_PATTERNS.search(k or ""))
        val = entry.get("Value") if isinstance(entry, dict) else None

        val_hit = False
        if isinstance(val, str) and MEDIA_PATTERNS.search(val):
            val_hit = True

        if key_hit or val_hit:
            flags["has_image_like"] = True
            if re.search(r"snapshot", k, re.IGNORECASE) or (
                isinstance(val, str) and "snapshot" in val.lower()
            ):
                flags["has_snapshot_url"] = True
            if re.search(r"hls|m3u8", k, re.IGNORECASE) or (
                isinstance(val, str) and re.search(r"hls|m3u8", val, re.IGNORECASE)
            ):
                flags["has_hls"] = True
            if re.search(r"rtsp", k, re.IGNORECASE) or (
                isinstance(val, str) and "rtsp" in val.lower()
            ):
                flags["has_rtsp"] = True

    return flags


# ====== Trích xuất theo "đường JSON" (khi parse được JSON) ======
def parse_json_path(data: Any) -> List[Dict[str, Any]]:
    # AjaxPro có thể bọc {"value": "...json string..."} hoặc {"value": {...}}
    if isinstance(data, dict) and "value" in data:
        inner = data["value"]
        if isinstance(inner, str):
            maybe = try_load_json(inner)
            data = maybe if maybe is not None else data
        else:
            data = inner

    rows: List[Dict[str, Any]] = []

    for node in walk(data):
        if not isinstance(node, dict):
            continue
        if node.get("Layer") != "CAMERA":
            continue

        node_meta: Dict[str, Any] = {}
        for k in ("Layer", "Type", "Title", "Name", "Path", "Created", "Modified"):
            if k in node:
                node_meta[k] = node[k]

        created_iso, created_ms = parse_dotnet_date(node_meta.get("Created"))
        modified_iso, modified_ms = parse_dotnet_date(node_meta.get("Modified"))
        if created_iso:
            node_meta["created_iso"] = created_iso
            node_meta["created_ms"] = created_ms
        if modified_iso:
            node_meta["modified_iso"] = modified_iso
            node_meta["modified_ms"] = modified_ms

        node_meta["raw_node_hash"] = sha256_of_obj(node)

        properties_raw = build_properties_raw_from_list(node.get("Properties"))
        properties_norm = normalize_properties(properties_raw)

        node_title = node.get("Title") or None
        code_for_label = properties_norm.get("code")
        cam_id = properties_norm.get("cam_id")

        loc_raw = choose_loc_raw(properties_norm, node_title, code_for_label)
        link_info = build_image_link(cam_id, loc_raw)
        media_flags = detect_media_flags(properties_raw)

        group_key_src = {}
        if cam_id:
            group_key_src["cam_id"] = cam_id
        if node_meta.get("Path"):
            group_key_src["Path"] = node_meta["Path"]
        group_key_src["properties_raw"] = properties_raw
        possible_duplicate_group = sha256_of_obj(group_key_src)

        rows.append(
            {
                "node_meta": node_meta,
                "properties_raw": properties_raw,
                "properties_norm": properties_norm,
                "media_flags": media_flags,
                "link_info": link_info,
                "possible_duplicate_group": possible_duplicate_group,
            }
        )

    return rows


# ====== Fallback Regex khi không parse được JSON ======
def parse_regex_fallback(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for m in NODE_ANCHOR_RE.finditer(text):
        start = max(0, m.start() - 5000)
        end = min(len(text), m.end() + 8000)
        blob = text[start:end]

        node_meta: Dict[str, Any] = {}
        type_m = TYPE_RE.search(blob)
        title_m = TITLE_RE.search(blob)
        name_m = NAME_RE.search(blob)
        path_m = PATH_RE.search(blob)
        created_m = CREATED_RE.search(blob)
        modified_m = MODIFIED_RE.search(blob)

        node_meta["Layer"] = "CAMERA"
        if type_m:
            node_meta["Type"] = type_m.group(1)
        if title_m:
            node_meta["Title"] = title_m.group(1)
        if name_m:
            node_meta["Name"] = name_m.group(1)
        if path_m:
            node_meta["Path"] = path_m.group(1)
        if created_m:
            node_meta["Created"] = created_m.group(1)
        if modified_m:
            node_meta["Modified"] = modified_m.group(1)

        created_iso, created_ms = parse_dotnet_date(node_meta.get("Created"))
        modified_iso, modified_ms = parse_dotnet_date(node_meta.get("Modified"))
        if created_iso:
            node_meta["created_iso"] = created_iso
            node_meta["created_ms"] = created_ms
        if modified_iso:
            node_meta["modified_iso"] = modified_iso
            node_meta["modified_ms"] = modified_ms

        node_meta["raw_node_hash"] = sha256_of_obj(node_meta)

        props_block_m = PROPS_BLOCK_RE.search(blob)
        properties_raw: Dict[str, Dict[str, Any]] = {}
        if props_block_m:
            props_block = props_block_m.group(1)
            for obj_m in PROP_OBJ_RE.finditer(props_block):
                obj_txt = obj_m.group(1)
                name = None
                display_name = None
                val = None
                dval = None

                nm = KVP_NAME_RE.search(obj_txt)
                if nm:
                    name = nm.group(1)

                dm = KVP_DISPLAYNAME_RE.search(obj_txt)
                if dm:
                    display_name = dm.group(1)

                vm = KVP_VALUE_RE.search(obj_txt)
                if vm:
                    val_raw = vm.group("val")
                    val = parse_json_like_scalar(val_raw)

                dvm = KVP_DEFAULT_RE.search(obj_txt)
                if dvm:
                    dval_raw = dvm.group("val")
                    dval = parse_json_like_scalar(dval_raw)

                if not name:
                    name = f"__NO_NAME__:{hashlib.sha1(obj_txt.encode('utf-8')).hexdigest()[:10]}"
                entry: Dict[str, Any] = {}
                if val is not None or "Value" in obj_txt:
                    entry["Value"] = val
                if dval is not None or "DefaultValue" in obj_txt:
                    entry["DefaultValue"] = dval
                if display_name is not None:
                    entry["DisplayName"] = display_name
                entry["_raw_fragment"] = obj_txt.strip()
                properties_raw[name] = entry

        properties_norm = normalize_properties(properties_raw)

        node_title = node_meta.get("Title")
        code_for_label = properties_norm.get("code")
        cam_id = properties_norm.get("cam_id")

        loc_raw = choose_loc_raw(properties_norm, node_title, code_for_label)
        link_info = build_image_link(cam_id, loc_raw)
        media_flags = detect_media_flags(properties_raw)

        group_key_src = {}
        if cam_id:
            group_key_src["cam_id"] = cam_id
        if node_meta.get("Path"):
            group_key_src["Path"] = node_meta["Path"]
        group_key_src["properties_raw"] = properties_raw
        possible_duplicate_group = sha256_of_obj(group_key_src)

        rows.append(
            {
                "node_meta": node_meta,
                "properties_raw": properties_raw,
                "properties_norm": properties_norm,
                "media_flags": media_flags,
                "link_info": link_info,
                "possible_duplicate_group": possible_duplicate_group,
            }
        )

    return rows


def parse_json_like_scalar(s: str) -> Any:
    """
    Chuyển chuỗi token JSON-like thành Python:
      - "text" -> "text"
      - true/false -> bool
      - null -> None
      - number -> int/float
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


# ====== MAIN ======
def main():
    with open(INPUT_PATH, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    items: List[Dict[str, Any]] = []
    data = try_load_json(text)
    if data is not None:
        items = parse_json_path(data)
    if not items:
        items = parse_regex_fallback(text)

    # Sắp xếp ổn định: Title rồi cam_id
    def sort_key(rec: Dict[str, Any]):
        node_meta = rec.get("node_meta") or {}
        title = (
            node_meta.get("Title") or rec.get("properties_norm", {}).get("code") or ""
        )
        cam_id = rec.get("properties_norm", {}).get("cam_id") or ""
        return (str(title), str(cam_id))

    items.sort(key=sort_key)

    # Xây JSON nhẹ (light) với ImageHandler URL
    light_rows = []
    for rec in items:
        node_meta = rec.get("node_meta", {})
        props_norm = rec.get("properties_norm", {})
        link_info = rec.get("link_info") or {}
        media_flags = rec.get("media_flags") or {}
        light_rows.append(
            {
                "cam_id": props_norm.get("cam_id"),
                "code": props_norm.get("code"),
                "title": node_meta.get("Title") or props_norm.get("title"),
                "district": props_norm.get("district"),
                "cam_type": props_norm.get("cam_type"),
                "ptz": props_norm.get("ptz"),
                "angle": props_norm.get("angle"),
                "image_url": link_info.get("image_url"),
                "loc_raw": link_info.get("loc_raw"),
                "has_snapshot_url": media_flags.get("has_snapshot_url"),
                "has_hls": media_flags.get("has_hls"),
                "has_rtsp": media_flags.get("has_rtsp"),
                "possible_duplicate_group": rec.get("possible_duplicate_group"),
                "_full_ref_hash": sha256_of_obj(
                    rec
                ),  # tham chiếu nhanh đến bản ghi đầy đủ (không lưu full để nhẹ)
            }
        )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(light_rows, f, ensure_ascii=False, indent=2)

    # Thống kê nhanh
    total = len(items)
    with_snapshot = sum(
        1 for r in items if (r.get("media_flags") or {}).get("has_snapshot_url")
    )
    with_hls = sum(1 for r in items if (r.get("media_flags") or {}).get("has_hls"))
    with_rtsp = sum(1 for r in items if (r.get("media_flags") or {}).get("has_rtsp"))

    print(f"[OK] Records (no-dedupe): {total}")
    print(f"[OK] Light JSON (ImageHandler): {OUTPUT_JSON}")
    print(
        f"[Info] Media flags            : snapshot={with_snapshot}, hls={with_hls}, rtsp={with_rtsp}"
    )


if __name__ == "__main__":
    main()
