import os
import sys
import csv
import json
import glob
import re
import base64
import math
import cv2
import threading
import queue
import subprocess
import zipfile
import multiprocessing
import sqlite3
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from flask import Flask, render_template_string, request, jsonify, Response
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
import ctypes

progress_queue = queue.Queue()
conversion_state = {"status": "idle", "percent": 0, "message": "Ready"}
io_choke_flag = multiprocessing.Value(ctypes.c_double, 0.0)

app = Flask(__name__)

# ==========================================
# CONFIGURATION PERSISTENCE
# ==========================================
MAPPING_FILE = "mcap_mapping_config.json"

DEFAULT_CONFIG = {
    "coord_mode": "auto",
    "roles": {
        "rov_lat": "", "rov_lat_dir": "", "rov_lon": "", "rov_lon_dir": "", "rov_depth": "",
        "rov_heading": "", "rov_pitch": "", "rov_roll": "",
        "ship_lat": "", "ship_lat_dir": "", "ship_lon": "", "ship_lon_dir": "", "ship_heading": ""
    },
    "telemetry": {},
    "ofop": []
}


def load_mapping_config():
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "r") as f:
                data = json.load(f)
                return {**DEFAULT_CONFIG, **data}
        except:
            pass
    return DEFAULT_CONFIG


def save_mapping_config(config):
    try:
        with open(MAPPING_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except:
        pass


# ==========================================
# NATIVE MACOS BROWSER
# ==========================================
def macos_browse(is_save=False):
    try:
        script = '''
        set frontApp to (path to frontmost application as text)
        tell application frontApp
            activate
            set outPath to POSIX path of (choose file name with prompt "Save MCAP As..." default name "Dive.mcap")
        end tell
        return outPath
        ''' if is_save else '''
        set frontApp to (path to frontmost application as text)
        tell application frontApp
            activate
            set outPath to POSIX path of (choose folder with prompt "Select Directory")
        end tell
        return outPath
        '''
        return subprocess.check_output(['osascript', '-e', script]).decode('utf-8').strip()
    except:
        return ""


# ==========================================
# PARSERS & ADAPTIVE DATA HELPERS
# ==========================================
def parse_metadata_xml(xml_path):
    meta = {}
    if not xml_path or not os.path.exists(xml_path): return meta
    try:
        tree = ET.parse(xml_path)
        for void in tree.getroot().findall('.//void[@method="put"]'):
            strings = void.findall('string')
            if len(strings) == 2 and strings[0].text:
                meta[strings[0].text] = strings[1].text or ""
    except:
        pass
    return meta


def parse_coordinate(val_str, direction="", mode="auto"):
    if not val_str or str(val_str).strip() in ("", "NA", "NaN", "None"):
        return None
    try:
        val = float(val_str)
        # If the number is already negative, or the direction field says S or W, it becomes negative
        is_negative = val < 0 or (direction and str(direction).strip().upper() in ("S", "W"))
        val = abs(val)

        if mode == "decimal":
            dec = val
        elif mode == "ddmm":
            deg = int(val / 100)
            dec = deg + ((val - (deg * 100)) / 60.0)
        else:  # auto mode
            if val > 180.0 or (val > 90.0 and ("N" in str(direction).upper() or "S" in str(direction).upper())):
                deg = int(val / 100)
                dec = deg + ((val - (deg * 100)) / 60.0)
            else:
                dec = val

        return -dec if is_negative else dec
    except:
        return None


def safe_float(val_str):
    if not val_str or str(val_str).strip() in ("", "NA", "NaN", "None"): return None
    try:
        return float(val_str)
    except:
        return None


def parse_robust_timestamp(ts_str):
    if not ts_str: return None
    ts_str = str(ts_str).strip()
    formats = [
        "%d.%m.%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%d/%m/%Y %H:%M:%S"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ==========================================
# PARALLEL VIDEO WORKER (AUTO-TUNE I/O)
# ==========================================
def extract_video_frames_worker(args):
    vid_path, cam_name, width, height, quality, sample_sec, worker_q = args
    fname = os.path.basename(vid_path)
    match = re.search(r"(\d{8})_(\d{2}-\d{2}-\d{2})", fname)
    if not match: return []

    dt_start = parse_robust_timestamp(
        f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]} {match.group(2).replace('-', ':')}")
    if not dt_start: return []
    start_ns = int(dt_start.timestamp() * 1e9)

    match_worker = re.search(r'(\d+)$', multiprocessing.current_process().name)
    core_num = match_worker.group(1) if match_worker else "1"
    core_id = f"core_{core_num}"

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened(): return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    estimated_out_frames = int(total_frames / (fps * sample_sec)) if fps > 0 else 100
    results = []

    ffmpeg_path = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not os.path.exists(ffmpeg_path): ffmpeg_path = "/usr/local/bin/ffmpeg"

    if os.path.exists(ffmpeg_path):
        ffmpeg_q = "5" if quality > 70 else "7"
        fps_str = str(1.0 / sample_sec)

        cmd = [
            ffmpeg_path, "-y", "-v", "error",
            "-hwaccel", "videotoolbox",
            "-i", vid_path,
            "-vf", f"fps={fps_str},scale={width}:{height}",
            "-q:v", ffmpeg_q,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1"
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buffer = b""
        frame_idx = 0
        ui_update_interval = max(1, int(10 / sample_sec))
        import time

        while True:
            if time.time() < io_choke_flag.value:
                time.sleep(0.05)

            t_start = time.time()
            chunk = proc.stdout.read(8192)
            read_latency = time.time() - t_start

            if read_latency > 0.1:
                io_choke_flag.value = time.time() + 0.25

            if not chunk: break
            buffer += chunk

            while True:
                start = buffer.find(b'\xff\xd8')
                end = buffer.find(b'\xff\xd9')
                if start != -1 and end != -1 and end > start:
                    jpg_data = buffer[start:end + 2]
                    buffer = buffer[end + 2:]

                    ts_ns = start_ns + int(frame_idx * sample_sec * 1e9)
                    json_timestamp = {"sec": int(ts_ns // 1e9), "nsec": int(ts_ns % 1e9)}
                    b64_str = base64.b64encode(jpg_data).decode('utf-8')

                    payload = {"timestamp": json_timestamp, "frame_id": cam_name, "format": "jpeg", "data": b64_str}
                    results.append((ts_ns, json.dumps(payload).encode('utf-8')))

                    if frame_idx % ui_update_interval == 0 and worker_q is not None:
                        pct = min(99, int((frame_idx / estimated_out_frames) * 100)) if estimated_out_frames > 0 else 0
                        throttle_warning = " ⚠️ I/O Choke" if time.time() < io_choke_flag.value else ""
                        try:
                            worker_q.put_nowait({
                                "type": "video_progress", "core_id": core_id, "core_num": core_num,
                                "label": f"(Auto-Tune) {fname}{throttle_warning}", "cam": cam_name,
                                "progress": pct, "b64": b64_str
                            })
                        except queue.Full:
                            pass
                    frame_idx += 1
                else:
                    break

        proc.stdout.close()
        proc.wait()

    if worker_q is not None:
        try:
            worker_q.put_nowait(
                {"type": "video_progress", "core_id": core_id, "core_num": core_num, "label": f"✅ {fname}",
                 "cam": cam_name, "progress": 100})
        except:
            pass

    return results


# ==========================================
# SPLIT FILE DISCOVERY & HEADER EXTRACTION
# ==========================================
def scan_dive_directory(data_base_dir, video_base_dir="", folder_id=None):
    csv_files, image_files, ofop_files, video_files = [], [], [], []
    xml_file = None
    detected_dives = set()

    if not os.path.exists(data_base_dir): return None

    dive_pattern = re.compile(r"(\d{2,4}[-_]?[A-Za-z0-9]+ROV[-_]?[A-Za-z0-9]+|ROV[-_]?\d+|Dive[-_]?\d+)", re.IGNORECASE)

    base_name = os.path.basename(os.path.normpath(data_base_dir))
    match_base = dive_pattern.search(base_name)
    if match_base: detected_dives.add(match_base.group(1))

    for root, dirs, _ in os.walk(data_base_dir):
        for d in dirs:
            match_dir = dive_pattern.search(d)
            if match_dir: detected_dives.add(match_dir.group(1))

    if not folder_id and detected_dives:
        folder_id = sorted(list(detected_dives))[0]

    extract_dir = os.path.join(os.getcwd(), "extracted_data")
    os.makedirs(extract_dir, exist_ok=True)

    for root, _, files in os.walk(data_base_dir):
        for f in files:
            if f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                target = os.path.join(extract_dir, f.replace('.zip', ''))
                if not os.path.exists(target):
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(target)
                    except:
                        pass

    for base in [data_base_dir, extract_dir]:
        for root, _, files in os.walk(base):
            for f in files:
                # Check XML
                if f.endswith(".xml") and "metadata" in f.lower():
                    if not folder_id or folder_id in f or folder_id in root:
                        xml_file = os.path.join(root, f)

                # Check Telemetry CSVs
                if (not folder_id or folder_id in root or folder_id in f) and f.endswith(".csv"):
                    if "telemetry" in root.lower() or "tls" in f.lower() or "track" in f.lower() or "dship" in f.lower():
                        csv_files.append(os.path.join(root, f))

                # Check Stills
                if (not folder_id or folder_id in root or folder_id in f) and (
                        f.endswith(".jpg") or f.endswith(".jpeg")):
                    if "still" in root.lower() or "photo" in root.lower():
                        image_files.append(os.path.join(root, f))

                # Check OFOP Logs (Forgiving Path Match)
                if not folder_id or folder_id in root or folder_id in f:
                    if ("protocol" in f.lower() or "ofop" in f.lower() or "obs" in f.lower()) and (
                            f.endswith(".csv") or f.endswith(".txt") or f.endswith(".tsv")):
                        ofop_files.append(os.path.join(root, f))

    # Video discovery
    target_vid = video_base_dir if video_base_dir and os.path.exists(video_base_dir) else data_base_dir
    for root, _, files in os.walk(target_vid):
        # Allow video if folder ID is in the path OR if we just want all of them
        if not folder_id or folder_id in root or folder_id in target_vid:
            cam_name = "unknown_camera"
            for part in os.path.normpath(root).split(os.sep):
                if part.startswith("cam_"):
                    cam_name = part
                    break
            for f in files:
                if f.endswith(".mov") or f.endswith(".mp4"):
                    if cam_name == "unknown_camera":
                        m = re.search(r"(cam_[a-zA-Z0-9_]+)", f)
                        if m: cam_name = m.group(1)
                    video_files.append((os.path.join(root, f), cam_name))

    tel_headers = []
    if csv_files:
        try:
            with open(sorted(csv_files)[0], "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                delim = '\t' if '\t' in first_line else (';' if ';' in first_line else ',')
                f.seek(0)
                tel_headers = next(csv.reader(f, delimiter=delim), [])
        except:
            pass

    ofop_headers = []
    if ofop_files:
        try:
            with open(sorted(ofop_files)[0], "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                delim = '\t' if '\t' in first_line else (';' if ';' in first_line else ',')
                f.seek(0)
                ofop_headers = next(csv.reader(f, delimiter=delim), [])
        except:
            pass

    return {
        "detected_dives": sorted(list(detected_dives)),
        "selected_dive": folder_id,
        "xml_file": xml_file,
        "csv_count": len(csv_files),
        "csv_files": sorted(csv_files),
        "tel_headers": tel_headers,
        "image_count": len(image_files),
        "image_files": sorted(image_files),
        "ofop_count": len(ofop_files),
        "ofop_files": sorted(ofop_files),
        "ofop_headers": ofop_headers,
        "video_count": len(video_files),
        "video_files": video_files,
        "cameras": sorted(list(set(c for _, c in video_files))),
        "available_cores": os.cpu_count() or 4
    }


# ==========================================
# MCAP BUILDER (UNIVERSAL & SHIP-AGNOSTIC)
# ==========================================
# ==========================================
# MCAP BUILDER (UNIVERSAL & SHIP-AGNOSTIC)
# ==========================================
def run_conversion_task(scan_results, selected_cams, output_mcap, width, height, quality, sample_sec, target_cores,
                        config):
    global conversion_state

    manager = multiprocessing.Manager()
    worker_q = manager.Queue(maxsize=100)

    def relay_messages():
        while True:
            msg = worker_q.get()
            if msg == "STOP": break
            progress_queue.put(msg)

    threading.Thread(target=relay_messages, daemon=True).start()

    def log(msg, pct=None):
        if pct is not None: conversion_state["percent"] = pct
        conversion_state["message"] = msg
        progress_queue.put({"type": "log", "message": msg, "percent": conversion_state["percent"]})

    db_path = os.path.join(os.getcwd(), "extracted_data", "mcap_spool.db")
    roles = config.get("roles", {})
    tel_mapping = config.get("telemetry", {})
    ofop_mapping = config.get("ofop", [])
    coord_mode = config.get("coord_mode", "auto")

    try:
        conversion_state["status"] = "running"
        log("🚀 Starting Universal MCAP build...", 2)

        os.makedirs(os.path.dirname(os.path.abspath(output_mcap)), exist_ok=True)
        if os.path.exists(db_path): os.remove(db_path)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE messages (ts INTEGER, ch_id INTEGER, data BLOB)")

        def flush_to_db(batch):
            if batch:
                c.executemany("INSERT INTO messages (ts, ch_id, data) VALUES (?, ?, ?)", batch)
                conn.commit()

        # --- ROGUE TIMESTAMP FILTER (The Anchor) ---
        min_valid_ns, max_valid_ns = None, None
        video_timestamps = []
        for v_path, _ in scan_results.get("video_files", []):
            match = re.search(r"(\d{8})_(\d{2}-\d{2}-\d{2})", os.path.basename(v_path))
            if match:
                dt_str = f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]} {match.group(2).replace('-', ':')}"
                dt = parse_robust_timestamp(dt_str)
                if dt: video_timestamps.append(dt.timestamp())

        if video_timestamps:
            # 24 hour buffer before first video, 48 hours after last video
            min_valid_ns = int((min(video_timestamps) - (24 * 3600)) * 1e9)
            max_valid_ns = int((max(video_timestamps) + (48 * 3600)) * 1e9)
            log("Temporal Filter Active: Ignoring rogue data >24h outside video timeframe...", 3)

        with open(output_mcap, "wb") as f_out:
            writer = Writer(f_out)
            writer.start()

            if scan_results.get("xml_file"):
                writer.add_metadata(name="Dive_Metadata", data=parse_metadata_xml(scan_results["xml_file"]))

            ts_prop = {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}}
            loc_schema_dict = {"type": "object", "properties": {"timestamp": ts_prop, "latitude": {"type": "number"},
                                                                "longitude": {"type": "number"},
                                                                "altitude": {"type": "number"}}}
            tel_schema_dict = {"type": "object"}
            img_schema_dict = {"type": "object", "properties": {"timestamp": ts_prop, "frame_id": {"type": "string"},
                                                                "format": {"type": "string"}, "data": {"type": "string",
                                                                                                       "contentEncoding": "base64"}}}
            log_schema_dict = {"type": "object", "properties": {"timestamp": ts_prop, "level": {"type": "integer"},
                                                                "message": {"type": "string"},
                                                                "name": {"type": "string"}}}
            tf_schema_dict = {"type": "object",
                              "properties": {"timestamp": ts_prop, "parent_frame_id": {"type": "string"},
                                             "child_frame_id": {"type": "string"}, "translation": {"type": "object",
                                                                                                   "properties": {"x": {
                                                                                                       "type": "number"},
                                                                                                                  "y": {
                                                                                                                      "type": "number"},
                                                                                                                  "z": {
                                                                                                                      "type": "number"}}},
                                             "rotation": {"type": "object", "properties": {"x": {"type": "number"},
                                                                                           "y": {"type": "number"},
                                                                                           "z": {"type": "number"},
                                                                                           "w": {"type": "number"}}}}}
            cube_properties = {"pose": {"type": "object", "properties": {"position": {"type": "object", "properties": {
                "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
                                                                         "orientation": {"type": "object",
                                                                                         "properties": {
                                                                                             "x": {"type": "number"},
                                                                                             "y": {"type": "number"},
                                                                                             "z": {"type": "number"},
                                                                                             "w": {
                                                                                                 "type": "number"}}}}},
                               "size": {"type": "object",
                                        "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                                                       "z": {"type": "number"}}}, "color": {"type": "object",
                                                                                            "properties": {
                                                                                                "r": {"type": "number"},
                                                                                                "g": {"type": "number"},
                                                                                                "b": {"type": "number"},
                                                                                                "a": {
                                                                                                    "type": "number"}}}}
            entity_properties = {"id": {"type": "string"}, "timestamp": ts_prop, "frame_id": {"type": "string"},
                                 "cubes": {"type": "array", "items": {"type": "object", "properties": cube_properties}}}
            scene_schema_dict = {"type": "object", "properties": {
                "entities": {"type": "array", "items": {"type": "object", "properties": entity_properties}}}}

            loc_schema = writer.register_schema(name="foxglove.LocationFix", encoding=SchemaEncoding.JSONSchema,
                                                data=json.dumps(loc_schema_dict).encode("utf-8"))
            rov_gps_ch = writer.register_channel(schema_id=loc_schema, topic="/rov/gps",
                                                 message_encoding=MessageEncoding.JSON)
            ship_gps_ch = writer.register_channel(schema_id=loc_schema, topic="/ship/gps",
                                                  message_encoding=MessageEncoding.JSON)
            tel_schema = writer.register_schema(name="Telemetries", encoding=SchemaEncoding.JSONSchema,
                                                data=json.dumps(tel_schema_dict).encode("utf-8"))
            rov_tel_ch = writer.register_channel(schema_id=tel_schema, topic="/rov/telemetry",
                                                 message_encoding=MessageEncoding.JSON)
            ship_tel_ch = writer.register_channel(schema_id=tel_schema, topic="/ship/telemetry",
                                                  message_encoding=MessageEncoding.JSON)
            img_schema = writer.register_schema(name="foxglove.CompressedImage", encoding=SchemaEncoding.JSONSchema,
                                                data=json.dumps(img_schema_dict).encode("utf-8"))
            img_ch = writer.register_channel(schema_id=img_schema, topic="/rov/camera_digistills",
                                             message_encoding=MessageEncoding.JSON)

            vid_channels = {}
            for cam in selected_cams:
                vid_channels[cam] = writer.register_channel(schema_id=img_schema, topic=f"/rov/video_preview/{cam}",
                                                            message_encoding=MessageEncoding.JSON)

            log_schema = writer.register_schema(name="foxglove.Log", encoding=SchemaEncoding.JSONSchema,
                                                data=json.dumps(log_schema_dict).encode("utf-8"))
            img_log_ch = writer.register_channel(schema_id=log_schema, topic="/rov/image_timeline",
                                                 message_encoding=MessageEncoding.JSON)
            ofop_log_ch = writer.register_channel(schema_id=log_schema, topic="/rov/ofop_timeline",
                                                  message_encoding=MessageEncoding.JSON)
            tf_schema = writer.register_schema(name="foxglove.FrameTransform", encoding=SchemaEncoding.JSONSchema,
                                               data=json.dumps(tf_schema_dict).encode("utf-8"))
            tf_ch = writer.register_channel(schema_id=tf_schema, topic="/tf", message_encoding=MessageEncoding.JSON)
            scene_schema = writer.register_schema(name="foxglove.SceneUpdate", encoding=SchemaEncoding.JSONSchema,
                                                  data=json.dumps(scene_schema_dict).encode("utf-8"))
            scene_ch = writer.register_channel(schema_id=scene_schema, topic="/rov/3d_model",
                                               message_encoding=MessageEncoding.JSON)

            origin_lat, origin_lon = None, None

            # 1. Telemetry Processing
            csv_files = scan_results.get("csv_files", [])
            if csv_files:
                log(f"Processing {len(csv_files)} Telemetry CSV segments...", 10)
                batch = []
                for csv_path in csv_files:
                    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f_in:
                        first_line = f_in.readline()
                        delim = '\t' if '\t' in first_line else (';' if ';' in first_line else ',')
                        f_in.seek(0)
                        reader = csv.DictReader(f_in, delimiter=delim)

                        for row in reader:
                            try:
                                dt = parse_robust_timestamp(
                                    row.get('Timestamp', '') or row.get('Date_Time', '') or row.get('time', ''))
                                if not dt: continue
                                ts_ns = int(dt.timestamp() * 1e9)

                                # INSTANT SKIP FOR ROGUE TIMESTAMPS
                                if min_valid_ns and (ts_ns < min_valid_ns or ts_ns > max_valid_ns):
                                    continue

                                json_ts = {"sec": int(ts_ns // 1e9), "nsec": int(ts_ns % 1e9)}

                                # --- Core ROV Navigation ---
                                r_lat = row.get(roles.get("rov_lat", ""))
                                r_lat_dir = row.get(roles.get("rov_lat_dir", ""))
                                r_lon = row.get(roles.get("rov_lon", ""))
                                r_lon_dir = row.get(roles.get("rov_lon_dir", ""))

                                rov_lat = parse_coordinate(r_lat, direction=r_lat_dir, mode=coord_mode)
                                rov_lon = parse_coordinate(r_lon, direction=r_lon_dir, mode=coord_mode)
                                rov_depth = safe_float(row.get(roles.get("rov_depth", "")))

                                if rov_lat is not None and rov_lon is not None:
                                    gps_msg = {"timestamp": json_ts, "latitude": rov_lat, "longitude": rov_lon}
                                    if rov_depth is not None: gps_msg["altitude"] = -abs(rov_depth)
                                    batch.append((ts_ns, rov_gps_ch, json.dumps(gps_msg).encode('utf-8')))

                                    if origin_lat is None: origin_lat, origin_lon = rov_lat, rov_lon
                                    x_m = (rov_lon - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))
                                    y_m = (rov_lat - origin_lat) * 110574.0
                                    z_m = -abs(rov_depth) if rov_depth else 0.0

                                    heading = safe_float(row.get(roles.get("rov_heading", ""))) or 0.0
                                    pitch = safe_float(row.get(roles.get("rov_pitch", ""))) or 0.0
                                    roll = safe_float(row.get(roles.get("rov_roll", ""))) or 0.0

                                    yaw_rad, pitch_rad, roll_rad = math.radians(90.0 - heading), math.radians(
                                        pitch), math.radians(roll)
                                    cy, sy, cp, sp, cr, sr = math.cos(yaw_rad * 0.5), math.sin(yaw_rad * 0.5), math.cos(
                                        pitch_rad * 0.5), math.sin(pitch_rad * 0.5), math.cos(roll_rad * 0.5), math.sin(
                                        roll_rad * 0.5)
                                    qw, qx, qy, qz = cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy

                                    batch.append((ts_ns, tf_ch, json.dumps(
                                        {"timestamp": json_ts, "parent_frame_id": "world", "child_frame_id": "rov",
                                         "translation": {"x": x_m, "y": y_m, "z": z_m},
                                         "rotation": {"x": qx, "y": qy, "z": qz, "w": qw}}).encode('utf-8')))
                                    batch.append((ts_ns, scene_ch, json.dumps({"entities": [
                                        {"id": "rov_box", "timestamp": json_ts, "frame_id": "rov", "cubes": [{"pose": {
                                            "position": {"x": 0, "y": 0, "z": 0},
                                            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}, "size": {"x": 2.1,
                                                                                                       "y": 1.3,
                                                                                                       "z": 1.85},
                                            "color": {
                                                "r": 1.0,
                                                "g": 0.8,
                                                "b": 0.0,
                                                "a": 1.0}}]}]}).encode(
                                        'utf-8')))

                                # --- Core Ship Navigation ---
                                s_lat = row.get(roles.get("ship_lat", ""))
                                s_lat_dir = row.get(roles.get("ship_lat_dir", ""))
                                s_lon = row.get(roles.get("ship_lon", ""))
                                s_lon_dir = row.get(roles.get("ship_lon_dir", ""))
                                ship_lat = parse_coordinate(s_lat, direction=s_lat_dir, mode=coord_mode)
                                ship_lon = parse_coordinate(s_lon, direction=s_lon_dir, mode=coord_mode)

                                if ship_lat is not None and ship_lon is not None:
                                    batch.append((ts_ns, ship_gps_ch, json.dumps(
                                        {"timestamp": json_ts, "latitude": ship_lat, "longitude": ship_lon}).encode(
                                        'utf-8')))

                                s_heading = safe_float(row.get(roles.get("ship_heading", "")))
                                if s_heading is not None:
                                    batch.append((ts_ns, ship_tel_ch,
                                                  json.dumps({"timestamp": json_ts, "heading": s_heading}).encode(
                                                      'utf-8')))

                                # Dynamic Telemetry Payloads
                                tel_payload = {"timestamp": json_ts}
                                for orig_col, out_key in tel_mapping.items():
                                    val_str = row.get(orig_col, "")
                                    if val_str != "":
                                        val_float = safe_float(val_str)
                                        tel_payload[out_key] = val_float if val_float is not None else val_str
                                batch.append((ts_ns, rov_tel_ch, json.dumps(tel_payload).encode('utf-8')))

                                if len(batch) > 5000:
                                    flush_to_db(batch)
                                    batch = []
                            except:
                                continue
                flush_to_db(batch)

            # 2. DigiStills
            image_files = scan_results.get("image_files", [])
            if image_files:
                log(f"Encoding {len(image_files)} high-res DigiStill photos...", 20)
                batch = []
                for img_path in image_files:
                    fname = os.path.basename(img_path)
                    match = re.search(r"(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})", fname)
                    if match:
                        norm_str = match.group(1).replace('_', '-').split('-')
                        dt_str = f"{norm_str[0]}-{norm_str[1]}-{norm_str[2]} {norm_str[3]}:{norm_str[4]}:{norm_str[5]}"
                        dt = parse_robust_timestamp(dt_str)
                        if dt:
                            ts_ns = int(dt.timestamp() * 1e9)

                            if min_valid_ns and (ts_ns < min_valid_ns or ts_ns > max_valid_ns):
                                continue

                            json_ts = {"sec": int(dt.timestamp()), "nsec": 0}
                            with open(img_path, "rb") as f_img:
                                b64_str = base64.b64encode(f_img.read()).decode('utf-8')
                                batch.append((ts_ns, img_ch, json.dumps(
                                    {"timestamp": json_ts, "frame_id": "rov_camera", "format": "jpeg",
                                     "data": b64_str}).encode('utf-8')))
                                batch.append((ts_ns, img_log_ch, json.dumps(
                                    {"timestamp": json_ts, "level": 3, "message": f"Photo Taken: {fname}",
                                     "name": "DigiStills"}).encode('utf-8')))
                flush_to_db(batch)

            # 3. Parallel Video
            raw_video_list = scan_results.get("video_files", [])
            filtered_videos = [v for v in raw_video_list if v[1] in selected_cams]
            if filtered_videos:
                cpu_cores = target_cores if target_cores > 0 else (os.cpu_count() or 4)
                log(f"Decoding {len(filtered_videos)} videos using {cpu_cores} active M5 Cores...", 30)

                tasks = [(v[0], v[1], width, height, quality, sample_sec, worker_q) for v in filtered_videos]
                completed = 0
                with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
                    future_to_vid = {executor.submit(extract_video_frames_worker, t): t[1] for t in tasks}
                    for future in as_completed(future_to_vid):
                        cam_name = future_to_vid[future]
                        try:
                            frames = future.result() or []
                            ch_id = vid_channels[cam_name]
                            batch = [(ts_ns, ch_id, payload) for ts_ns, payload in frames]
                            flush_to_db(batch)
                        except Exception as e:
                            log(f"Warning: Video error ({e})")

                        completed += 1
                        pct = 30 + int((completed / len(filtered_videos)) * 50)
                        log(f"Finished compiling video batch [{completed}/{len(filtered_videos)}]", pct)

            # 4. OFOP Logs
            ofop_files = scan_results.get("ofop_files", [])
            if ofop_files:
                log(f"Processing {len(ofop_files)} OFOP protocol file(s)...", 85)
                batch = []
                for ofop_path in ofop_files:
                    with open(ofop_path, "r", encoding="utf-8", errors="ignore") as f_in:
                        first_line = f_in.readline()
                        delim = '\t' if '\t' in first_line else (';' if ';' in first_line else ',')
                        f_in.seek(0)
                        reader = csv.DictReader(f_in, delimiter=delim)
                        success_count = 0

                        for row in reader:
                            try:
                                # Look for variations of Date/Time
                                date_val = row.get('Date', '') or row.get('#Date', '')
                                time_val = row.get('Time', '')
                                time_str = row.get('Timestamp', '') or row.get('SystemTime', '')

                                if not time_str and date_val and time_val:
                                    time_str = f"{date_val} {time_val}"

                                dt = parse_robust_timestamp(time_str)

                                if not dt: continue

                                ts_ns = int(dt.timestamp() * 1e9)

                                if min_valid_ns and (ts_ns < min_valid_ns or ts_ns > max_valid_ns):
                                    continue

                                json_ts = {"sec": int(dt.timestamp()), "nsec": int(ts_ns % 1e9)}

                                msg_parts = []
                                if ofop_mapping:
                                    for k in ofop_mapping:
                                        v = row.get(k, "")
                                        if v: msg_parts.append(f"{k}: {v}")
                                else:
                                    msg_parts = [f"{k}: {v}" for k, v in row.items() if
                                                 k not in ['Date', '#Date', 'Time', 'Timestamp', 'SystemTime'] and v]

                                if msg_parts:
                                    batch.append((ts_ns, ofop_log_ch, json.dumps(
                                        {"timestamp": json_ts, "level": 4, "message": " | ".join(msg_parts),
                                         "name": "OFOP Observer"}).encode('utf-8')))
                                    success_count += 1
                            except Exception as e:
                                continue

                print(f"✅ Successfully encoded {success_count} OFOP entries!")
                flush_to_db(batch)

            # 5. Zero-RAM SQLite Write
            log("Chronologically sorting records from SQLite Spool (Zero-RAM)...", 90)
            c.execute("SELECT ts, ch_id, data FROM messages ORDER BY ts ASC")

            log("Streaming sorted records into MCAP file...", 95)
            for row in c:
                writer.add_message(channel_id=row[1], log_time=row[0], publish_time=row[0], data=row[2])

            writer.finish()
            conn.close()

        log("✅ Universal MCAP build completed successfully!", 100)
        conversion_state["status"] = "completed"

    except Exception as e:
        log(f"❌ Conversion failed with error: {str(e)}", 100)
        conversion_state["status"] = "error"
    finally:
        worker_q.put("STOP")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except:
                pass


# ==========================================
# FLASK WEB INTERFACE & API ENDPOINTS
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ROV MCAP Builder | Universal Vessel Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        pre, code { font-family: 'JetBrains Mono', monospace; }
        .preview-pulse { animation: pulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .4; } }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 xl:h-screen xl:overflow-hidden flex flex-col">
    <div class="max-w-[1900px] w-full mx-auto px-4 py-4 flex flex-col h-full">

        <!-- HEADER -->
        <header class="flex items-center justify-between border-b border-slate-800 pb-3 mb-4 shrink-0">
            <div class="flex items-center space-x-3">
                <div class="w-9 h-9 bg-cyan-500/10 border border-cyan-500/30 rounded-lg flex items-center justify-center text-cyan-400 font-bold text-lg">⚓</div>
                <div>
                    <h1 class="text-lg font-bold text-white tracking-tight">ROV Mission to MCAP Pipeline</h1>
                    <p class="text-[11px] text-slate-400">Universal Vessel Adapter & Multi-Core Hardware Decoding (GEOMAR)</p>
                </div>
            </div>
            <span class="inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 uppercase tracking-wider">
                M5 Multi-Core Studio
            </span>
        </header>

        <!-- MAIN 3-COLUMN LAYOUT -->
        <div class="grid grid-cols-1 xl:grid-cols-12 gap-4 flex-1 xl:min-h-0">

            <!-- ========================================================================= -->
            <!-- COLUMN 1: SETUP & EXECUTE (3/12) -->
            <!-- ========================================================================= -->
            <div class="xl:col-span-3 flex flex-col gap-4 xl:min-h-0">

                <!-- Paths -->
                <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl shrink-0">
                    <h2 class="text-xs font-semibold text-cyan-400 uppercase tracking-wider mb-3 flex items-center gap-2"><span>📁</span> 1. Data, Video & Output Paths</h2>
                    <div class="space-y-3">
                        <div>
                            <label class="block text-[10px] font-medium text-slate-400 mb-1">Local Data Folder</label>
                            <div class="flex gap-2">
                                <input id="dataPath" type="text" class="flex-1 bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-500 font-mono" placeholder="e.g. ./data/MSM145">
                                <button onclick="browsePath('dataPath', 'folder')" class="bg-slate-800 hover:bg-slate-700 border border-slate-600 px-2.5 rounded-md">📁</button>
                            </div>
                        </div>
                        <div>
                            <label class="block text-[10px] font-medium text-slate-400 mb-1">Video Source Folder (External Drive)</label>
                            <div class="flex gap-2">
                                <input id="videoPath" type="text" class="flex-1 bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-500 font-mono" placeholder="e.g. /Volumes/SSD/Videos">
                                <button onclick="browsePath('videoPath', 'folder')" class="bg-slate-800 hover:bg-slate-700 border border-slate-600 px-2.5 rounded-md">📁</button>
                            </div>
                        </div>
                        <button onclick="scanDirectory()" id="scanBtn" class="w-full bg-slate-800 hover:bg-slate-700 border border-slate-600 text-white text-[11px] font-semibold px-4 py-2 rounded-md transition duration-200">
                            🔍 Scan & Unpack Both Directories
                        </button>
                        <div class="pt-3 border-t border-slate-800">
                            <label class="block text-[10px] font-medium text-slate-400 mb-1">Output MCAP File Location</label>
                            <div class="flex gap-2">
                                <input id="outputPath" type="text" class="flex-1 bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-500 font-mono" placeholder="e.g. ./output/064-1ROV06/064-1ROV06.mcap">
                                <button onclick="browsePath('outputPath', 'save')" class="bg-slate-800 hover:bg-slate-700 border border-slate-600 px-2.5 rounded-md">💾</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Hardware & Encoding -->
                <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl shrink-0">
                    <h2 class="text-xs font-semibold text-cyan-400 uppercase tracking-wider mb-3 flex items-center gap-2"><span>⚡</span> 4. Hardware & Encoding Controls</h2>
                    <div class="grid grid-cols-2 gap-3 mb-3">
                        <div>
                            <label class="block text-[10px] font-medium text-slate-400 mb-1">Parallel M5 Cores</label>
                            <select id="coresSelect" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-200">
                                <option value="4">4 (Safe HDD)</option>
                                <option value="8" selected>8 (Fast SSD)</option>
                                <option value="12">12 (M5 Pro)</option>
                                <option value="0">Max Available</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-[10px] font-medium text-slate-400 mb-1">Preview Resolution</label>
                            <select id="resSelect" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-200">
                                <option value="640x360" selected>640 × 360 (Fastest)</option>
                                <option value="960x540">960 × 540 (Balanced)</option>
                                <option value="1280x720">1280 × 720 (High Res)</option>
                            </select>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-[10px] font-medium text-slate-400 mb-1">JPEG Quality (<span id="qualityVal">55</span>%)</label>
                            <input id="qualityRange" type="range" min="30" max="90" value="55" oninput="document.getElementById('qualityVal').innerText = this.value" class="w-full accent-cyan-500">
                        </div>
                        <div>
                            <label class="block text-[10px] font-medium text-slate-400 mb-1">Sample Interval</label>
                            <select id="sampleRate" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-200">
                                <option value="1.0" selected>1 frame / sec</option>
                                <option value="2.0">1 frame / 2 sec (2x Speed)</option>
                                <option value="5.0">1 frame / 5 sec (Fastest)</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Console -->
                <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl flex flex-col xl:flex-1 xl:min-h-0">
                    <h2 class="text-xs font-semibold text-cyan-400 uppercase tracking-wider mb-2 shrink-0"><span>📊</span> Pipeline Status</h2>
                    <div class="space-y-1.5 mb-2 shrink-0">
                        <div class="flex justify-between text-[10px] font-mono">
                            <span id="statusText" class="text-slate-400">Idle</span>
                            <span id="percentText" class="text-cyan-400 font-bold">0%</span>
                        </div>
                        <div class="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden border border-slate-800">
                            <div id="progressBar" class="bg-gradient-to-r from-cyan-500 to-blue-500 h-full w-0 transition-all duration-300 rounded-full"></div>
                        </div>
                    </div>
                    <div id="consoleLog" class="bg-slate-950 border border-slate-800/80 rounded-md p-2 flex-1 overflow-y-auto text-[9px] text-slate-400 space-y-1 font-mono h-32 xl:h-auto">
                        <div class="text-slate-600">Console ready...</div>
                    </div>
                </div>

            </div>

            <!-- ========================================================================= -->
            <!-- COLUMN 2: DATA & MAPPINGS (4/12) -->
            <!-- ========================================================================= -->
            <div class="xl:col-span-4 flex flex-col gap-4 xl:min-h-0">

                <!-- Detected Assets -->
                <div id="discoveryCard" class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl shrink-0 hidden">
                    <h2 class="text-xs font-semibold text-cyan-400 uppercase tracking-wider mb-3 flex items-center gap-2"><span>🔍</span> 2. Detected Mission Assets</h2>
                    <div class="mb-3">
                        <label class="block text-[10px] font-medium text-slate-400 mb-1">Detected Dive ID</label>
                        <select id="diveSelect" onchange="scanDirectory(this.value)" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-500 font-mono"></select>
                    </div>
                    <div class="grid grid-cols-4 gap-2 mb-3">
                        <div class="bg-slate-950 p-1.5 rounded-lg text-center border border-slate-800/50"><span class="text-[9px] text-slate-500 block">CSVs</span><span id="csvCount" class="text-xs font-bold font-mono">0</span></div>
                        <div class="bg-slate-950 p-1.5 rounded-lg text-center border border-slate-800/50"><span class="text-[9px] text-slate-500 block">Stills</span><span id="imgCount" class="text-xs font-bold font-mono">0</span></div>
                        <div class="bg-slate-950 p-1.5 rounded-lg text-center border border-slate-800/50"><span class="text-[9px] text-slate-500 block">OFOP</span><span id="ofopCount" class="text-xs font-bold font-mono">0</span></div>
                        <div class="bg-slate-950 p-1.5 rounded-lg text-center border border-slate-800/50"><span class="text-[9px] text-slate-500 block">Videos</span><span id="vidCount" class="text-xs font-bold text-cyan-400 font-mono">0</span></div>
                    </div>
                    <div>
                        <label class="block text-[10px] font-medium text-slate-400 mb-1">Camera Feeds</label>
                        <div id="cameraList" class="space-y-1 max-h-24 overflow-y-auto pr-1"></div>
                    </div>
                </div>

                <!-- Navigation Roles & Data Mapping -->
                <div id="mappingCard" class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl flex-col xl:flex-1 xl:min-h-0 xl:overflow-y-auto hidden">
                    <h2 class="text-xs font-semibold text-cyan-400 uppercase tracking-wider mb-3 flex items-center gap-2 shrink-0"><span>🧭</span> 3. Navigation Roles & Data Mapping</h2>

                    <!-- Warning Banner -->
                    <div class="mb-3 bg-amber-500/10 border border-amber-500/20 rounded-lg p-2.5 flex items-start gap-2 shrink-0">
                        <span class="text-amber-400 text-sm leading-none mt-0.5">⚠️</span>
                        <p class="text-[10px] text-amber-200/90 leading-snug">
                            <strong>Warning:</strong> Danger Zone! Don't change these navigation roles if you have no clue. And even if you have one, think and check twice! Incorrect coordinate mappings will break the coordinate system in Lichtblick.
                        </p>
                    </div>

                    <!-- Coord Mode -->
                    <div class="mb-3 flex justify-between items-center bg-slate-950 border border-slate-800 rounded-lg p-2 shrink-0">
                        <span class="text-[10px] font-semibold text-slate-300">Coordinate Interpretation Mode</span>
                        <select id="coordModeSelect" class="bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-[10px] text-cyan-300">
                            <option value="auto">Auto-Detect</option>
                            <option value="ddmm">NMEA DDMM.mmmm</option>
                            <option value="decimal">Decimal Degrees</option>
                        </select>
                    </div>

                    <!-- ROV/Ship Nav Roles -->
                    <div class="grid grid-cols-1 gap-3 mb-4 shrink-0">
                        <div class="bg-slate-950 border border-slate-800 rounded-lg p-3">
                            <h3 class="text-[10px] font-bold text-cyan-400 mb-2 border-b border-slate-800/80 pb-1">🤖 ROV Roles</h3>
                            <div class="grid grid-cols-2 gap-x-2 gap-y-1.5 text-xs">
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Latitude</label><select id="role_rov_lat" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Latitude Dir. (N/S)</label><select id="role_rov_lat_dir" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Longitude</label><select id="role_rov_lon" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Longitude Dir. (E/W)</label><select id="role_rov_lon_dir" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Depth</label><select id="role_rov_depth" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Heading</label><select id="role_rov_heading" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Pitch</label><select id="role_rov_pitch" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Roll</label><select id="role_rov_roll" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                            </div>
                        </div>

                        <div class="bg-slate-950 border border-slate-800 rounded-lg p-3">
                            <h3 class="text-[10px] font-bold text-cyan-400 mb-2 border-b border-slate-800/80 pb-1">🚢 Ship Roles</h3>
                            <div class="grid grid-cols-2 gap-x-2 gap-y-1.5 text-xs">
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Latitude</label><select id="role_ship_lat" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Latitude Dir. (N/S)</label><select id="role_ship_lat_dir" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Longitude</label><select id="role_ship_lon" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div><label class="text-[9px] text-slate-500 block mb-0.5">Longitude Dir. (E/W)</label><select id="role_ship_lon_dir" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                                <div class="col-span-2"><label class="text-[9px] text-slate-500 block mb-0.5">Heading</label><select id="role_ship_heading" class="w-full bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-[9px] text-slate-200 font-mono"></select></div>
                            </div>
                        </div>
                    </div>

                    <!-- Telemetry & OFOP (Can grow) -->
                    <div class="space-y-4 shrink-0 pb-2">
                        <div>
                            <span class="text-[10px] font-semibold text-slate-300 block mb-1">📡 Telemetry Config</span>
                            <div id="telMappingContainer" class="space-y-1 max-h-40 overflow-y-auto p-1.5 bg-slate-950 rounded-lg border border-slate-800"></div>
                        </div>
                        <div>
                            <span class="text-[10px] font-semibold text-slate-300 block mb-1">📝 OFOP Protocol</span>
                            <div id="ofopMappingContainer" class="space-y-1 max-h-40 overflow-y-auto p-1.5 bg-slate-950 rounded-lg border border-slate-800"></div>
                        </div>
                    </div>
                </div>

            </div>

            <!-- ========================================================================= -->
            <!-- COLUMN 3: PROCESS PREVIEW (5/12) -->
            <!-- ========================================================================= -->
            <div class="xl:col-span-5 flex flex-col gap-4 xl:min-h-0">

                <!-- Action Button at the Top -->
                <button onclick="startConversion()" id="startBtn" class="shrink-0 w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold py-4 rounded-xl transition duration-200 shadow-xl shadow-cyan-900/30 text-sm tracking-wide uppercase disabled:opacity-50">
                    Build MCAP File
                </button>

                <!-- Process Preview Matrix -->
                <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl flex flex-col xl:flex-1 xl:min-h-0">
                    <div class="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-3 shrink-0">
                        <h2 class="text-xs font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                            <span>👁️</span> Live Processing Preview
                        </h2>
                        <span id="activeWorkersBadge" class="text-[9px] font-mono px-2 py-0.5 rounded-md bg-slate-800 text-slate-400 border border-slate-700">
                            0 Cores
                        </span>
                    </div>

                    <div id="coreGridContainer" class="grid grid-cols-1 sm:grid-cols-2 gap-3 flex-1 overflow-y-auto pr-1 h-[600px] xl:h-auto content-start">
                        <div id="gridPlaceholder" class="col-span-full flex flex-col items-center justify-center h-full text-slate-600 font-mono text-xs border border-dashed border-slate-800 rounded-xl min-h-[300px]">
                            <span class="text-3xl mb-2 opacity-40">🎬</span>
                            Awaiting job start...
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- JAVASCRIPT FUNCTIONS -->
    <script>
        let scanData = null;
        let activeCoreSet = new Set();
        let savedConfig = { coord_mode: "auto", roles: {}, telemetry: {}, ofop: [] };

        fetch("/api/config").then(r => r.json()).then(data => { savedConfig = data; });

        function appendLog(msg) {
            const b = document.getElementById("consoleLog");
            const div = document.createElement("div");
            div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            b.appendChild(div);
            b.scrollTop = b.scrollHeight;
        }

        async function browsePath(inputId, type) {
            try {
                const res = await fetch(`/api/browse?type=${type}`);
                const data = await res.json();
                if (data.path) { document.getElementById(inputId).value = data.path; }
            } catch (err) {}
        }

        async function scanDirectory(diveOverride = null) {
            const dataPath = document.getElementById("dataPath").value;
            const videoPath = document.getElementById("videoPath").value;

            appendLog(`Scanning & Extracting ZIPs in Data Path...`);
            document.getElementById("scanBtn").innerText = "⏳ Extracting & Scanning...";
            document.getElementById("scanBtn").disabled = true;

            try {
                let url = `/api/scan?data_path=${encodeURIComponent(dataPath)}&video_path=${encodeURIComponent(videoPath)}`;
                if (diveOverride) url += `&dive=${encodeURIComponent(diveOverride)}`;

                const res = await fetch(url);
                const data = await res.json();

                if (data.error) {
                    appendLog(`❌ Scan Error: ${data.error}`);
                    return;
                }

                scanData = data;
                document.getElementById("discoveryCard").classList.remove("hidden");
                document.getElementById("csvCount").innerText = data.csv_count;
                document.getElementById("imgCount").innerText = data.image_count;
                document.getElementById("ofopCount").innerText = data.ofop_count;
                document.getElementById("vidCount").innerText = data.video_count;

                const diveSelect = document.getElementById("diveSelect");
                diveSelect.innerHTML = "";
                data.detected_dives.forEach(d => {
                    const opt = document.createElement("option");
                    opt.value = d;
                    opt.innerText = d;
                    if (d === data.selected_dive) opt.selected = true;
                    diveSelect.appendChild(opt);
                });

                const camList = document.getElementById("cameraList");
                camList.innerHTML = "";
                data.cameras.forEach(cam => {
                    camList.innerHTML += `
                        <label class="flex items-center space-x-2 bg-slate-950 p-1.5 rounded-lg border border-slate-800 text-xs font-mono">
                            <input type="checkbox" checked value="${cam}" class="cam-checkbox rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0">
                            <span class="truncate">${cam}</span>
                        </label>
                    `;
                });
        
                renderMappings(data.tel_headers, data.ofop_headers);
                if (data.selected_dive) {
                    document.getElementById("outputPath").value = `./output/${data.selected_dive}/${data.selected_dive}.mcap`;
                }

                appendLog(`✅ Scan Complete for: ${data.selected_dive || 'Dive'}`);
            } catch (err) {
                appendLog(`❌ Error scanning: ${err.message}`);
            } finally {
                document.getElementById("scanBtn").innerText = "🔍 Scan & Unpack Both Directories";
                document.getElementById("scanBtn").disabled = false;
            }
        }

        function populateRoleDropdown(elId, headers, keywords, currentVal) {
            const el = document.getElementById(elId);
            el.innerHTML = "<option value=''>-- Not Assigned --</option>";
            headers.forEach(h => {
                const opt = document.createElement("option");
                opt.value = h;
                opt.innerText = h;
                if (currentVal === h) opt.selected = true;
                el.appendChild(opt);
            });
            if (!currentVal) {
                for (let h of headers) {
                    const hLow = h.toLowerCase();
                    if (keywords.some(k => hLow.includes(k))) {
                        el.value = h;
                        break;
                    }
                }
            }
        }

        function renderMappings(telHeaders, ofopHeaders) {
            document.getElementById("mappingCard").classList.remove("hidden");
            document.getElementById("mappingCard").classList.add("flex");
            if (savedConfig.coord_mode) document.getElementById("coordModeSelect").value = savedConfig.coord_mode;

            const r = savedConfig.roles || {};
            populateRoleDropdown("role_rov_lat", telHeaders, ["usbl", "rov.lat", "rov_lat", "lat"], r.rov_lat);
            populateRoleDropdown("role_rov_lat_dir", telHeaders, ["latns", "lat_dir", "lat.ns"], r.rov_lat_dir);
            populateRoleDropdown("role_rov_lon", telHeaders, ["usbl", "rov.lon", "rov_lon", "lon"], r.rov_lon);
            populateRoleDropdown("role_rov_lon_dir", telHeaders, ["lonew", "lon_dir", "lon.ew"], r.rov_lon_dir);
            populateRoleDropdown("role_rov_depth", telHeaders, ["depth", "tiefe"], r.rov_depth);
            populateRoleDropdown("role_rov_heading", telHeaders, ["heading", "kurs", "yaw"], r.rov_heading);
            populateRoleDropdown("role_rov_pitch", telHeaders, ["pitch", "nick"], r.rov_pitch);
            populateRoleDropdown("role_rov_roll", telHeaders, ["roll"], r.rov_roll);

            populateRoleDropdown("role_ship_lat", telHeaders, ["sysposlat", "ship.lat", "shiplat"], r.ship_lat);
            populateRoleDropdown("role_ship_lat_dir", telHeaders, ["sysposlatn", "shiplatns", "latn"], r.ship_lat_dir);
            populateRoleDropdown("role_ship_lon", telHeaders, ["sysposlon", "ship.lon", "shiplon"], r.ship_lon);
            populateRoleDropdown("role_ship_lon_dir", telHeaders, ["sysposlonw", "shiplonew", "lonw"], r.ship_lon_dir);
            populateRoleDropdown("role_ship_heading", telHeaders, ["ship.heading", "sysheading", "shipheading"], r.ship_heading);

            const telContainer = document.getElementById("telMappingContainer");
            telContainer.innerHTML = "";
            telHeaders.forEach((h, i) => {
                const hLow = h.toLowerCase();
                if (hLow.includes("time") || hLow.includes("date")) return;
                const isChecked = savedConfig.telemetry && savedConfig.telemetry.hasOwnProperty(h);
                const outVal = (savedConfig.telemetry && savedConfig.telemetry[h]) || hLow.replace(/[^a-z0-9]/g, '_');

                telContainer.innerHTML += `
                    <div class="flex items-center gap-2">
                        <input type="checkbox" id="tel_chk_${i}" value="${h}" ${isChecked ? 'checked' : ''} class="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0">
                        <span class="text-[9px] text-slate-400 w-1/2 truncate" title="${h}">${h}</span>
                        <span class="text-slate-600 text-xs">➔</span>
                        <input type="text" id="tel_out_${i}" value="${outVal}" class="flex-1 bg-slate-950 border border-slate-800 rounded px-1.5 py-0.5 text-[9px] text-cyan-300 font-mono">
                    </div>
                `;
            });

            const ofopContainer = document.getElementById("ofopMappingContainer");
            ofopContainer.innerHTML = "";
            ofopHeaders.forEach((h, i) => {
                const isChecked = !savedConfig.ofop || savedConfig.ofop.length === 0 || savedConfig.ofop.includes(h);
                ofopContainer.innerHTML += `
                    <div class="flex items-center gap-2">
                        <input type="checkbox" id="ofop_chk_${i}" value="${h}" ${isChecked ? 'checked' : ''} class="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0">
                        <span class="text-[9px] text-slate-400 w-full truncate" title="${h}">${h}</span>
                    </div>
                `;
            });
        }

        function updateCoreGridTile(data) {
            const container = document.getElementById("coreGridContainer");
            const placeholder = document.getElementById("gridPlaceholder");
            if (placeholder) placeholder.remove();

            activeCoreSet.add(data.core_id);
            document.getElementById("activeWorkersBadge").innerText = `${activeCoreSet.size} Cores`;

            let tile = document.getElementById(`grid-tile-${data.core_id}`);
            if (!tile) {
                tile = document.createElement("div");
                tile.id = `grid-tile-${data.core_id}`;
                tile.className = "bg-slate-950 border border-slate-800 rounded-xl overflow-hidden p-2.5 shadow-lg flex flex-col justify-between";
                tile.innerHTML = `
                    <div class="flex items-center justify-between text-[10px] font-mono mb-1.5 pb-1 border-b border-slate-800">
                        <span class="flex items-center gap-1 font-semibold text-cyan-400"><span class="w-1.5 h-1.5 rounded-full bg-cyan-400 preview-pulse"></span>Core #${data.core_num}</span>
                        <span class="text-slate-400 truncate max-w-[100px]">${data.cam || ''}</span>
                    </div>
                    <div class="aspect-video bg-black/90 rounded-lg overflow-hidden relative flex items-center justify-center border border-slate-800/50 mb-1.5">
                        <img id="tile-img-${data.core_id}" class="w-full h-full object-contain ${data.b64 ? '' : 'hidden'}" src="${data.b64 ? 'data:image/jpeg;base64,' + data.b64 : ''}" />
                        <span id="tile-ph-${data.core_id}" class="text-[9px] font-mono text-slate-600 ${data.b64 ? 'hidden' : ''}">Streaming...</span>
                    </div>
                    <div>
                        <div class="flex justify-between text-[9px] font-mono mb-1">
                            <span id="tile-label-${data.core_id}" class="truncate text-slate-300 pr-2 max-w-[140px]">${data.label}</span>
                            <span id="tile-pct-${data.core_id}" class="text-cyan-400 font-bold">${data.progress}%</span>
                        </div>
                        <div class="w-full bg-slate-900 h-1 rounded-full overflow-hidden">
                            <div id="tile-bar-${data.core_id}" class="bg-cyan-500 h-full transition-all duration-300 rounded-full" style="width: ${data.progress}%"></div>
                        </div>
                    </div>
                `;
                container.appendChild(tile);
            } else {
                if (data.b64) {
                    const img = document.getElementById(`tile-img-${data.core_id}`);
                    img.src = "data:image/jpeg;base64," + data.b64;
                    img.classList.remove("hidden");
                    document.getElementById(`tile-ph-${data.core_id}`).classList.add("hidden");
                }
                document.getElementById(`tile-label-${data.core_id}`).innerText = data.label;
                document.getElementById(`tile-pct-${data.core_id}`).innerText = `${data.progress}%`;
                document.getElementById(`tile-bar-${data.core_id}`).style.width = `${data.progress}%`;
            }
        }

        async function startConversion() {
            if (!scanData) { alert("Please scan a folder first!"); return; }

            const selectedCams = Array.from(document.querySelectorAll(".cam-checkbox:checked")).map(cb => cb.value);
            const [width, height] = document.getElementById("resSelect").value.split("x").map(Number);
            const quality = parseInt(document.getElementById("qualityRange").value);
            const sampleSec = parseFloat(document.getElementById("sampleRate").value);
            const outputPath = document.getElementById("outputPath").value;
            const targetCores = parseInt(document.getElementById("coresSelect").value);

            // Harvest Custom Config
            const config = {
                coord_mode: document.getElementById("coordModeSelect").value,
                roles: {
                    rov_lat: document.getElementById("role_rov_lat").value,
                    rov_lat_dir: document.getElementById("role_rov_lat_dir").value,
                    rov_lon: document.getElementById("role_rov_lon").value,
                    rov_lon_dir: document.getElementById("role_rov_lon_dir").value,
                    rov_depth: document.getElementById("role_rov_depth").value,
                    rov_heading: document.getElementById("role_rov_heading").value,
                    rov_pitch: document.getElementById("role_rov_pitch").value,
                    rov_roll: document.getElementById("role_rov_roll").value,
                    ship_lat: document.getElementById("role_ship_lat").value,
                    ship_lat_dir: document.getElementById("role_ship_lat_dir").value,
                    ship_lon: document.getElementById("role_ship_lon").value,
                    ship_lon_dir: document.getElementById("role_ship_lon_dir").value,
                    ship_heading: document.getElementById("role_ship_heading").value
                },
                telemetry: {},
                ofop: []
            };

            scanData.tel_headers.forEach((h, i) => {
                const chk = document.getElementById(`tel_chk_${i}`);
                if (chk && chk.checked) config.telemetry[h] = document.getElementById(`tel_out_${i}`).value;
            });

            scanData.ofop_headers.forEach((h, i) => {
                const chk = document.getElementById(`ofop_chk_${i}`);
                if (chk && chk.checked) config.ofop.push(h);
            });

            document.getElementById("startBtn").disabled = true;
            document.getElementById("coreGridContainer").innerHTML = ""; 
            activeCoreSet.clear();
            appendLog("Launching conversion across M5 cores...");

            await fetch("/api/convert", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    scan_results: scanData, selected_cams: selectedCams, output_mcap: outputPath,
                    width: width, height: height, quality: quality, sample_sec: sampleSec,
                    target_cores: targetCores, config: config
                })
            });

            const evtSource = new EventSource("/api/progress");
            evtSource.onmessage = function(e) {
                const data = JSON.parse(e.data);
                if (data.type === 'log') {
                    document.getElementById("progressBar").style.width = `${data.percent}%`;
                    document.getElementById("percentText").innerText = `${data.percent}%`;
                    document.getElementById("statusText").innerText = data.message;
                    appendLog(data.message);
                    if (data.percent >= 100) {
                        evtSource.close();
                        document.getElementById("startBtn").disabled = false;
                    }
                } else if (data.type === 'video_progress') {
                    updateCoreGridTile(data);
                }
            };
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index(): return render_template_string(HTML_TEMPLATE)


@app.route("/api/browse")
def api_browse():
    path = macos_browse(is_save=(request.args.get("type", "folder") == "save")) if sys.platform == 'darwin' else ""
    return jsonify({"path": path})


@app.route("/api/config")
def api_config():
    return jsonify(load_mapping_config())


@app.route("/api/scan")
def api_scan():
    data_path = request.args.get("data_path", "")
    if not data_path or not os.path.exists(data_path): return jsonify({"error": f"Path not found: {data_path}"}), 400
    return jsonify(scan_dive_directory(data_path, request.args.get("video_path", ""), request.args.get("dive", None)))


@app.route("/api/convert", methods=["POST"])
def api_convert():
    req = request.json
    config = req.get("config", {})
    save_mapping_config(config)

    t = threading.Thread(
        target=run_conversion_task,
        args=(
            req["scan_results"], req["selected_cams"], req["output_mcap"],
            req["width"], req["height"], req["quality"], req["sample_sec"],
            req.get("target_cores", 8), config
        )
    )
    t.daemon = True
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/progress")
def api_progress():
    def event_stream():
        while True:
            try:
                data = progress_queue.get(timeout=20)
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("percent", 0) >= 100: break
            except queue.Empty:
                yield f"data: {json.dumps(conversion_state)}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    print("\n🌐 Starting Universal Vessel Studio...")
    print("👉 Open your browser at: http://127.0.0.1:5000\n")
    app.run(port=5000, debug=False)
