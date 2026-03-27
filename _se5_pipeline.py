"""
Sniper Elite 5 model training pipeline.

Steps:
1. Download SE5 gameplay screenshots from the web
2. Auto-label them using a pre-trained YOLOv8n (COCO 'person' class)
3. Upload to Roboflow project via API
4. Trigger training
5. Download ONNX model

Usage:  python _se5_pipeline.py [step]
  step = download | label | upload | train | export | all
"""

import os, sys, time, json, glob, hashlib, shutil
import urllib.request
import urllib.error

# Directories
IMG_DIR = "_se5_data/images"
LABEL_DIR = "_se5_data/labels"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

ROBOFLOW_API_KEY = "0evoZYUzmc3xGyhjJ9Vj"
ROBOFLOW_WORKSPACE = None  # will be auto-detected
ROBOFLOW_PROJECT = "sniper-elite-5"

# ─── STEP 1: Download screenshots ────────────────────────────
# We'll search for SE5 gameplay images via Google/Bing-style direct URLs.
# These are publicly available press screenshots and gameplay captures.

SCREENSHOT_URLS = []  # Will be populated dynamically from Steam API


def download_screenshots():
    """Download SE5 screenshots. Also try fetching the Steam store page
    to discover real screenshot URLs."""
    print("\n=== Step 1: Downloading SE5 screenshots ===")

    # First, try to scrape the actual Steam page for real screenshot URLs
    real_urls = _scrape_steam_screenshots()
    all_urls = real_urls if real_urls else SCREENSHOT_URLS

    downloaded = 0
    for i, url in enumerate(all_urls):
        fname = f"se5_{i:03d}.jpg"
        fpath = os.path.join(IMG_DIR, fname)
        if os.path.exists(fpath):
            downloaded += 1
            continue
        try:
            print(f"  Downloading {i+1}/{len(all_urls)}: {url[:80]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                if len(data) < 5000:  # too small, probably an error page
                    print(f"    SKIP (too small: {len(data)} bytes)")
                    continue
                with open(fpath, "wb") as f:
                    f.write(data)
                downloaded += 1
                print(f"    OK ({len(data)//1024} KB)")
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"  Total images: {downloaded}")
    return downloaded


def _scrape_steam_screenshots():
    """Get real screenshot URLs from Steam API (appdetails endpoint)."""
    try:
        url = "https://store.steampowered.com/api/appdetails?appids=1029690"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        screenshots = data["1029690"]["data"]["screenshots"]
        urls = [s["path_full"] for s in screenshots]
        print(f"  Found {len(urls)} screenshot URLs from Steam API")

        # Also add DLC screenshots (SE5 DLC app IDs)
        dlc_ids = [1872750, 1724940, 2053020, 2053021, 1724942, 1724941,
                   2167701, 2167702, 2167703, 1872751]
        for dlc_id in dlc_ids:
            try:
                dlc_url = f"https://store.steampowered.com/api/appdetails?appids={dlc_id}"
                dlc_req = urllib.request.Request(dlc_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(dlc_req, timeout=10) as dlc_resp:
                    dlc_data = json.loads(dlc_resp.read())
                if str(dlc_id) in dlc_data and dlc_data[str(dlc_id)].get("success"):
                    dlc_ss = dlc_data[str(dlc_id)]["data"].get("screenshots", [])
                    urls.extend([s["path_full"] for s in dlc_ss])
            except Exception:
                pass
            time.sleep(0.3)  # rate limiting

        # Also try Sniper Elite Resistance (similar art style, more soldiers)
        try:
            res_url = "https://store.steampowered.com/api/appdetails?appids=2169200"
            res_req = urllib.request.Request(res_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(res_req, timeout=10) as res_resp:
                res_data = json.loads(res_resp.read())
            if "2169200" in res_data and res_data["2169200"].get("success"):
                res_ss = res_data["2169200"]["data"].get("screenshots", [])
                urls.extend([s["path_full"] for s in res_ss])
                print(f"  Added {len(res_ss)} screenshots from SE: Resistance")
        except Exception:
            pass

        # Deduplicate
        urls = list(dict.fromkeys(urls))
        print(f"  Total unique URLs: {len(urls)}")
        return urls
    except Exception as e:
        print(f"  Could not fetch Steam API: {e}")
        return []


# ─── STEP 2: Auto-label with YOLOv8 COCO person detector ─────
def auto_label():
    """Run YOLOv8n (COCO) on each image to detect 'person' (class 0)."""
    print("\n=== Step 2: Auto-labeling with YOLOv8n COCO ===")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # downloads automatically if not present
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not images:
        print("  No images found! Run 'download' step first.")
        return 0

    labeled = 0
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, base + ".txt")

        results = model(img_path, verbose=False, conf=0.25)
        lines = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # 0 = 'person' in COCO
                    continue
                # YOLO format: class cx cy w h (normalized 0-1)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                img_h, img_w = r.orig_shape
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                # Single class "enemy" = class 0
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        if lines:
            labeled += 1
            print(f"  {base}: {len(lines)} enemies detected")
        else:
            print(f"  {base}: no persons detected")


# ─── STEP 3: Upload to Roboflow ──────────────────────────────
def upload_to_roboflow():
    """Create a Roboflow project and upload images + labels."""
    print("\n=== Step 3: Upload to Roboflow ===")
    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    # Get workspace name
    ws = rf.workspace()
    ws_name = ws.name if hasattr(ws, 'name') else str(ws)
    print(f"  Workspace: {ws_name}")

    # Try to get existing project or create new one
    try:
        project = ws.project(ROBOFLOW_PROJECT)
        print(f"  Using existing project: {ROBOFLOW_PROJECT}")
    except Exception:
        print(f"  Creating new project: {ROBOFLOW_PROJECT}")
        project = ws.create_project(
            project_name="Sniper Elite 5",
            project_license="MIT",
            project_type="object-detection",
            annotation=ROBOFLOW_PROJECT,
        )
        print(f"  Project created!")

    # Upload images with annotations
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    uploaded = 0
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, base + ".txt")
        if not os.path.exists(label_path):
            continue
        # Only upload images that have at least 1 detection
        with open(label_path) as f:
            if not f.read().strip():
                continue
        try:
            project.upload(
                image_path=img_path,
                annotation_path=label_path,
                annotation_format="yolov5",
            )
            uploaded += 1
            print(f"  Uploaded {uploaded}: {base}")
        except Exception as e:
            print(f"  Failed {base}: {e}")

    print(f"  Total uploaded: {uploaded}")
    return uploaded


# ─── STEP 4: Generate dataset version & train ────────────────
def train_model():
    """Generate a version and start training on Roboflow."""
    print("\n=== Step 4: Train model on Roboflow ===")
    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(ROBOFLOW_PROJECT)

    # Generate a new version with preprocessing/augmentation
    print("  Generating dataset version with augmentations...")
    try:
        version = project.generate_version(settings={
            "preprocessing": {
                "auto-orient": True,
                "resize": {"width": 640, "height": 640, "format": "Stretch to"},
            },
            "augmentation": {
                "flip": {"horizontal": True},
                "brightness": {"min": -25, "max": 25},
                "blur": {"pixels": 1.5},
            },
        })
        print(f"  Version generated: v{version.version}")
    except Exception as e:
        print(f"  Version generation note: {e}")
        # Try to get latest version
        versions = project.versions()
        if versions:
            version = versions[-1]
            print(f"  Using existing version: v{version.version}")
        else:
            print("  ERROR: No versions available.")
            return None

    # Start training
    print("  Starting YOLOv8n training on Roboflow...")
    try:
        version.train(model_type="yolov8", speed="fast")
        print("  Training started! This usually takes 15-30 minutes.")
        print("  The model will be available for download once complete.")
    except Exception as e:
        print(f"  Training note: {e}")
        print("  You may need to start training from the Roboflow web UI.")

    return version


# ─── STEP 5: Download trained ONNX model ─────────────────────
def download_model():
    """Download the trained model as ONNX."""
    print("\n=== Step 5: Download trained ONNX model ===")
    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(ROBOFLOW_PROJECT)

    # Get the latest trained version
    versions = project.versions()
    trained_version = None
    for v in reversed(versions):
        try:
            model = v.model
            if model:
                trained_version = v
                break
        except Exception:
            continue

    if not trained_version:
        print("  ERROR: No trained model found yet.")
        print("  Training may still be in progress. Try again later.")
        return False

    print(f"  Found trained model: v{trained_version.version}")

    # Download as ONNX
    dst = os.path.join("Model", "Sniper_Elite_5.onnx")
    print(f"  Downloading to {dst}...")
    try:
        trained_version.model.download("onnx", location="Model/")
        # Roboflow may save it with a different name, find and rename
        onnx_files = glob.glob("Model/*.onnx")
        for f in onnx_files:
            if "sniper" in f.lower() or "best" in f.lower():
                if f != dst:
                    shutil.move(f, dst)
                break
        if os.path.exists(dst):
            size_mb = os.path.getsize(dst) / (1024*1024)
            print(f"  SUCCESS: {dst} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  Model downloaded but file location unclear.")
            print(f"  Check Model/ directory for new .onnx files.")
            return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step in ("download", "all"):
        download_screenshots()

    if step in ("label", "all"):
        auto_label()

    if step in ("upload", "all"):
        upload_to_roboflow()

    if step in ("train", "all"):
        train_model()

    if step in ("export", "download_model"):
        download_model()

    if step == "all":
        print("\n" + "="*60)
        print("  Pipeline complete up to training.")
        print("  Training takes 15-30 min on Roboflow.")
        print("  Run: python _se5_pipeline.py export")
        print("  to download the model once training finishes.")
        print("="*60)

