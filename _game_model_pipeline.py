"""
Game Model Training Pipeline — Downloads gameplay screenshots, auto-labels, trains YOLOv8.
Supports: Sniper Elite 5 (SE5) and Warhammer 40K: Space Marine 2 (SM2)

Usage:
  python _game_model_pipeline.py se5       # full pipeline for Sniper Elite 5
  python _game_model_pipeline.py sm2       # full pipeline for Space Marine 2
  python _game_model_pipeline.py both      # do both games
  python _game_model_pipeline.py se5 download   # just download SE5 images
  python _game_model_pipeline.py sm2 label      # just label SM2 images
"""

import os, sys, glob, shutil, time
import cv2
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────
GAMES = {
    "se5": {
        "name": "Sniper Elite 5",
        "img_dir": "_gamedata/se5/images_v2",
        "label_dir": "_gamedata/se5/labels_v2",
        "train_dir": "_gamedata/se5/train_v2",
        "model_out": "Model/Sniper_Elite_5.onnx",
        "search_queries": [
            "sniper elite 5 gameplay screenshot",
            "sniper elite 5 enemy soldiers gameplay",
            "sniper elite 5 combat gameplay screenshot",
            "sniper elite 5 multiplayer gameplay",
            "sniper elite 5 sniper scope gameplay",
            "sniper elite 5 invasion mode gameplay",
            "sniper elite 5 nazi soldiers targets",
            "sniper elite 5 stealth kill gameplay",
        ],
    },
    "sm2": {
        "name": "Space Marine 2",
        "img_dir": "_gamedata/sm2/images_v2",
        "label_dir": "_gamedata/sm2/labels_v2",
        "train_dir": "_gamedata/sm2/train_v2",
        "model_out": "Model/Space_Marine_2.onnx",
        "search_queries": [
            "space marine 2 gameplay screenshot",
            "space marine 2 tyranid combat gameplay",
            "space marine 2 enemies gameplay screenshot",
            "space marine 2 horde mode gameplay",
            "space marine 2 PvP gameplay screenshot",
            "space marine 2 operations gameplay",
            "warhammer space marine 2 battle screenshot",
            "space marine 2 multiplayer combat",
        ],
    },
    "plaz": {
        "name": "Project Lazarus (Roblox)",
        "img_dir": "_gamedata/plaz/images_v2",
        "label_dir": "_gamedata/plaz/labels_v2",
        "train_dir": "_gamedata/plaz/train_v2",
        "model_out": "Model/Project_Lazarus.onnx",
        "search_queries": [
            "project lazarus roblox gameplay screenshot",
            "project lazarus roblox zombies gameplay",
            "project lazarus roblox zombie horde",
            "project lazarus roblox shooting zombies",
            "project lazarus roblox gameplay 2024",
            "project lazarus roblox first person",
            "roblox project lazarus zombie survival gameplay",
            "roblox project lazarus wave gameplay",
        ],
    },
}

MIN_IMAGE_SIZE = 10000  # bytes — skip tiny thumbnails
TARGET_IMAGES_PER_QUERY = 30  # aim for ~200+ total per game


# ─── STEP 1: Download screenshots via icrawler ──────────────
def download_screenshots(game_key):
    cfg = GAMES[game_key]
    img_dir = cfg["img_dir"]
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DOWNLOADING: {cfg['name']} gameplay screenshots")
    print(f"{'='*60}")

    try:
        from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
    except ImportError:
        print("ERROR: icrawler not installed. Run: pip install icrawler")
        return 0

    total_before = len(glob.glob(os.path.join(img_dir, "*.*")))

    for i, query in enumerate(cfg["search_queries"]):
        print(f"\n  Query {i+1}/{len(cfg['search_queries'])}: '{query}'")
        try:
            # Use Bing (more reliable than Google for crawling)
            crawler = BingImageCrawler(
                storage={"root_dir": img_dir},
                log_level=40,  # ERROR only — suppress noise
            )
            crawler.crawl(
                keyword=query,
                max_num=TARGET_IMAGES_PER_QUERY,
                min_size=(400, 400),  # skip tiny images
                file_idx_offset="auto",
            )
        except Exception as e:
            print(f"    Bing failed: {e}, trying Google...")
            try:
                crawler = GoogleImageCrawler(
                    storage={"root_dir": img_dir},
                    log_level=40,
                )
                crawler.crawl(
                    keyword=query,
                    max_num=TARGET_IMAGES_PER_QUERY,
                )
            except Exception as e2:
                print(f"    Google also failed: {e2}")
        time.sleep(1)  # polite delay

    # Clean up tiny / broken images
    cleaned = _clean_images(img_dir)
    total_after = len(glob.glob(os.path.join(img_dir, "*.*")))
    print(f"\n  Total images after download: {total_after} ({cleaned} removed as junk)")
    return total_after


def _clean_images(img_dir):
    """Remove images that are too small or can't be opened by OpenCV."""
    removed = 0
    for fpath in glob.glob(os.path.join(img_dir, "*.*")):
        try:
            size = os.path.getsize(fpath)
            if size < MIN_IMAGE_SIZE:
                os.remove(fpath)
                removed += 1
                continue
            img = cv2.imread(fpath)
            if img is None:
                os.remove(fpath)
                removed += 1
                continue
            h, w = img.shape[:2]
            if w < 400 or h < 300:
                os.remove(fpath)
                removed += 1
        except Exception:
            try:
                os.remove(fpath)
            except:
                pass
            removed += 1
    return removed


# ─── STEP 2: Auto-label with YOLOv8 COCO 'person' detector ──
def auto_label(game_key):
    """Auto-label images.  For PLAZ (Roblox zombies) uses a specialised
    multi-strategy labeler because the COCO 'person' detector misses
    most of the blocky zombie shapes.  For all other games the standard
    COCO person detector at conf=0.20 is used."""

    cfg = GAMES[game_key]
    img_dir = cfg["img_dir"]
    label_dir = cfg["label_dir"]
    os.makedirs(label_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AUTO-LABELING: {cfg['name']}")
    print(f"{'='*60}")

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    images = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    if not images:
        print("  No images found! Run download step first.")
        return 0

    # Choose labeling strategy
    if game_key == "plaz":
        return _auto_label_plaz(model, images, label_dir)
    else:
        return _auto_label_standard(model, images, label_dir)


def _auto_label_standard(model, images, label_dir):
    """Standard labeler: COCO 'person' class at conf=0.20."""
    labeled = 0
    total_boxes = 0
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")

        results = model(img_path, verbose=False, conf=0.20)
        lines = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                img_h, img_w = r.orig_shape
                bw_pct = (x2 - x1) / img_w
                bh_pct = (y2 - y1) / img_h
                if bw_pct * bh_pct > 0.50:
                    continue
                if bw_pct < 0.01 or bh_pct < 0.01:
                    continue
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        if lines:
            labeled += 1
            total_boxes += len(lines)

    print(f"  Images with detections: {labeled}/{len(images)}")
    print(f"  Total bounding boxes: {total_boxes}")
    return labeled


# ─── PLAZ-SPECIFIC: Multi-strategy zombie labeler ────────────
def _detect_glowing_eyes(img):
    """Find bright yellow/green glowing eye blobs in a PLAZ screenshot.
    Returns a list of (x1, y1, x2, y2) body-sized bounding boxes
    estimated by pairing nearby eye blobs and expanding downward."""

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow glow: H=15-45, high saturation, high brightness
    mask_y = cv2.inRange(hsv, np.array([15, 100, 180]), np.array([45, 255, 255]))
    # Green glow: H=45-80
    mask_g = cv2.inRange(hsv, np.array([45, 80, 180]), np.array([80, 255, 255]))
    mask = mask_y | mask_g

    # Dilate a bit to merge close pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep eye-sized blobs (between 0.005% and 1.5% of image area)
    min_area = (w * h) * 0.00005
    max_area = (w * h) * 0.015
    eye_blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = bx + bw / 2
            cy = by + bh / 2
            eye_blobs.append((cx, cy, bw, bh, area))

    # Pair nearby blobs horizontally (zombie has 2 eyes side by side)
    used = set()
    pairs = []
    for i, (cx1, cy1, bw1, bh1, a1) in enumerate(eye_blobs):
        if i in used:
            continue
        best_j = -1
        best_dist = float("inf")
        for j, (cx2, cy2, bw2, bh2, a2) in enumerate(eye_blobs):
            if j <= i or j in used:
                continue
            # Eyes should be at similar Y, close X
            dy = abs(cy1 - cy2)
            dx = abs(cx1 - cx2)
            # Max vertical difference = 1.5x the average eye height
            avg_h = (bh1 + bh2) / 2
            if dy > avg_h * 1.5:
                continue
            # Horizontal distance: 0.5x to 4x the average eye width
            avg_w = (bw1 + bw2) / 2
            if dx < avg_w * 0.3 or dx > avg_w * 6:
                continue
            dist = dx + dy
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0:
            used.add(i)
            used.add(best_j)
            pairs.append((eye_blobs[i], eye_blobs[best_j]))

    # Build body boxes from eye pairs
    boxes = []
    for (cx1, cy1, bw1, bh1, _), (cx2, cy2, bw2, bh2, _) in pairs:
        # Head center
        head_cx = (cx1 + cx2) / 2
        head_cy = (cy1 + cy2) / 2
        eye_span = abs(cx1 - cx2) + max(bw1, bw2)

        # Estimate body size from eye span
        # Head width ≈ 1.8x eye span, body height ≈ 3.5x head width
        head_w = eye_span * 1.8
        body_h = head_w * 3.0
        body_w = head_w * 1.6

        # Box: head is top ~30% of body
        bx1 = head_cx - body_w / 2
        by1 = head_cy - head_w * 0.6  # a bit above eyes
        bx2 = head_cx + body_w / 2
        by2 = by1 + body_h

        # Clamp to image bounds
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(w, bx2)
        by2 = min(h, by2)

        boxes.append((bx1, by1, bx2, by2))

    # Also create boxes for unpaired single eye blobs (distant/partially-visible zombies)
    for i, (cx, cy, bw, bh, area) in enumerate(eye_blobs):
        if i in used:
            continue
        # Single eye → smaller estimated body
        head_w = bw * 3.5
        body_h = head_w * 2.8
        body_w = head_w * 1.5

        bx1 = max(0, cx - body_w / 2)
        by1 = max(0, cy - head_w * 0.5)
        bx2 = min(w, cx + body_w / 2)
        by2 = min(h, by1 + body_h)

        boxes.append((bx1, by1, bx2, by2))

    return boxes


def _nms_boxes(boxes, iou_threshold=0.4):
    """Simple non-maximum suppression on a list of (x1,y1,x2,y2) boxes."""
    if not boxes:
        return []
    arr = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    # Sort by area (larger = more likely correct body box)
    order = areas.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        remaining = np.where(iou < iou_threshold)[0]
        order = order[remaining + 1]

    return [boxes[i] for i in keep]


def _auto_label_plaz(model, images, label_dir):
    """Specialised PLAZ labeler combining 3 strategies:
    1. Glowing-eye detection (the distinctive yellow eyes)
    2. Ultra-low-confidence YOLO 'person' detection (conf=0.05)
    3. Multi-class COCO detection for blocky shapes (cat, dog, horse, teddy bear)
    All results are merged with NMS."""

    # COCO classes that frequently fire on blocky Roblox zombies
    EXTRA_CLASSES = {15, 16, 17, 77}  # cat, dog, horse, teddy bear

    labeled = 0
    total_boxes = 0
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            with open(label_path, "w") as f:
                f.write("")
            continue

        img_h, img_w = img.shape[:2]
        all_boxes = []

        # ── Strategy 1: Glowing-eye detection ──
        eye_boxes = _detect_glowing_eyes(img)
        all_boxes.extend(eye_boxes)

        # ── Strategy 2+3: YOLO at very low confidence ──
        results = model(img_path, verbose=False, conf=0.05)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # Accept 'person' OR the extra blocky-shape classes
                if cls_id != 0 and cls_id not in EXTRA_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw_pct = (x2 - x1) / img_w
                bh_pct = (y2 - y1) / img_h
                # Skip huge boxes (covers >40% of image)
                if bw_pct * bh_pct > 0.40:
                    continue
                # Skip microscopic boxes
                if bw_pct < 0.015 or bh_pct < 0.015:
                    continue
                all_boxes.append((x1, y1, x2, y2))

        # ── Merge with NMS ──
        final_boxes = _nms_boxes(all_boxes, iou_threshold=0.35)

        # Convert to YOLO format
        lines = []
        for (x1, y1, x2, y2) in final_boxes:
            bw_pct = (x2 - x1) / img_w
            bh_pct = (y2 - y1) / img_h
            # Final sanity: skip if still too big or too tiny
            if bw_pct * bh_pct > 0.50 or bw_pct < 0.01 or bh_pct < 0.01:
                continue
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        if lines:
            labeled += 1
            total_boxes += len(lines)

    print(f"  [PLAZ multi-strategy] Images with detections: {labeled}/{len(images)}")
    print(f"  Total bounding boxes: {total_boxes}")
    print(f"  Detection rate: {labeled/len(images)*100:.1f}%")
    return labeled


# ─── STEP 3: Prepare train/val split and train locally ───────
def train_model(game_key):
    cfg = GAMES[game_key]
    img_dir = cfg["img_dir"]
    label_dir = cfg["label_dir"]
    train_dir = cfg["train_dir"]

    print(f"\n{'='*60}")
    print(f"  TRAINING: {cfg['name']}")
    print(f"{'='*60}")

    # Clear and recreate train/val split directories
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(train_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "labels", split), exist_ok=True)

    # Get images that have labels with at least 1 box
    valid_pairs = []
    images = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        if os.path.exists(label_path):
            with open(label_path) as f:
                content = f.read().strip()
            if content:
                valid_pairs.append((img_path, label_path))

    if len(valid_pairs) < 5:
        print(f"  ERROR: Only {len(valid_pairs)} labeled images. Need at least 5.")
        return None

    # 80/20 split
    np.random.seed(42)
    indices = np.random.permutation(len(valid_pairs))
    split_idx = int(len(valid_pairs) * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    for idx_set, split in [(train_idx, "train"), (val_idx, "val")]:
        for idx in idx_set:
            img_path, label_path = valid_pairs[idx]
            base = os.path.basename(img_path)
            lbase = os.path.splitext(base)[0] + ".txt"
            shutil.copy2(img_path, os.path.join(train_dir, "images", split, base))
            shutil.copy2(label_path, os.path.join(train_dir, "labels", split, lbase))

    print(f"  Train images: {len(train_idx)}")
    print(f"  Val images:   {len(val_idx)}")

    # Write data.yaml
    data_yaml = os.path.join(train_dir, "data.yaml")
    abs_train_dir = os.path.abspath(train_dir).replace("\\", "/")
    with open(data_yaml, "w") as f:
        f.write(f"path: {abs_train_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['enemy']\n")

    # Train YOLOv8n
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    safe_name = game_key + "_v2"

    print(f"  Starting training (50 epochs, 640px)...")
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        name=safe_name,
        project=f"runs/detect/_game_runs",
        patience=10,
        pretrained=True,
        verbose=True,
        amp=False,
    )
    print(f"  Training complete!")
    return safe_name


# ─── STEP 4: Export to ONNX ─────────────────────────────────
def export_model(game_key, run_name=None):
    cfg = GAMES[game_key]
    model_out = cfg["model_out"]

    print(f"\n{'='*60}")
    print(f"  EXPORTING: {cfg['name']} → {model_out}")
    print(f"{'='*60}")

    if run_name is None:
        run_name = game_key + "_enemy"

    best_pt = f"runs/detect/_game_runs/{run_name}/weights/best.pt"
    if not os.path.exists(best_pt):
        # Try with numeric suffix
        candidates = sorted(glob.glob(f"runs/detect/_game_runs/{run_name}*/weights/best.pt"))
        if candidates:
            best_pt = candidates[-1]
        else:
            print(f"  ERROR: No best.pt found at {best_pt}")
            return False

    from ultralytics import YOLO
    model = YOLO(best_pt)
    model.export(format="onnx", opset=12, simplify=True, imgsz=640)

    onnx_path = best_pt.replace(".pt", ".onnx")
    if os.path.exists(onnx_path):
        os.makedirs("Model", exist_ok=True)
        shutil.copy2(onnx_path, model_out)
        size_mb = os.path.getsize(model_out) / (1024 * 1024)
        print(f"  SUCCESS: {model_out} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ERROR: ONNX export didn't produce expected file")
        return False


# ─── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    game = sys.argv[1].lower()
    step = sys.argv[2].lower() if len(sys.argv) > 2 else "all"

    if game == "both":
        game_keys = ["se5", "sm2"]
    elif game in GAMES:
        game_keys = [game]
    else:
        print(f"Unknown game: {game}. Use: se5, sm2, or both")
        sys.exit(1)

    for gk in game_keys:
        print(f"\n{'#'*60}")
        print(f"  GAME: {GAMES[gk]['name']}")
        print(f"{'#'*60}")

        if step in ("download", "all"):
            download_screenshots(gk)
        if step in ("label", "all"):
            auto_label(gk)
        if step in ("train", "all"):
            run_name = train_model(gk)
        if step in ("export", "all"):
            rn = run_name if step == "all" else None
            export_model(gk, rn)

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
