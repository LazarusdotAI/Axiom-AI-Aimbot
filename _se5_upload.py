"""Upload SE5 images with VOC XML annotations to Roboflow and train."""
import requests, os, glob, json, base64, time, sys

API_KEY = "0evoZYUzmc3xGyhjJ9Vj"
WORKSPACE = "stephens-workspace-upbkb"
PROJECT_SLUG = "se5-enemy-detect"
IMG_DIR = "_se5_data/images"
XML_DIR = "_se5_data/xml_labels"

def list_projects():
    r = requests.get(f"https://api.roboflow.com/{WORKSPACE}?api_key={API_KEY}", timeout=15)
    data = r.json()
    print("Workspace:", data.get("workspace", {}).get("name"))
    for p in data.get("projects", []):
        print(f"  Project: {p['id']} | images={p.get('images',0)} | unannotated={p.get('unannotated',0)}")
    return data

def create_project():
    r = requests.post(
        f"https://api.roboflow.com/{WORKSPACE}/projects?api_key={API_KEY}",
        json={"name": "SE5 Enemies", "type": "object-detection", "license": "MIT"},
        timeout=15,
    )
    print("Create project:", r.status_code, r.text[:500])
    return r.json()

def upload_images():
    """Upload images with annotations using Roboflow upload API."""
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    uploaded = 0
    skipped = 0
    
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(XML_DIR, base + ".xml")
        if not os.path.exists(xml_path):
            skipped += 1
            continue
        
        # Read image as base64
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Read annotation
        with open(xml_path, "r") as f:
            annotation = f.read()
        
        # Upload via REST API
        upload_url = (
            f"https://api.roboflow.com/dataset/{PROJECT_SLUG}/upload"
            f"?api_key={API_KEY}"
            f"&name={base}.jpg"
            f"&split=train"
        )
        
        try:
            r = requests.post(
                upload_url,
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
            result = r.json()
            
            if result.get("success") or r.status_code == 200:
                # Now upload annotation
                img_id = result.get("id", base)
                annot_url = (
                    f"https://api.roboflow.com/dataset/{PROJECT_SLUG}/annotate/{img_id}"
                    f"?api_key={API_KEY}"
                    f"&name={base}.xml"
                )
                r2 = requests.post(
                    annot_url,
                    data=annotation,
                    headers={"Content-Type": "text/plain"},
                    timeout=30,
                )
                uploaded += 1
                if uploaded % 10 == 0:
                    print(f"  Uploaded {uploaded}...")
            else:
                print(f"  Failed {base}: {r.text[:200]}")
        except Exception as e:
            print(f"  Error {base}: {e}")
    
    print(f"Total uploaded: {uploaded}, skipped: {skipped}")

def generate_and_train():
    """Generate a version and start training."""
    # Generate version
    r = requests.post(
        f"https://api.roboflow.com/{WORKSPACE}/{PROJECT_SLUG}/generate?api_key={API_KEY}",
        json={"settings": {
            "preprocessing": {"auto-orient": True, "resize": {"width": 640, "height": 640, "format": "Stretch to"}},
            "augmentation": {"flip": {"horizontal": True}, "brightness": {"min": -15, "max": 15}},
        }},
        timeout=30,
    )
    print("Generate version:", r.status_code, r.text[:500])
    
    # Wait for generation
    time.sleep(15)
    
    # Start training
    r = requests.post(
        f"https://api.roboflow.com/{WORKSPACE}/{PROJECT_SLUG}/1/train?api_key={API_KEY}",
        json={"model_type": "yolov8n"},
        timeout=30,
    )
    print("Train:", r.status_code, r.text[:500])

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "list"
    if step == "list":
        list_projects()
    elif step == "create":
        create_project()
    elif step == "upload":
        upload_images()
    elif step == "train":
        generate_and_train()
    elif step == "all":
        create_project()
        upload_images()
        generate_and_train()

