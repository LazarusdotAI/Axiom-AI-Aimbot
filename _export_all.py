"""Export all trained game models to ONNX and copy to Model/ folder."""
from ultralytics import YOLO
import shutil, os

games = {
    "se5": {
        "weights": "runs/detect/runs/detect/_game_runs/se5_v2/weights/best.pt",
        "out": "Model/Sniper_Elite_5.onnx",
    },
    "sm2": {
        "weights": "runs/detect/runs/detect/_game_runs/sm2_v2/weights/best.pt",
        "out": "Model/Space_Marine_2.onnx",
    },
    "plaz": {
        "weights": "runs/detect/runs/detect/_game_runs/plaz_v2/weights/best.pt",
        "out": "Model/Project_Lazarus.onnx",
    },
}

for key, cfg in games.items():
    wpath = cfg["weights"]
    if not os.path.exists(wpath):
        # Try alternate path without double runs
        alt = wpath.replace("runs/detect/runs/detect/", "runs/detect/")
        if os.path.exists(alt):
            wpath = alt
        else:
            print(f"SKIP {key}: weights not found at {cfg['weights']}")
            continue

    print(f"\n--- Exporting {key} ---")
    print(f"  Weights: {wpath}")
    model = YOLO(wpath)
    onnx_path = model.export(format="onnx", opset=12, simplify=True)
    print(f"  Exported to: {onnx_path}")

    os.makedirs("Model", exist_ok=True)
    shutil.copy2(onnx_path, cfg["out"])
    size = os.path.getsize(cfg["out"])
    print(f"  Copied to: {cfg['out']} ({size/1024/1024:.1f} MB)")

print("\n=== ALL EXPORTS COMPLETE ===")
for key, cfg in games.items():
    if os.path.exists(cfg["out"]):
        sz = os.path.getsize(cfg["out"]) / 1024 / 1024
        print(f"  {cfg['out']}: {sz:.1f} MB")
    else:
        print(f"  {cfg['out']}: MISSING")

