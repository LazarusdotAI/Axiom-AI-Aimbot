import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core.roboflow_utils import download_roboflow_model

path, classes = download_roboflow_model(
    workspace="fortnite-ai-aim",
    project="cod-mw-warzone-catlb",
    version=1,
    api_key="0evoZYUzmc3xGyhjJ9Vj",
)
print(f"\nDone! Model saved to: {path}")
print(f"Classes: {classes}")

