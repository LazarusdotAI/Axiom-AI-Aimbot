"""Roboflow Universe model download & hosted inference utilities.

**Download utility** — downloads YOLOv8 ONNX models from Roboflow
Universe and saves them into the local ``Model/`` directory so they can
be loaded by the inference pipeline like any other ``.onnx`` file.

**Hosted inference adapter** — ``RoboflowInferenceAdapter`` sends
screen-capture frames to the Roboflow hosted API and returns detection
results ``(boxes, confidences, class_ids)`` in the exact same format
the rest of the bot's pipeline expects (matching ``postprocess_outputs``
output).  This is slower than local ONNX but requires no GPU or model
file on disk.

Usage (download):
    >>> from core.roboflow_utils import download_roboflow_model
    >>> path = download_roboflow_model(
    ...     workspace="kwan-li-jqief",
    ...     project="valorant-object-detection2",
    ...     version=1,
    ...     api_key="YOUR_API_KEY",
    ... )

Usage (hosted inference):
    >>> from core.roboflow_utils import RoboflowInferenceAdapter
    >>> adapter = RoboflowInferenceAdapter(
    ...     api_key="YOUR_API_KEY",
    ...     workspace="fortnite-ai-aim",
    ...     project="cod-mw-warzone-catlb",
    ...     version=1,
    ... )
    >>> boxes, confs, class_ids = adapter.detect(frame, offset_x, offset_y)

NOTE: This requires the ``roboflow`` pip package.
      ``pip install roboflow``
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  Known game / dataset presets
#  Maps  (workspace, project)  →  (friendly_name, [class_names])
# ---------------------------------------------------------------------------
ROBOFLOW_PRESETS: Dict[Tuple[str, str], Tuple[str, List[str]]] = {
    # ── COD MW / Warzone ──
    ("fortnite-ai-aim", "cod-mw-warzone-catlb"): (
        "COD_MW_Warzone",
        ["enemy", "head"],
    ),
    # ── Valorant (Kwan Li — 2-class, body + head) ──
    ("kwan-li-jqief", "valorant-object-detection2"): (
        "Valorant_KwanLi",
        ["enemyBody", "enemyHead"],
    ),
    # ── Valorant (Kw4L0r fork — same 2 classes) ──
    ("kw4l0r-pbsii", "valorant-detection-hpoae"): (
        "Valorant_Kw4L0r",
        ["enemyBody", "enemyHead"],
    ),
    # ── Valorant (David Hong — 3 classes) ──
    ("david-hong", "valorant-enemy"): (
        "Valorant_DavidHong",
        ["enemy", "enemy_head", "Valorant-enemy"],
    ),
    # ── Valorant large dataset (many agent classes — not for aiming) ──
    ("valorant-object-detection", "valorant-object-detection-r9qkl"): (
        "Valorant_Agents",
        [],  # too many classes to list; user should check the dataset page
    ),
    # ── Splitgate (Matias Kovero — YOLOv5, single-class "enemy") ──
    # Source: https://github.com/matias-kovero/ObjectDetection
    # Models are .pt (PyTorch), must be exported to ONNX before use.
    # The repo ships best.pt (300 epochs, 600 images) and bestv2.pt
    # (600 epochs total, 1500+ images).
    ("matias-kovero", "splitgate-objectdetection"): (
        "Splitgate_MatiasKovero",
        ["enemy"],
    ),
    # ── CS:GO / CS2 style 4-class models (common YOLOv5 community format) ──
    # Many community-trained YOLOv5 models for CS-style games use these labels.
    # The exact workspace/project may vary — this serves as a template preset.
    ("csgo-community", "csgo-4class"): (
        "CSGO_4Class",
        ["ct_body", "ct_head", "t_body", "t_head"],
    ),
}


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Model")


def download_roboflow_model(
    workspace: str,
    project: str,
    version: int = 1,
    api_key: str = "",
    model_format: str = "yolov8",
    dest_dir: str | None = None,
) -> Tuple[str, List[str]]:
    """Download an ONNX model from Roboflow Universe.

    Returns:
        ``(saved_onnx_path, class_names)``

    Raises:
        ImportError: if the ``roboflow`` package is not installed.
        RuntimeError: if the download or export fails.
    """
    try:
        from roboflow import Roboflow  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'roboflow' package is required.  Install it with:\n"
            "    pip install roboflow"
        )

    if not api_key:
        raise ValueError("A Roboflow API key is required.  Pass it as api_key='...'")

    dest = dest_dir or MODEL_DIR
    os.makedirs(dest, exist_ok=True)

    # Look up preset class names if available.
    preset_key = (workspace, project)
    friendly_name, preset_classes = ROBOFLOW_PRESETS.get(
        preset_key, (project, [])
    )

    print(f"[Roboflow] Connecting to workspace={workspace} project={project} v{version} ...")
    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    rf_version = rf_project.version(version)

    print(f"[Roboflow] Downloading {model_format} model ...")
    dataset = rf_version.download(model_format, location=os.path.join(dest, "_rf_tmp"))

    # The Roboflow SDK typically puts the model in a subdirectory.
    # We look for any .onnx or .pt file and move it up.
    src_dir = dataset.location if hasattr(dataset, "location") else os.path.join(dest, "_rf_tmp")
    onnx_file = None
    for root, _dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith((".onnx", ".pt")):
                onnx_file = os.path.join(root, f)
                break
        if onnx_file:
            break

    if onnx_file is None:
        raise RuntimeError(
            f"No .onnx or .pt file found in {src_dir}.  "
            "You may need to export the model to ONNX format on the Roboflow website first."
        )

    # Move to Model/ with a clean name.
    ext = os.path.splitext(onnx_file)[1]
    final_name = f"{friendly_name}{ext}"
    final_path = os.path.join(dest, final_name)
    shutil.move(onnx_file, final_path)

    # Clean up temp directory.
    tmp_dir = os.path.join(dest, "_rf_tmp")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Read class names from data.yaml if preset is empty.
    class_names = preset_classes
    if not class_names:
        yaml_path = os.path.join(src_dir, "data.yaml")
        if os.path.isfile(yaml_path):
            class_names = _parse_class_names_from_yaml(yaml_path)

    print(f"[Roboflow] Saved model → {final_path}")
    print(f"[Roboflow] Class names: {class_names}")
    print(f'[Roboflow] Set config:  model_path = "{final_path}"')
    print(f'[Roboflow]              model_class_names = {class_names}')

    return final_path, class_names


def _parse_class_names_from_yaml(yaml_path: str) -> List[str]:
    """Best-effort parse of ``names:`` from a Roboflow data.yaml."""
    names: List[str] = []
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            in_names = False
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("names:"):
                    in_names = True
                    continue
                if in_names:
                    if stripped.startswith("- "):
                        names.append(stripped[2:].strip().strip("'\""))
                    elif stripped and not stripped.startswith("#"):
                        break
    except Exception:
        pass
    return names



# =========================================================================
#  Roboflow Hosted Inference Adapter
# =========================================================================

class RoboflowInferenceAdapter:
    """Send frames to the Roboflow hosted API and return detections.

    This class wraps the ``roboflow`` SDK so the rest of the bot can call
    ``adapter.detect(frame, ...)`` and get back ``(boxes, confidences,
    class_ids)`` — the same tuple shape that ``postprocess_outputs``
    returns.  That means every downstream function (NMS, FOV filter,
    aiming, overlay) works without changes.

    **How it works (plain English):**

    1.  You give it a numpy frame (the screenshot the bot captured).
    2.  It saves that frame to a tiny temp file (Roboflow SDK needs a path
        or URL — it cannot accept a raw numpy array directly).
    3.  It calls ``model.predict()`` which sends the image to Roboflow's
        cloud servers and gets back a list of detections (JSON).
    4.  Each detection has a center-x, center-y, width, height, class
        name, and confidence score.
    5.  The adapter converts those into ``[x1, y1, x2, y2]`` boxes and
        maps each class name to a numeric class ID using the preset
        class list from ``ROBOFLOW_PRESETS``.
    6.  It returns the results so the bot can aim at them.

    **Limitations:**
    -  Network latency: each call is a round-trip to Roboflow's servers
       (~200-800 ms depending on connection).  For real-time aiming the
       local ONNX path is far faster.
    -  Requires an internet connection and a valid API key.
    -  Free-tier Roboflow accounts have rate limits.

    Attributes:
        model: The Roboflow model object (set after ``initialize()``).
        class_names: Ordered list of class names for this model.
        confidence: Confidence threshold (0-100) sent to the API.
    """

    def __init__(
        self,
        api_key: str,
        workspace: str = "fortnite-ai-aim",
        project: str = "cod-mw-warzone-catlb",
        version: int = 1,
        confidence: int = 40,
    ) -> None:
        if not api_key:
            raise ValueError(
                "A Roboflow API key is required.  Set 'roboflow_api_key' in "
                "config.json or pass it directly."
            )
        self._api_key = api_key
        self._workspace = workspace
        self._project = project
        self._version = version
        self.confidence = confidence

        # Populated after initialize().
        self.model: Any = None
        # Look up class names from the preset table.
        preset_key = (workspace, project)
        _, self.class_names = ROBOFLOW_PRESETS.get(preset_key, (project, []))
        # Build a reverse map: class-name-string → integer index.
        self._class_name_to_id: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

    # ------------------------------------------------------------------
    #  Lazy initialisation — connect to Roboflow on first use
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Connect to Roboflow and load the model object.

        Call this once before the first ``detect()`` call.  It is
        separated from ``__init__`` so construction is cheap and any
        network errors surface at a clear point.
        """
        try:
            from roboflow import Roboflow  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "The 'roboflow' package is required.  Install it with:\n"
                "    pip install roboflow"
            )

        print(
            f"[Roboflow] Connecting to {self._workspace}/{self._project} "
            f"v{self._version} …"
        )
        rf = Roboflow(api_key=self._api_key)
        rf_project = rf.workspace(self._workspace).project(self._project)
        self.model = rf_project.version(self._version).model
        print("[Roboflow] Model loaded — ready for hosted inference.")

    # ------------------------------------------------------------------
    #  Main detection entry-point
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        min_confidence: float = 0.0,
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """Run detection on a single frame via the Roboflow hosted API.

        Args:
            frame: BGR or BGRA numpy array (the screenshot).
            offset_x: Pixel offset to add to all X coordinates (so boxes
                      are in full-screen space, not capture-region space).
            offset_y: Same for Y.
            min_confidence: Extra local confidence filter (0.0–1.0).
                           The API already filters at ``self.confidence``
                           (0–100) so this is a second, finer pass.

        Returns:
            ``(boxes, confidences, class_ids)`` — same shape as
            ``postprocess_outputs`` so the rest of the pipeline works
            unchanged.
        """
        if self.model is None:
            self.initialize()

        # The Roboflow SDK's predict() needs a file path, not a numpy
        # array.  We write to a tiny temp file, call predict, then
        # delete it.  This adds ~1-2 ms which is negligible next to the
        # network round-trip.
        tmp_path = os.path.join(tempfile.gettempdir(), "_rf_frame.jpg")
        # Convert BGRA → BGR if needed before saving.
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(tmp_path, frame)

        try:
            prediction = self.model.predict(
                tmp_path,
                confidence=self.confidence,
            )
            raw_json = prediction.json()
        finally:
            # Clean up temp file.
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # ----- Parse the JSON predictions into our standard format -----
        predictions_list = raw_json.get("predictions", [])

        boxes: List[List[float]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for pred in predictions_list:
            conf = pred.get("confidence", 0.0)
            if conf < min_confidence:
                continue

            # Roboflow returns center-x, center-y, width, height.
            cx = pred.get("x", 0.0)
            cy = pred.get("y", 0.0)
            w = pred.get("width", 0.0)
            h = pred.get("height", 0.0)

            x1 = cx - w / 2.0 + offset_x
            y1 = cy - h / 2.0 + offset_y
            x2 = cx + w / 2.0 + offset_x
            y2 = cy + h / 2.0 + offset_y

            # Map class name → numeric ID.
            class_name = pred.get("class", "")
            cls_id = self._class_name_to_id.get(class_name, 0)

            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(cls_id)

        return boxes, confidences, class_ids