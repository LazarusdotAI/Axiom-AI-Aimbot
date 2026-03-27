# inference.py
"""AI inference module - Image preprocessing, post-processing, and PID controller"""

from __future__ import annotations

from typing import List, Tuple, Any

import cv2
import numpy as np
import numpy.typing as npt


class PIDController:
    """PID Controller - used for smooth aiming movement
    
    Implements Proportional-Integral-Derivative (PID) control algorithm for calculating mouse movement.
    Supports independent X/Y axis settings and includes dynamic P-parameter adjustment.
    
    Attributes:
        Kp: Proportional coefficient, controls reaction speed
        Ki: Integral coefficient, corrects static error
        Kd: Derivative coefficient, suppresses jitter and overshoot
    """
    
    def __init__(self, Kp: float, Ki: float, Kd: float) -> None:
        self.Kp = Kp  # Proportional
        self.Ki = Ki  # Integral
        self.Kd = Kd  # Derivative
        self.reset()

    def reset(self) -> None:
        """Reset controller state"""
        self.integral: float = 0.0
        self.previous_error: float = 0.0

    def update(self, error: float) -> float:
        """
        Calculates control output based on current error
        
        Args:
            error: Current error (e.g., target_x - current_x)
            
        Returns:
            Control amount (e.g., amount mouse should move)
        """
        # Integral term (with anti-windup clamping)
        self.integral += error
        self.integral = max(-1000.0, min(1000.0, self.integral))
        
        # Derivative term
        derivative = error - self.previous_error
        
        # Adjust P parameter response curve
        adjusted_kp = self._calculate_adjusted_kp(self.Kp)
        
        # Calculate output
        output = (adjusted_kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        # Update previous error
        self.previous_error = error
        
        return output
    
    def _calculate_adjusted_kp(self, kp: float) -> float:
        """Calculate dynamically adjusted P parameter
        
        Implements non-linear P parameter response curve:
        - 0% ~ 50%: Linear growth, maintains original proportion
        - 50% ~ 100%: Accelerated growth, eventually scaling to 200%
        
        This design allows for smoother low sensitivity and more aggressive high sensitivity.
        
        Args:
            kp: Original P parameter value (0.0 ~ 1.0)
            
        Returns:
            Adjusted P parameter value (0.0 ~ 2.0)
        """
        if kp <= 0.5:
            return kp
        else:
            # When kp=0.5, output=0.5; when kp=1.0, output=2.0
            return 0.5 + (kp - 0.5) * 3.0


def preprocess_image(
    image: npt.NDArray[np.uint8],
    model_input_size: int,
    model_type: str = "auto",
) -> npt.NDArray[np.float32]:
    """
    Preprocess image to fit ONNX model.

    Args:
        image: Input image (BGRA or BGR format)
        model_input_size: Model input size (square edge)
        model_type: ``"yolov8"`` / ``"auto"`` → direct resize (fast).
                    ``"yolov5"`` → letterbox resize (preserves aspect
                    ratio, pads with gray 114).

    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # For YOLOv5 letterbox path we need BGR first, then letterbox, then blob.
    if model_type == "yolov5":
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = _letterbox(image, model_input_size)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(model_input_size, model_input_size),
            swapRB=True,
            crop=False,
        )
        return np.ascontiguousarray(blob, dtype=np.float32)

    # ── Fast path (YOLOv8 / auto) ──
    # Skip the separate BGRA→BGR conversion — blobFromImage handles the
    # channel swap via swapRB.  Also skip the manual cv2.resize since
    # blobFromImage already resizes to (model_input_size, model_input_size).
    # This eliminates ~1-2ms of redundant work per frame.
    #
    # If the input is BGRA (4 channels), drop alpha first with a fast
    # slice instead of cvtColor (avoids a full array copy).
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # drop alpha channel — near-zero cost

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 255.0,
        size=(model_input_size, model_input_size),
        swapRB=True,
        crop=False,
    )

    return np.ascontiguousarray(blob, dtype=np.float32)


def _letterbox(
    image: npt.NDArray[np.uint8],
    target_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> npt.NDArray[np.uint8]:
    """Resize ``image`` while preserving aspect ratio, padding remainder.

    This matches the YOLOv5 letterbox behaviour used in the Matias Kovero
    repo and most YOLOv5-trained models.

    Args:
        image: BGR input image (H, W, 3).
        target_size: Square output edge length.
        color: Padding fill colour (default: gray 114).

    Returns:
        Padded square image of shape ``(target_size, target_size, 3)``.
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas and center the resized image on it
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas


def _detect_model_format(raw_output: np.ndarray) -> str:
    """Auto-detect whether an ONNX output tensor is YOLOv5 or YOLOv8 format.

    YOLOv8 shape: ``(1, 4+C, N)`` — rows are attributes, columns are
    detections.  ``N`` (number of detections) is typically >> ``4+C``.

    YOLOv5 shape: ``(1, N, 5+C)`` — rows are detections, columns are
    attributes.  ``5+C`` is small (e.g. 6 for single-class with obj_conf).

    Heuristic:
      YOLOv8 ``(1, 4+C, N)`` → dim1 is small (attributes), dim2 is large (N).
      YOLOv5 ``(1, N, 5+C)`` → dim1 is large (N), dim2 is small (attributes).

    We identify YOLOv5 only when dim1 is clearly the "many detections"
    axis (>= 20) AND dim2 is clearly the "few attributes" axis (<= 20).
    This avoids false positives on small test tensors.

    Returns:
        ``"yolov8"`` or ``"yolov5"``
    """
    if raw_output.ndim == 3:
        _, dim1, dim2 = raw_output.shape
        # YOLOv5: many rows (N >= 20), few columns (5+C <= ~20)
        if dim1 >= 20 and dim2 <= 20 and dim1 > dim2:
            return "yolov5"
        return "yolov8"
    if raw_output.ndim == 2:
        rows, cols = raw_output.shape
        if rows >= 20 and cols <= 20 and rows > cols:
            return "yolov5"
        return "yolov8"
    return "yolov8"  # default


def postprocess_outputs(
    outputs: List[Any],
    original_width: int,
    original_height: int,
    model_input_size: int,
    min_confidence: float,
    offset_x: int = 0,
    offset_y: int = 0
) -> Tuple[List[List[float]], List[float], List[int]]:
    """
    後處理 ONNX 模型輸出 — supports YOLOv5, YOLOv8, and multi-class models.

    **YOLOv8** (default / legacy):
        Output shape ``(1, 4+C, N)``.
        Columns (after transpose): ``[cx, cy, w, h, cls0, cls1, ...]``.
        Coordinates are center-width-height, normalised to model_input_size.

    **YOLOv5** (e.g. Matias Kovero / Splitgate):
        Output shape ``(1, N, 5+C)``.
        Columns: ``[x1, y1, x2, y2, obj_conf, cls0, cls1, ...]``.
        Coordinates are already **xyxy** in model_input_size pixel space.
        The effective per-class confidence is ``obj_conf * cls_conf``.

    Single-class:
        All returned ``class_ids`` will be ``0``.

    Multi-class:
        ``confidence`` = max across classes; ``class_id`` = argmax.

    Args:
        outputs: 模型輸出
        original_width: 原始圖像寬度
        original_height: 原始圖像高度
        model_input_size: 模型輸入尺寸
        min_confidence: 最小置信度閾值
        offset_x: X 軸偏移
        offset_y: Y 軸偏移

    Returns:
        ``(boxes, confidences, class_ids)`` 三元組
    """
    raw = outputs[0]  # shape: (1, ?, ?) typically
    fmt = _detect_model_format(raw)

    if fmt == "yolov5":
        return _postprocess_yolov5(
            raw, original_width, original_height,
            model_input_size, min_confidence, offset_x, offset_y,
        )

    # ── YOLOv8 path (original) ──
    predictions = raw[0].T  # shape: (N, 4+C) or (N, 5)
    num_cols = predictions.shape[1]
    num_classes = num_cols - 4  # at least 1

    if num_classes <= 0:
        return [], [], []

    if num_classes == 1:
        # ── Single-class model (legacy path) ──
        confs = predictions[:, 4]
        class_ids_all = np.zeros(len(predictions), dtype=np.int32)
    else:
        # ── Multi-class model ──
        class_scores = predictions[:, 4:]          # shape (N, C)
        confs = class_scores.max(axis=1)           # best class score per detection
        class_ids_all = class_scores.argmax(axis=1).astype(np.int32)

    # 向量化過濾：先篩選高置信度的檢測
    conf_mask = confs >= min_confidence
    filtered_predictions = predictions[conf_mask]
    filtered_confs = confs[conf_mask]
    filtered_class_ids = class_ids_all[conf_mask]

    if len(filtered_predictions) == 0:
        return [], [], []

    # 向量化計算邊界框
    scale_x = original_width / model_input_size
    scale_y = original_height / model_input_size

    cx, cy, w, h = (filtered_predictions[:, 0], filtered_predictions[:, 1],
                    filtered_predictions[:, 2], filtered_predictions[:, 3])

    x1 = (cx - w / 2) * scale_x + offset_x
    y1 = (cy - h / 2) * scale_y + offset_y
    x2 = (cx + w / 2) * scale_x + offset_x
    y2 = (cy + h / 2) * scale_y + offset_y

    boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
    confidences = filtered_confs.tolist()
    class_ids = filtered_class_ids.tolist()

    return boxes, confidences, class_ids


def _postprocess_yolov5(
    raw: np.ndarray,
    original_width: int,
    original_height: int,
    model_input_size: int,
    min_confidence: float,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Tuple[List[List[float]], List[float], List[int]]:
    """Post-process a **YOLOv5** ONNX output tensor.

    YOLOv5 output shape: ``(1, N, 5+C)``
        Columns: ``[cx, cy, w, h, obj_conf, cls0_conf, cls1_conf, ...]``

    * Coordinates are **center-point + width/height** in model-input
      pixel space (e.g. 0‥640).  They are converted to corner format
      ``[x1, y1, x2, y2]`` before scaling.
    * ``obj_conf`` is the objectness score.
    * Per-class confidence = ``obj_conf * cls_conf``.
    """
    predictions = raw[0]  # (N, 5+C) — already rows=detections
    if predictions.ndim != 2 or predictions.shape[1] < 5:
        return [], [], []

    num_cols = predictions.shape[1]
    num_classes = num_cols - 5  # subtract cx,cy,w,h,obj_conf

    obj_conf = predictions[:, 4]

    if num_classes <= 0:
        # Edge case: exactly 5 columns → treat obj_conf as confidence
        confs = obj_conf
        class_ids_all = np.zeros(len(predictions), dtype=np.int32)
    elif num_classes == 1:
        confs = obj_conf * predictions[:, 5]
        class_ids_all = np.zeros(len(predictions), dtype=np.int32)
    else:
        class_scores = predictions[:, 5:] * obj_conf[:, None]  # (N, C)
        confs = class_scores.max(axis=1)
        class_ids_all = class_scores.argmax(axis=1).astype(np.int32)

    # Filter by confidence
    conf_mask = confs >= min_confidence
    filtered_predictions = predictions[conf_mask]
    filtered_confs = confs[conf_mask]
    filtered_class_ids = class_ids_all[conf_mask]

    if len(filtered_predictions) == 0:
        return [], [], []

    # Convert center-point (cx, cy, w, h) → corner (x1, y1, x2, y2)
    # then scale from model-input space to original image space.
    scale_x = original_width / model_input_size
    scale_y = original_height / model_input_size

    cx = filtered_predictions[:, 0]
    cy = filtered_predictions[:, 1]
    w  = filtered_predictions[:, 2]
    h  = filtered_predictions[:, 3]

    bx1 = (cx - w / 2) * scale_x + offset_x
    by1 = (cy - h / 2) * scale_y + offset_y
    bx2 = (cx + w / 2) * scale_x + offset_x
    by2 = (cy + h / 2) * scale_y + offset_y

    boxes = np.stack([bx1, by1, bx2, by2], axis=1).tolist()
    confidences = filtered_confs.tolist()
    class_ids = filtered_class_ids.tolist()

    return boxes, confidences, class_ids


def non_max_suppression(
    boxes: List[List[float]],
    confidences: List[float],
    iou_threshold: float = 0.4,
    class_ids: List[int] | None = None,
) -> Tuple[List[List[float]], List[float], List[int]]:
    """
    非極大值抑制

    Now also carries ``class_ids`` through unchanged so callers can
    keep track of which class each surviving detection belongs to.

    Args:
        boxes: 邊界框列表
        confidences: 置信度列表
        iou_threshold: IoU 閾值
        class_ids: (optional) per-detection class index list.
                   If *None*, returns all-zero class IDs for backward
                   compatibility with single-class models.

    Returns:
        ``(filtered_boxes, filtered_confidences, filtered_class_ids)`` 三元組
    """
    if len(boxes) == 0:
        return [], [], []

    boxes_arr = np.array(boxes)
    confidences_arr = np.array(confidences)
    if class_ids is not None:
        class_ids_arr = np.array(class_ids, dtype=np.int32)
    else:
        class_ids_arr = np.zeros(len(boxes), dtype=np.int32)

    areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
    order = confidences_arr.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(boxes_arr[i, 0], boxes_arr[order[1:], 0])
        yy1 = np.maximum(boxes_arr[i, 1], boxes_arr[order[1:], 1])
        xx2 = np.minimum(boxes_arr[i, 2], boxes_arr[order[1:], 2])
        yy2 = np.minimum(boxes_arr[i, 3], boxes_arr[order[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)  # 防止除零

        order = order[1:][iou <= iou_threshold]

    return (boxes_arr[keep].tolist(),
            confidences_arr[keep].tolist(),
            class_ids_arr[keep].tolist())
