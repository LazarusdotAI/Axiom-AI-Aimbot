from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, List, Tuple

from win_utils import send_mouse_move

from .ai_loop_state import LoopState
from .inference import PIDController
from .smart_tracker import SmartTracker

if TYPE_CHECKING:
    from .config import Config


def calculate_aim_target(
    box: List[float],
    aim_part: str,
    head_height_ratio: float,
    class_name: str | None = None,
) -> Tuple[float, float]:
    """計算瞄準點座標

    When the model itself identifies a detection as a head class (e.g.
    ``class_name`` is ``"head"`` / ``"enemyHead"`` / ``"enemy_head"``),
    we aim directly at the bounding-box center — there's no need to
    estimate the head position with ``head_height_ratio``.

    For body-class detections (or when ``class_name`` is *None* / unknown)
    the original ``aim_part`` + ``head_height_ratio`` logic is used.
    """

    abs_x1, abs_y1, abs_x2, abs_y2 = box
    box_w, box_h = abs_x2 - abs_x1, abs_y2 - abs_y1
    box_center_x = abs_x1 + box_w * 0.5

    # ── Multi-class shortcut: model says this IS a head ──
    if class_name is not None and _is_head_class(class_name):
        return box_center_x, abs_y1 + box_h * 0.5  # center of head box

    # ── Legacy single-class / body-class path ──
    if aim_part == 'head':
        target_x = box_center_x
        target_y = abs_y1 + box_h * head_height_ratio * 0.5
    else:
        target_x = box_center_x
        head_h = box_h * head_height_ratio
        target_y = (abs_y1 + head_h + abs_y2) * 0.5

    return target_x, target_y


# Recognized head-class name variants across datasets:
#   COD MW Warzone:  "head"
#   Kwan Li / Kw4L0r Valorant: "enemyHead"
#   David Hong Valorant: "enemy_head"
#   CS:GO / CS2 community YOLOv5 models: "ct_head", "t_head"
#   Splitgate (Matias Kovero): single-class "enemy" — not a head class
_HEAD_CLASS_KEYWORDS = {"head", "enemyhead", "enemy_head", "ct_head", "t_head"}


def _is_head_class(class_name: str) -> bool:
    """Return *True* if ``class_name`` looks like a head class.

    Matches exact known labels (case-insensitive) such as ``"head"``,
    ``"enemyHead"``, ``"enemy_head"``, ``"ct_head"``, ``"t_head"``.
    """
    return class_name.lower().strip() in _HEAD_CLASS_KEYWORDS


# =====================================================================
#  Sticky Aim (Magnetic Aim Assist) helpers
# =====================================================================

def calculate_sticky_pull(
    error_x: float,
    error_y: float,
    strength: float,
    fov_radius: float,
    box: List[float],
) -> Tuple[float, float]:
    """Compute a gentle magnetic pull vector toward the detected target.

    The pull is strongest when the crosshair is near the target's bounding
    box and tapers off as the distance increases.  It is designed to feel
    like subtle magnetism, not a hard snap-lock.

    Args:
        error_x:    Horizontal distance from crosshair to target centre (px).
        error_y:    Vertical distance from crosshair to target centre (px).
        strength:   User-configurable pull intensity in [0.0, 1.0].
        fov_radius: Half the FOV size (px).  Used as the reference distance
                    for the falloff curve — targets at the FOV edge produce
                    almost zero pull.
        box:        Detection bounding box [x1, y1, x2, y2].  The pull is
                    stronger when the crosshair is inside or close to the
                    box boundaries.

    Returns:
        ``(pull_x, pull_y)`` — the pixel delta to add to the movement
        output.  Both will be 0.0 when the pull is negligible.
    """
    if strength <= 0.0 or fov_radius <= 0.0:
        return 0.0, 0.0

    distance = math.sqrt(error_x * error_x + error_y * error_y)
    if distance < 0.5:
        # Already on top of the target — no pull needed.
        return 0.0, 0.0

    # ── Distance-based falloff ──
    # The pull uses a smooth inverse-square-ish curve:
    #   falloff = 1 / (1 + (distance / ref)²)
    # where *ref* is the bounding-box diagonal.  When the crosshair is
    # inside the box the falloff is close to 1; when it is far away the
    # falloff drops toward 0.
    box_w = max(box[2] - box[0], 1.0)
    box_h = max(box[3] - box[1], 1.0)
    box_diag = math.sqrt(box_w * box_w + box_h * box_h)

    # Reference distance: the larger of box diagonal or a sensible minimum.
    # This prevents the pull from being too aggressive on tiny boxes.
    ref_distance = max(box_diag, 30.0)

    # Smooth falloff: strongest near the box, fades as distance grows.
    ratio = distance / ref_distance
    falloff = 1.0 / (1.0 + ratio * ratio)

    # Additionally clamp to zero beyond the FOV radius so targets near the
    # edge of the detection area don't cause erratic pulls.
    if distance > fov_radius:
        falloff = 0.0
    elif distance > fov_radius * 0.7:
        # Gentle linear taper in the outer 30 % of the FOV.
        edge_factor = (fov_radius - distance) / (fov_radius * 0.3)
        falloff *= max(0.0, edge_factor)

    # ── Build the pull vector ──
    # Direction is toward the target; magnitude is scaled by strength
    # and falloff.  The maximum pull per frame is capped so it never
    # overshoots the target.
    #
    # "strength" maps to a fraction of the error that we apply each
    # frame.  With strength = 0.3 and perfect falloff the crosshair
    # drifts toward the target at ~30 % of the remaining distance per
    # frame — smooth and subtle.
    pull_magnitude = strength * falloff

    pull_x = error_x * pull_magnitude
    pull_y = error_y * pull_magnitude

    return pull_x, pull_y


def process_aiming(
    config: Config,
    boxes: List[List[float]],
    crosshair_x: int,
    crosshair_y: int,
    pid_x: PIDController,
    pid_y: PIDController,
    mouse_method: str,
    state: LoopState,
    current_time: float,
    class_ids: List[int] | None = None,
) -> None:
    """處理瞄準邏輯 (包含卡爾曼濾波預判和幽靈目標/貝塞爾曲線偏移)"""

    aim_part = config.aim_part
    head_height_ratio = config.head_height_ratio

    # Resolve per-detection class names when a multi-class model is active.
    model_class_names: List[str] = getattr(config, 'model_class_names', []) or []

    valid_targets = []
    for idx, box in enumerate(boxes):
        # Look up the human-readable class name for this detection.
        cls_id = class_ids[idx] if class_ids is not None and idx < len(class_ids) else 0
        cls_name = model_class_names[cls_id] if cls_id < len(model_class_names) else None

        target_x, target_y = calculate_aim_target(
            box, aim_part, head_height_ratio, class_name=cls_name,
        )
        moveX = target_x - crosshair_x
        moveY = target_y - crosshair_y
        distance_sq = moveX * moveX + moveY * moveY
        valid_targets.append((distance_sq, target_x, target_y, box))

    if valid_targets:
        valid_targets.sort(key=lambda x: x[0])
        _, target_x, target_y, box = valid_targets[0]

        tracker_enabled = getattr(config, 'tracker_enabled', False)
        if tracker_enabled:
            if state.smart_tracker is None:
                state.smart_tracker = SmartTracker(
                    smoothing_factor=getattr(config, 'tracker_smoothing_factor', 0.5),
                    stop_threshold=getattr(config, 'tracker_stop_threshold', 20.0),
                )
                state.tracker_last_time = current_time
            else:
                state.smart_tracker.alpha = getattr(config, 'tracker_smoothing_factor', 0.5)
                state.smart_tracker.stop_threshold = getattr(config, 'tracker_stop_threshold', 20.0)

            current_box_tuple = tuple(box)
            if state.tracker_last_target_box is not None:
                last_box = state.tracker_last_target_box
                last_cx = (last_box[0] + last_box[2]) * 0.5
                last_cy = (last_box[1] + last_box[3]) * 0.5
                curr_cx = (box[0] + box[2]) * 0.5
                curr_cy = (box[1] + box[3]) * 0.5
                box_distance_sq = (curr_cx - last_cx) ** 2 + (curr_cy - last_cy) ** 2
                if box_distance_sq > 40000:
                    state.smart_tracker.reset()
            state.tracker_last_target_box = current_box_tuple

            dt = current_time - state.tracker_last_time
            if dt <= 0:
                dt = 0.01
            state.tracker_last_time = current_time

            state.smart_tracker.update(target_x, target_y, dt)

            prediction_time = getattr(config, 'tracker_prediction_time', 0.05)
            pred_x, pred_y = state.smart_tracker.get_predicted_position(prediction_time)

            config.tracker_current_x = target_x
            config.tracker_current_y = target_y
            config.tracker_predicted_x = pred_x
            config.tracker_predicted_y = pred_y
            config.tracker_has_prediction = True

            target_x, target_y = pred_x, pred_y
        else:
            config.tracker_has_prediction = False
            if state.smart_tracker is not None:
                state.smart_tracker.reset()
                state.smart_tracker = None

        errorX = target_x - crosshair_x
        errorY = target_y - crosshair_y

        if getattr(config, 'bezier_curve_enabled', False):
            if not state.target_locked:
                state.target_locked = True
                state.bezier_curve_scalar = random.uniform(-1.0, 1.0)

            strength = float(getattr(config, 'bezier_curve_strength', 0.35))
            perp_x = -errorY
            perp_y = errorX

            offset_x = perp_x * strength * state.bezier_curve_scalar
            offset_y = perp_y * strength * state.bezier_curve_scalar

            errorX += offset_x
            errorY += offset_y
        else:
            state.target_locked = False

        dx, dy = pid_x.update(errorX), pid_y.update(errorY)

        if getattr(config, 'aim_y_reduce_enabled', False) and state.aiming_start_time > 0:
            aim_duration = current_time - state.aiming_start_time
            delay = getattr(config, 'aim_y_reduce_delay', 0.6)

            if aim_duration > delay:
                dy = 0.0

        # ── Sticky Aim (Magnetic Aim Assist) ──
        # When enabled, we blend the PID-computed movement with a gentle
        # magnetic pull toward the target.  For the Xbox method we also
        # read the user's physical right stick and add the pull on top,
        # so the user retains full manual control with a subtle bias.
        sticky_enabled = getattr(config, 'sticky_aim_enabled', False)
        sticky_strength = float(getattr(config, 'sticky_aim_strength', 0.3))

        if sticky_enabled and sticky_strength > 0.0:
            fov_radius = float(config.fov_size) * 0.5
            pull_x, pull_y = calculate_sticky_pull(
                errorX, errorY, sticky_strength, fov_radius, box,
            )

            if mouse_method == 'xbox':
                # Xbox path: read the user's physical right stick and
                # add the magnetic pull on top.  This way the user keeps
                # full manual camera control and just feels a gentle
                # bias toward the target.
                from win_utils.gamepad_input import get_right_stick
                phys_rx, phys_ry = get_right_stick()

                # Convert physical stick [-1, 1] to a pixel-ish scale
                # that matches the dx/dy coordinate system used by
                # move_right_stick (which divides by BASE_PIXELS=50).
                _STICK_TO_PX = 50.0
                user_dx = phys_rx * _STICK_TO_PX
                user_dy = phys_ry * _STICK_TO_PX

                # Blend: user input + magnetic pull.  The PID output is
                # NOT added here — the pull replaces the hard PID snap
                # with a softer bias.  If you want both PID *and* sticky
                # pull, add `dx` and `dy` as well.
                move_x = int(round(user_dx + pull_x))
                move_y = int(round(user_dy + pull_y))
            else:
                # Mouse path: the user's mouse movement happens through
                # the OS natively.  We just inject the pull as a small
                # additional cursor movement.
                move_x = int(round(pull_x))
                move_y = int(round(pull_y))
        else:
            move_x, move_y = int(round(dx)), int(round(dy))

        if move_x != 0 or move_y != 0:
            send_mouse_move(move_x, move_y, method=mouse_method)
    else:
        state.target_locked = False
        pid_x.reset()
        pid_y.reset()
