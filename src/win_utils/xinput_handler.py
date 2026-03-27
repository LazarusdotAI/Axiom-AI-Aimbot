# xinput_handler.py - Physical Xbox Controller Input Mapping
"""
High-level XInput physical controller handler.

Reads a real (physical) Xbox / XInput controller and maps its inputs to
standardised game and UI actions:

  Left  Stick  → movement axes  (strafe / walk)
  Right Stick  → camera / look  axes (yaw / pitch)
  A  button    → primary action  (Jump) / UI confirm
  B  button    → secondary action / UI cancel
  D-pad        → UI navigation   (Up / Down / Left / Right)

Deadzone processing uses the XInput radial deadzone method so sticks feel
natural and do not drift when at rest.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from .gamepad_input import (
    get_gamepad_state,
    apply_radial_deadzone,
    get_left_stick,
    get_right_stick,
    XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
    XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE,
    XINPUT_GAMEPAD_TRIGGER_THRESHOLD,
    XINPUT_GAMEPAD_A,
    XINPUT_GAMEPAD_B,
    XINPUT_GAMEPAD_X,
    XINPUT_GAMEPAD_Y,
    XINPUT_GAMEPAD_LEFT_SHOULDER,
    XINPUT_GAMEPAD_RIGHT_SHOULDER,
    XINPUT_GAMEPAD_START,
    XINPUT_GAMEPAD_BACK,
    XINPUT_GAMEPAD_DPAD_UP,
    XINPUT_GAMEPAD_DPAD_DOWN,
    XINPUT_GAMEPAD_DPAD_LEFT,
    XINPUT_GAMEPAD_DPAD_RIGHT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input state dataclass
# ---------------------------------------------------------------------------

@dataclass
class XboxInputState:
    """Snapshot of a single controller poll.

    Analog axes are normalized to [-1.0, 1.0] after deadzone processing.
    Edge-detected ``*_pressed`` fields are ``True`` only on the frame the
    button transitions from released to pressed.
    """

    # --- Analog axes (deadzone applied, normalized) -------------------------
    # Left Stick  → movement
    left_stick_x: float = 0.0   # +right  / -left
    left_stick_y: float = 0.0   # +up(fwd)/ -down(back)
    # Right Stick → camera / look
    right_stick_x: float = 0.0  # +look-right / -look-left
    right_stick_y: float = 0.0  # +look-up    / -look-down

    # --- Triggers (0.0 – 1.0) ----------------------------------------------
    left_trigger: float = 0.0
    right_trigger: float = 0.0

    # --- Digital buttons – held state --------------------------------------
    button_a: bool = False       # Primary action / Jump / UI confirm
    button_b: bool = False       # Cancel / Back
    button_x: bool = False
    button_y: bool = False
    button_lb: bool = False
    button_rb: bool = False
    button_start: bool = False
    button_back: bool = False
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False

    # --- Edge-detected press events (rising edge only) ---------------------
    a_pressed: bool = False      # A just pressed this poll
    b_pressed: bool = False      # B just pressed this poll
    x_pressed: bool = False
    y_pressed: bool = False
    start_pressed: bool = False
    back_pressed: bool = False
    dpad_up_pressed: bool = False
    dpad_down_pressed: bool = False
    dpad_left_pressed: bool = False
    dpad_right_pressed: bool = False

    # --- Connection flag ----------------------------------------------------
    connected: bool = False

    # --- Convenience aliases ------------------------------------------------
    @property
    def movement_x(self) -> float:
        """Left-stick X mapped to horizontal movement."""
        return self.left_stick_x

    @property
    def movement_y(self) -> float:
        """Left-stick Y mapped to forward/back movement."""
        return self.left_stick_y

    @property
    def look_x(self) -> float:
        """Right-stick X mapped to camera horizontal (yaw)."""
        return self.right_stick_x

    @property
    def look_y(self) -> float:
        """Right-stick Y mapped to camera vertical (pitch)."""
        return self.right_stick_y

    @property
    def jump(self) -> bool:
        """A button pressed — primary action (Jump)."""
        return self.a_pressed

    @property
    def ui_select(self) -> bool:
        """A button pressed — UI selection / confirm."""
        return self.a_pressed

    @property
    def ui_cancel(self) -> bool:
        """B button pressed — UI cancel / back."""
        return self.b_pressed

    @property
    def ui_navigate_up(self) -> bool:
        return self.dpad_up_pressed

    @property
    def ui_navigate_down(self) -> bool:
        return self.dpad_down_pressed

    @property
    def ui_navigate_left(self) -> bool:
        return self.dpad_left_pressed

    @property
    def ui_navigate_right(self) -> bool:
        return self.dpad_right_pressed


# ---------------------------------------------------------------------------
# XboxInputHandler
# ---------------------------------------------------------------------------

class XboxInputHandler:
    """Polls a physical XInput controller and maps axes/buttons to game actions.

    Usage example::

        handler = XboxInputHandler()
        while running:
            state = handler.poll()
            if state.connected:
                move_player(state.movement_x, state.movement_y)
                rotate_camera(state.look_x, state.look_y)
                if state.jump:
                    player.jump()
                if state.ui_navigate_down:
                    menu.select_next()
                if state.ui_select:
                    menu.confirm()
                if state.ui_cancel:
                    menu.back()

    Parameters
    ----------
    user_index:
        XInput player slot (0–3).  Slot 0 is the first connected controller.
    left_deadzone:
        Radial deadzone for the left stick (movement).  Defaults to the XInput
        standard value of 7849.
    right_deadzone:
        Radial deadzone for the right stick (camera).  Defaults to the XInput
        standard value of 8689.
    trigger_threshold:
        Trigger activation threshold (0–255).  Defaults to 30.
    """

    def __init__(
        self,
        user_index: int = 0,
        left_deadzone: int = XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
        right_deadzone: int = XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE,
        trigger_threshold: int = XINPUT_GAMEPAD_TRIGGER_THRESHOLD,
    ) -> None:
        self.user_index = user_index
        self.left_deadzone = left_deadzone
        self.right_deadzone = right_deadzone
        self.trigger_threshold = trigger_threshold

        # Previous button word for edge detection
        self._prev_buttons: int = 0
        self._prev_lt: int = 0
        self._prev_rt: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll(self) -> XboxInputState:
        """Query the controller and return an :class:`XboxInputState`.

        Call this once per game / UI update loop frame.  The returned state
        contains both held-state flags and rising-edge ``*_pressed`` events so
        callers do not need to track previous state themselves.

        Returns:
            :class:`XboxInputState` with all fields populated.  If no
            controller is connected, ``state.connected`` is ``False`` and all
            axes/buttons are zero / ``False``.
        """
        raw = get_gamepad_state(self.user_index)
        if raw is None:
            self._prev_buttons = 0
            return XboxInputState(connected=False)

        gp = raw.Gamepad
        buttons = gp.wButtons

        # --- Analog sticks with radial deadzone ----------------------------
        lx, ly = apply_radial_deadzone(gp.sThumbLX, gp.sThumbLY,
                                        self.left_deadzone)
        rx, ry = apply_radial_deadzone(gp.sThumbRX, gp.sThumbRY,
                                        self.right_deadzone)

        # --- Triggers (normalize to 0.0 – 1.0) ----------------------------
        lt = gp.bLeftTrigger / 255.0
        rt = gp.bRightTrigger / 255.0

        # --- Rising-edge detection (current & NOT previous) ----------------
        rising = buttons & ~self._prev_buttons

        state = XboxInputState(
            connected=True,

            # Movement – left stick
            left_stick_x=lx,
            left_stick_y=ly,   # +Y = forward (up in XInput)

            # Camera – right stick
            right_stick_x=rx,
            right_stick_y=ry,

            # Triggers
            left_trigger=lt,
            right_trigger=rt,

            # Held digital buttons
            button_a=(buttons & XINPUT_GAMEPAD_A) != 0,
            button_b=(buttons & XINPUT_GAMEPAD_B) != 0,
            button_x=(buttons & XINPUT_GAMEPAD_X) != 0,
            button_y=(buttons & XINPUT_GAMEPAD_Y) != 0,
            button_lb=(buttons & XINPUT_GAMEPAD_LEFT_SHOULDER) != 0,
            button_rb=(buttons & XINPUT_GAMEPAD_RIGHT_SHOULDER) != 0,
            button_start=(buttons & XINPUT_GAMEPAD_START) != 0,
            button_back=(buttons & XINPUT_GAMEPAD_BACK) != 0,
            dpad_up=(buttons & XINPUT_GAMEPAD_DPAD_UP) != 0,
            dpad_down=(buttons & XINPUT_GAMEPAD_DPAD_DOWN) != 0,
            dpad_left=(buttons & XINPUT_GAMEPAD_DPAD_LEFT) != 0,
            dpad_right=(buttons & XINPUT_GAMEPAD_DPAD_RIGHT) != 0,

            # Edge-detected press events (rising edge only)
            a_pressed=(rising & XINPUT_GAMEPAD_A) != 0,
            b_pressed=(rising & XINPUT_GAMEPAD_B) != 0,
            x_pressed=(rising & XINPUT_GAMEPAD_X) != 0,
            y_pressed=(rising & XINPUT_GAMEPAD_Y) != 0,
            start_pressed=(rising & XINPUT_GAMEPAD_START) != 0,
            back_pressed=(rising & XINPUT_GAMEPAD_BACK) != 0,
            dpad_up_pressed=(rising & XINPUT_GAMEPAD_DPAD_UP) != 0,
            dpad_down_pressed=(rising & XINPUT_GAMEPAD_DPAD_DOWN) != 0,
            dpad_left_pressed=(rising & XINPUT_GAMEPAD_DPAD_LEFT) != 0,
            dpad_right_pressed=(rising & XINPUT_GAMEPAD_DPAD_RIGHT) != 0,
        )

        # Save current state for next frame's edge detection
        self._prev_buttons = buttons
        self._prev_lt = gp.bLeftTrigger
        self._prev_rt = gp.bRightTrigger

        return state

    def reconfigure(
        self,
        left_deadzone: Optional[int] = None,
        right_deadzone: Optional[int] = None,
        trigger_threshold: Optional[int] = None,
    ) -> None:
        """Update deadzone / threshold settings at runtime.

        Only the keyword arguments that are not ``None`` are changed.
        """
        if left_deadzone is not None:
            self.left_deadzone = int(left_deadzone)
        if right_deadzone is not None:
            self.right_deadzone = int(right_deadzone)
        if trigger_threshold is not None:
            self.trigger_threshold = int(trigger_threshold)


# ---------------------------------------------------------------------------
# Module-level singleton (controller slot 0)
# ---------------------------------------------------------------------------

#: Ready-to-use singleton for the first connected controller.
xbox_input_handler: XboxInputHandler = XboxInputHandler()

