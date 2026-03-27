# gamepad_input.py - Gamepad Input Reading Module
"""
Use XInput API to read physical gamepad button states
For key bindings and runtime key detection
"""

import ctypes
import ctypes.wintypes
import time
from typing import Optional, Dict

# ===== XInput 結構定義 =====

class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.wintypes.WORD),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.wintypes.DWORD),
        ("Gamepad", XINPUT_GAMEPAD),
    ]

# ===== XInput 按鈕常數 =====
XINPUT_GAMEPAD_DPAD_UP        = 0x0001
XINPUT_GAMEPAD_DPAD_DOWN      = 0x0002
XINPUT_GAMEPAD_DPAD_LEFT      = 0x0004
XINPUT_GAMEPAD_DPAD_RIGHT     = 0x0008
XINPUT_GAMEPAD_START          = 0x0010
XINPUT_GAMEPAD_BACK           = 0x0020
XINPUT_GAMEPAD_LEFT_THUMB     = 0x0040
XINPUT_GAMEPAD_RIGHT_THUMB    = 0x0080
XINPUT_GAMEPAD_LEFT_SHOULDER  = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
XINPUT_GAMEPAD_A              = 0x1000
XINPUT_GAMEPAD_B              = 0x2000
XINPUT_GAMEPAD_X              = 0x4000
XINPUT_GAMEPAD_Y              = 0x8000

# Trigger threshold (0-255, values above this are considered pressed)
TRIGGER_THRESHOLD = 100

# ===== XInput Standard Deadzone Constants (per XInput specification) =====
# These match the values from xinput.h in the Windows SDK.
XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE  = 7849   # Standard left-stick deadzone
XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE = 8689   # Standard right-stick deadzone
XINPUT_GAMEPAD_TRIGGER_THRESHOLD    = 30     # Standard trigger activation threshold

# Maximum raw thumb value (signed 16-bit, exclusive of -32768 per XInput spec)
_THUMB_MAX = 32767.0

# ===== Custom Virtual Key Codes (0x300+ range, avoid conflict with Windows VK codes) =====
# Buttons
GP_VK_A              = 0x0301
GP_VK_B              = 0x0302
GP_VK_X              = 0x0303
GP_VK_Y              = 0x0304
GP_VK_LB             = 0x0305  # Left Shoulder / LB
GP_VK_RB             = 0x0306  # Right Shoulder / RB
GP_VK_LT             = 0x0307  # Left Trigger
GP_VK_RT             = 0x0308  # Right Trigger
GP_VK_BACK           = 0x0309
GP_VK_START          = 0x030A
GP_VK_LSTICK         = 0x030B  # Left Stick Click
GP_VK_RSTICK         = 0x030C  # Right Stick Click
GP_VK_DPAD_UP        = 0x030D
GP_VK_DPAD_DOWN      = 0x030E
GP_VK_DPAD_LEFT      = 0x030F
GP_VK_DPAD_RIGHT     = 0x0310

# Range check
GP_VK_MIN = 0x0301
GP_VK_MAX = 0x0310

def is_gamepad_vk(vk_code: int) -> bool:
    """Check if the virtual key code is a gamepad button"""
    return GP_VK_MIN <= vk_code <= GP_VK_MAX

# Mapping of XInput button flags to custom VK codes
_BUTTON_FLAG_TO_GP_VK = {
    XINPUT_GAMEPAD_A:              GP_VK_A,
    XINPUT_GAMEPAD_B:              GP_VK_B,
    XINPUT_GAMEPAD_X:              GP_VK_X,
    XINPUT_GAMEPAD_Y:              GP_VK_Y,
    XINPUT_GAMEPAD_LEFT_SHOULDER:  GP_VK_LB,
    XINPUT_GAMEPAD_RIGHT_SHOULDER: GP_VK_RB,
    XINPUT_GAMEPAD_BACK:           GP_VK_BACK,
    XINPUT_GAMEPAD_START:          GP_VK_START,
    XINPUT_GAMEPAD_LEFT_THUMB:     GP_VK_LSTICK,
    XINPUT_GAMEPAD_RIGHT_THUMB:    GP_VK_RSTICK,
    XINPUT_GAMEPAD_DPAD_UP:        GP_VK_DPAD_UP,
    XINPUT_GAMEPAD_DPAD_DOWN:      GP_VK_DPAD_DOWN,
    XINPUT_GAMEPAD_DPAD_LEFT:      GP_VK_DPAD_LEFT,
    XINPUT_GAMEPAD_DPAD_RIGHT:     GP_VK_DPAD_RIGHT,
}

# Reverse mapping of custom VK codes to XInput button flags
_GP_VK_TO_BUTTON_FLAG: Dict[int, int] = {v: k for k, v in _BUTTON_FLAG_TO_GP_VK.items()}

# Mapping of custom VK codes to translation keys
GP_VK_TRANSLATION_MAP = {
    GP_VK_A:          "key_gp_a",
    GP_VK_B:          "key_gp_b",
    GP_VK_X:          "key_gp_x",
    GP_VK_Y:          "key_gp_y",
    GP_VK_LB:         "key_gp_lb",
    GP_VK_RB:         "key_gp_rb",
    GP_VK_LT:         "key_gp_lt",
    GP_VK_RT:         "key_gp_rt",
    GP_VK_BACK:       "key_gp_back",
    GP_VK_START:      "key_gp_start",
    GP_VK_LSTICK:     "key_gp_lstick",
    GP_VK_RSTICK:     "key_gp_rstick",
    GP_VK_DPAD_UP:    "key_gp_dpad_up",
    GP_VK_DPAD_DOWN:  "key_gp_dpad_down",
    GP_VK_DPAD_LEFT:  "key_gp_dpad_left",
    GP_VK_DPAD_RIGHT: "key_gp_dpad_right",
}

# ===== XInput DLL 載入 =====

_xinput = None
_xinput_loaded = False

# ===== Active-slot cache =====
# XInput supports slots 0-3. Most systems have the first physical controller
# appear on slot 0, but some configurations (e.g. after ViGEmBus creates a
# virtual device) push the real controller to slot 1 or higher.
# We track the last-known connected slot and rescan periodically so the rest
# of the code never has to think about slot numbers.
_active_slot: int = 0           # last slot that returned ERROR_SUCCESS
_slot_last_scanned: float = 0.0 # monotonic timestamp of the last full scan
_SLOT_RESCAN_INTERVAL: float = 3.0  # seconds between full scans

# ===== Virtual-controller slot exclusion =====
# When a ViGEmBus virtual controller is created it occupies an XInput slot
# that would otherwise be indistinguishable from a physical controller.  If
# get_gamepad_state() reads from that slot, it sees the *virtual* device's
# state (last values written by the app itself) instead of the physical
# controller — which breaks the left-stick passthrough completely.
#
# set_virtual_slot() / clear_virtual_slot() let xbox_controller.py tell us
# which slot to skip.  snapshot_connected_slots() lets it diff before/after
# virtual-device creation to figure out the slot number.
_virtual_slot: Optional[int] = None


def set_virtual_slot(slot: int) -> None:
    """Mark an XInput slot as belonging to the virtual controller.

    All subsequent calls to ``get_gamepad_state`` and
    ``_scan_for_connected_slot`` will skip this slot so that only the
    physical controller is ever read.
    """
    global _virtual_slot, _active_slot, _slot_last_scanned
    _virtual_slot = slot
    # Invalidate the cached active slot – force a rescan on the next read
    # so we don't keep returning stale data from the virtual device.
    if _active_slot == slot:
        _active_slot = -1
    _slot_last_scanned = 0.0


def clear_virtual_slot() -> None:
    """Remove the virtual-slot exclusion (e.g. after disconnecting)."""
    global _virtual_slot
    _virtual_slot = None


def snapshot_connected_slots() -> set:
    """Return the set of XInput slot indices that currently have a controller.

    Used by ``xbox_controller.py`` to diff before / after virtual-device
    creation and determine which slot the new device occupied.
    """
    if not _load_xinput():
        return set()
    result = set()
    for idx in range(4):
        test = XINPUT_STATE()
        if _xinput.XInputGetState(idx, ctypes.byref(test)) == 0:
            result.add(idx)
    return result

def _load_xinput():
    """嘗試載入 XInput DLL"""
    global _xinput, _xinput_loaded
    if _xinput_loaded:
        return _xinput is not None

    _xinput_loaded = True
    # 依序嘗試不同版本的 XInput
    for dll_name in ["xinput1_4", "xinput1_3", "xinput9_1_0"]:
        try:
            _xinput = ctypes.windll.LoadLibrary(dll_name + ".dll")
            return True
        except OSError:
            continue

    print("[Gamepad] 找不到 XInput DLL，手柄按鍵綁定不可用")
    _xinput = None
    return False


def _scan_for_connected_slot() -> int:
    """Scan all four XInput slots and return the first *physical* controller.

    Slots marked as belonging to the ViGEmBus virtual controller
    (``_virtual_slot``) are skipped so we never accidentally read back
    our own output.

    Returns slot 0 as a fallback when nothing is connected.
    This is called at most once every ``_SLOT_RESCAN_INTERVAL`` seconds.
    """
    if _xinput is None:
        return 0
    for idx in range(4):
        if idx == _virtual_slot:
            continue  # skip the virtual controller
        test = XINPUT_STATE()
        if _xinput.XInputGetState(idx, ctypes.byref(test)) == 0:
            return idx
    return 0  # nothing connected; default to 0


def get_gamepad_state(user_index: int = 0) -> Optional[XINPUT_STATE]:
    """Return the XInput state for the first connected *physical* controller.

    The ``user_index`` parameter is honoured as a *hint* (slot 0 is the
    default for every caller), but if that slot returns an error **or is the
    virtual-controller slot**, the function automatically checks the cached
    active slot and then does a full four-slot rescan (rate-limited to every
    ``_SLOT_RESCAN_INTERVAL`` seconds).

    Slots marked via ``set_virtual_slot()`` are never read, preventing the
    feedback loop where the passthrough would read back its own output from
    the ViGEmBus virtual device instead of the real physical controller.

    Args:
        user_index: Preferred XInput player index (0-3).

    Returns:
        XINPUT_STATE on success, or None if no controller is connected.
    """
    global _active_slot, _slot_last_scanned

    if not _load_xinput():
        return None

    state = XINPUT_STATE()

    # 1. Fast path: try the requested slot — but SKIP the virtual device.
    if user_index != _virtual_slot:
        result = _xinput.XInputGetState(user_index, ctypes.byref(state))
        if result == 0:
            _active_slot = user_index  # keep cache fresh
            return state

    # 2. Try the last known good slot (may differ from user_index).
    if _active_slot != user_index and _active_slot != _virtual_slot:
        result2 = _xinput.XInputGetState(_active_slot, ctypes.byref(state))
        if result2 == 0:
            return state

    # 3. Full rescan — but no more often than _SLOT_RESCAN_INTERVAL seconds.
    now = time.monotonic()
    if now - _slot_last_scanned >= _SLOT_RESCAN_INTERVAL:
        _slot_last_scanned = now
        found = _scan_for_connected_slot()
        _active_slot = found
        # Always try the found slot as long as it isn't the virtual device.
        # Previous code skipped the read when ``found == user_index``, but
        # that caused a silent None return when the fast-path (step 1) had
        # been skipped because ``user_index`` WAS the virtual slot and the
        # rescan happened to return the same numeric index for a *different*
        # reason (e.g. after a replug).
        if found != _virtual_slot and found >= 0:
            result3 = _xinput.XInputGetState(found, ctypes.byref(state))
            if result3 == 0:
                return state

    return None


def poll_pressed_gamepad_button(user_index: int = 0) -> int:
    """輪詢手柄，回傳目前按下的第一個按鈕的自訂 VK 碼
    
    用於按鍵綁定時偵測手柄按鈕。
    
    Returns:
        自訂 VK 碼 (0x0301-0x0310)，若無按下則回傳 0
    """
    state = get_gamepad_state(user_index)
    if state is None:
        return 0
    
    gp = state.Gamepad
    
    # 檢查數位按鈕
    for flag, gp_vk in _BUTTON_FLAG_TO_GP_VK.items():
        if gp.wButtons & flag:
            return gp_vk
    
    # 檢查扳機（類比 → 數位）
    if gp.bLeftTrigger > TRIGGER_THRESHOLD:
        return GP_VK_LT
    if gp.bRightTrigger > TRIGGER_THRESHOLD:
        return GP_VK_RT
    
    return 0


def apply_radial_deadzone(
    raw_x: int,
    raw_y: int,
    deadzone: int,
) -> tuple:
    """Apply a circular (radial) deadzone to raw XInput thumb-stick values.

    The radial deadzone method checks the magnitude of the 2-D stick vector
    rather than each axis independently.  This is the approach recommended by
    Microsoft's XInput documentation and avoids the cross-shaped dead region
    produced by per-axis clamping.

    Args:
        raw_x:    Raw sThumbLX / sThumbRX value (-32768 to 32767).
        raw_y:    Raw sThumbLY / sThumbRY value (-32768 to 32767).
        deadzone: Circular deadzone radius (e.g. 7849 or 8689).

    Returns:
        Tuple (norm_x, norm_y) in [-1.0, 1.0] with the deadzone applied.
        Both components are 0.0 when the stick is inside the dead zone.
    """
    import math
    magnitude = math.sqrt(raw_x * raw_x + raw_y * raw_y)
    if magnitude < deadzone:
        return (0.0, 0.0)

    # Scale so the value 0.0 maps to the edge of the deadzone (no jump)
    # and 1.0 maps to the maximum physical throw.
    scale = (magnitude - deadzone) / (_THUMB_MAX - deadzone)
    scale = min(1.0, scale)  # clamp

    norm_x = (raw_x / magnitude) * scale
    norm_y = (raw_y / magnitude) * scale
    return (norm_x, norm_y)


def get_left_stick(user_index: int = 0) -> tuple:
    """Return the normalized left-stick axes with the standard XInput deadzone.

    The left stick maps to **movement** (strafe left/right, move forward/back).

    Args:
        user_index: Controller slot (0-3).

    Returns:
        ``(x, y)`` floats in [-1.0, 1.0].  Both are 0.0 when no controller is
        connected or the stick is inside the dead zone.
        - x > 0 → right,  x < 0 → left
        - y > 0 → up (forward in most games),  y < 0 → down (backward)
    """
    state = get_gamepad_state(user_index)
    if state is None:
        return (0.0, 0.0)
    gp = state.Gamepad
    return apply_radial_deadzone(gp.sThumbLX, gp.sThumbLY,
                                  XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE)


def get_right_stick(user_index: int = 0) -> tuple:
    """Return the normalized right-stick axes with the standard XInput deadzone.

    The right stick maps to **camera / look** (yaw left/right, pitch up/down).

    Args:
        user_index: Controller slot (0-3).

    Returns:
        ``(x, y)`` floats in [-1.0, 1.0].  Both are 0.0 when no controller is
        connected or the stick is inside the dead zone.
        - x > 0 → look right,  x < 0 → look left
        - y > 0 → look up,     y < 0 → look down
    """
    state = get_gamepad_state(user_index)
    if state is None:
        return (0.0, 0.0)
    gp = state.Gamepad
    return apply_radial_deadzone(gp.sThumbRX, gp.sThumbRY,
                                  XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE)


def is_gamepad_button_pressed(gp_vk: int, user_index: int = 0) -> bool:
    """檢查指定的手柄按鈕是否被按下
    
    Args:
        gp_vk: 自訂手柄 VK 碼 (0x0301-0x0310)
        user_index: 玩家索引 (0-3)
        
    Returns:
        是否被按下
    """
    if not is_gamepad_vk(gp_vk):
        return False
    
    state = get_gamepad_state(user_index)
    if state is None:
        return False
    
    gp = state.Gamepad
    
    # 扳機特殊處理
    if gp_vk == GP_VK_LT:
        return gp.bLeftTrigger > TRIGGER_THRESHOLD
    if gp_vk == GP_VK_RT:
        return gp.bRightTrigger > TRIGGER_THRESHOLD
    
    # 數位按鈕
    flag = _GP_VK_TO_BUTTON_FLAG.get(gp_vk, 0)
    if flag:
        return bool(gp.wButtons & flag)
    
    return False
