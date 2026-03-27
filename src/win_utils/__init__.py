# win_utils/__init__.py - Windows Toolkit
"""
Windows Toolkit - Providing mouse control, key detection, administrator privileges, terminal control, etc.

Module Structure:
- vk_codes: Virtual key codes and translation
- mouse_move: Basic mouse movement functions

- ddxoft_mouse: DDXoft mouse control
- mouse_click: Mouse click functions
- key_utils: Key detection
- admin: Administrator privilege management
- console: Terminal window control
"""

# Virtual key codes
from .vk_codes import (
    VK_CODE_MAP,
    VK_TRANSLATIONS,
    get_vk_name,
)

# Mouse move - Basic
from .mouse_move import (
    MOUSEINPUT,
    INPUT,
    INPUT_MOUSE,
    MOUSEEVENTF_MOVE,
    send_mouse_move_sendinput,
    send_mouse_move_mouse_event,
)



# Mouse move - ddxoft
from .ddxoft_mouse import (
    DDXoftMouse,
    ddxoft_mouse,
    send_mouse_move_ddxoft,
    ensure_ddxoft_ready,
    test_ddxoft_functions,
    get_ddxoft_statistics,
    print_ddxoft_statistics,
    reset_ddxoft_statistics,
)

# Mouse move - Arduino Leonardo
from .arduino_mouse import (
    ArduinoMouse,
    arduino_mouse,
    send_mouse_move_arduino,
    get_available_com_ports,
    connect_arduino,
    disconnect_arduino,
    is_arduino_connected,
)

# Mouse move - MAKCU KM Host
from .makcu_mouse import (
    MakcuMouse,
    makcu_mouse,
    send_mouse_move_makcu,
    send_mouse_click_makcu,
    connect_makcu,
    disconnect_makcu,
    is_makcu_connected,
)

# Mouse move - Xbox 360 Virtual Gamepad
from .xbox_controller import (
    XboxController,
    xbox_controller,
    send_mouse_move_xbox,
    send_mouse_click_xbox,
    send_movement_xbox,
    connect_xbox,
    disconnect_xbox,
    is_xbox_connected,
    is_xbox_available,
    set_xbox_sensitivity,
    set_xbox_deadzone,
    get_xbox_statistics,
    diagnose_xbox,
    detect_anti_cheat,
    check_double_input_conflict,
)

# Physical Xbox / XInput controller input handler
from .xinput_handler import (
    XboxInputState,
    XboxInputHandler,
    xbox_input_handler,
)

# Analog stick reading with standard XInput deadzones
from .gamepad_input import (
    apply_radial_deadzone,
    get_left_stick,
    get_right_stick,
    XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
    XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE,
    XINPUT_GAMEPAD_TRIGGER_THRESHOLD,
)

# Mouse click
from .mouse_click import (
    send_mouse_click_sendinput,
    send_mouse_click_hardware,
    send_mouse_click_mouse_event,
    send_mouse_click_ddxoft,
    send_mouse_click,
    test_mouse_click_methods,
)
from .arduino_mouse import send_mouse_click_arduino

# Key detection
from .key_utils import is_key_pressed

# Gamepad button reading
from .gamepad_input import (
    is_gamepad_vk,
    is_gamepad_button_pressed,
    poll_pressed_gamepad_button,
    GP_VK_TRANSLATION_MAP,
    GP_VK_MIN,
    GP_VK_MAX,
)

# Administrator privileges
from .admin import (
    is_admin,
    request_admin_privileges,
    check_and_request_admin,
)

# 終端控制
from .console import (
    get_console_window,
    show_console,
    hide_console,
    is_console_visible,
)


# ===== 主要滑鼠移動函數 =====

def send_mouse_move(dx, dy, method="xbox"):
    """
    主要滑鼠移動函數 (Primary mouse / right-stick movement dispatcher)

    Default method is 'xbox' because this build targets virtual gamepad output.
    The AI aiming loop passes the cached config.mouse_move_method here, so the
    default is only used when the caller does not specify a method explicitly.

    method 選項:
    - "xbox":        Xbox 360 虛擬手把右搖桿 (primary — ViGEmBus)
    - "mouse_event": mouse_event (legacy fallback, detectable)
    - "sendinput":   SendInput (legacy fallback, detectable)
    - "ddxoft":      ddxoft (requires ddxoft.dll)
    - "arduino":     Arduino Leonardo (USB HID)
    - "makcu":       MAKCU KM Host (hardware USB HID)
    """
    if method == "xbox":
        # For the xbox path the pixel-minimum guard does not apply —
        # send_mouse_move_xbox converts dx/dy to a normalised axis value
        # and small moves are perfectly valid for fine aim correction.
        send_mouse_move_xbox(dx, dy)
    elif abs(dx) < 1 and abs(dy) < 1:
        # For all legacy pixel-based methods, skip sub-pixel moves.
        return
    elif method == "sendinput":
        send_mouse_move_sendinput(dx, dy)
    elif method == "mouse_event":
        send_mouse_move_mouse_event(dx, dy)
    elif method == "ddxoft":
        send_mouse_move_ddxoft(dx, dy)
    elif method == "arduino":
        send_mouse_move_arduino(dx, dy)
    elif method == "makcu":
        send_mouse_move_makcu(dx, dy)
    # Unknown method → silent no-op; do NOT fall back to mouse_event to avoid
    # unintended raw mouse injection when the user expects gamepad output.


# 公開的 API 列表
__all__ = [
    # 虛擬按鍵碼
    'VK_CODE_MAP',
    'VK_TRANSLATIONS',
    'get_vk_name',
    
    # 滑鼠移動
    'MOUSEINPUT',
    'INPUT',
    'INPUT_MOUSE',
    'MOUSEEVENTF_MOVE',
    'send_mouse_move',
    'send_mouse_move_sendinput',
    'send_mouse_move_mouse_event',
    'send_mouse_move_ddxoft',
    'send_mouse_move_arduino',
    'send_mouse_move_makcu',
    'send_mouse_move_xbox',
    
    # 控制器類
    'DDXoftMouse',
    'XboxController',
    'xbox_controller',
    'ddxoft_mouse',
    
    # ddxoft 公共接口
    'ensure_ddxoft_ready',
    'test_ddxoft_functions',
    'get_ddxoft_statistics',
    'print_ddxoft_statistics',
    'reset_ddxoft_statistics',
    
    # Arduino 控制
    'ArduinoMouse',
    'arduino_mouse',
    'get_available_com_ports',
    'connect_arduino',
    'disconnect_arduino',
    'is_arduino_connected',
    
    # MAKCU KM Host 控制
    'MakcuMouse',
    'makcu_mouse',
    'connect_makcu',
    'disconnect_makcu',
    'is_makcu_connected',
    'send_mouse_click_makcu',
    
    # Xbox 360 虛擬手把
    'connect_xbox',
    'disconnect_xbox',
    'is_xbox_connected',
    'is_xbox_available',
    'set_xbox_sensitivity',
    'set_xbox_deadzone',
    'get_xbox_statistics',
    'send_mouse_click_xbox',
    'send_movement_xbox',
    'diagnose_xbox',
    'detect_anti_cheat',
    'check_double_input_conflict',

    # 實體 XInput 控制器輸入處理
    'XboxInputState',
    'XboxInputHandler',
    'xbox_input_handler',

    # 類比搖桿讀取（含標準 XInput 死區）
    'apply_radial_deadzone',
    'get_left_stick',
    'get_right_stick',
    'XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE',
    'XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE',
    'XINPUT_GAMEPAD_TRIGGER_THRESHOLD',
    
    # 滑鼠點擊
    'send_mouse_click',
    'send_mouse_click_sendinput',
    'send_mouse_click_hardware',
    'send_mouse_click_mouse_event',
    'send_mouse_click_ddxoft',
    'send_mouse_click_arduino',
    'send_mouse_click_makcu',
    'test_mouse_click_methods',
    
    # 按鍵檢測
    'is_key_pressed',
    
    # 手柄按鍵
    'is_gamepad_vk',
    'is_gamepad_button_pressed',
    'poll_pressed_gamepad_button',
    'GP_VK_TRANSLATION_MAP',
    'GP_VK_MIN',
    'GP_VK_MAX',
    
    # 管理員權限
    'is_admin',
    'request_admin_privileges',
    'check_and_request_admin',
    
    # 終端控制
    'get_console_window',
    'show_console',
    'hide_console',
    'is_console_visible',
]

