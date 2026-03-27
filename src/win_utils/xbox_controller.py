# xbox_controller.py - Xbox 360 Gamepad Emulation Module
"""
Emulate Xbox 360 controller right stick using vgamepad library
Convert mouse movement (dx, dy) to right stick input for in-game view control

Principle:
- Create virtual Xbox 360 controller (via ViGEmBus driver)
- Map AI-calculated mouse movement to Right Stick offset
- Support sensitivity adjustment, deadzone setting, response curves, etc.
"""

from __future__ import annotations

import os
import sys
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ViGEmBus installer path (relative to this file)
_VIGEM_INSTALLER = os.path.join(os.path.dirname(__file__), "ViGEmBus_1.22.0_x64_x86_arm64.exe")


def _is_vigem_error(exc: Exception) -> bool:
    """Determine if exception is caused by ViGEmBus driver not installed"""
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "vigem", "vigembus", "bus not found", "driver", "not found",
        "cannot connect", "failed to connect", "0xE0000001".lower(),
    ])


def _is_vigem_installed_in_registry() -> bool:
    """Check the Windows registry to see if ViGEmBus is already installed.

    Returns True even if the driver was installed so recently that it has not
    yet been activated by Windows (i.e. a reboot is pending).  In that case we
    should NOT re-run the installer or kill the app — we should simply tell the
    user to reboot.
    """
    try:
        import winreg
        paths = (
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        )
        for path in paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as parent:
                    count = winreg.QueryInfoKey(parent)[0]
                    for i in range(count):
                        try:
                            sub_name = winreg.EnumKey(parent, i)
                            with winreg.OpenKey(parent, sub_name) as sub:
                                name, _ = winreg.QueryValueEx(sub, "DisplayName")
                                if "ViGEm" in str(name):
                                    return True
                        except OSError:
                            continue
            except OSError:
                continue
    except Exception:
        pass
    return False


def _is_admin() -> bool:
    """Return True if the current process is running with admin privileges."""
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _launch_vigem_installer_and_exit() -> None:
    """Launch ViGEmBus installer and exit current program"""
    if os.path.exists(_VIGEM_INSTALLER):
        import subprocess
        print("[Xbox] ViGEmBus driver not detected, launching installer...")
        try:
            subprocess.Popen([_VIGEM_INSTALLER], shell=False)
        except Exception as launch_err:
            print(f"[Xbox] Could not launch installer: {launch_err}")
    else:
        print(f"[Xbox] Could not find ViGEmBus installer: {_VIGEM_INSTALLER}")
        print("[Xbox] Please manually download and install from https://github.com/nefarius/ViGEmBus/releases")

    print("[Xbox] Please restart Axiom after installation. Program will now close...")
    time.sleep(2)
    os._exit(0)

# vgamepad uses lazy import, only loaded when connect() is called
# to avoid triggering ViGEmBus driver check during module import
vg = None


def _import_vgamepad():
    """Attempt to import vgamepad; if ViGEmBus is not installed, launch installer and exit program"""
    global vg
    if vg is not None:
        return True
    try:
        import vgamepad as _vg
        vg = _vg
        return True
    except ImportError:
        print("[Xbox] vgamepad not installed, please run: pip install vgamepad")
        return False
    except Exception as e:
        # ViGEmBus driver not installed (e.g., VIGEM_ERROR_BUS_NOT_FOUND)
        print(f"[Xbox] vgamepad load failed: {e}")
        _launch_vigem_installer_and_exit()
        return False


class XboxController:
    """Xbox 360 Virtual Gamepad Controller
    
    Use vgamepad to create a virtual Xbox 360 controller,
    mapping mouse movement to right stick input.
    
    Attributes:
        sensitivity: Sensitivity multiplier (default 1.0)
        deadzone: Deadzone threshold, input below this value is ignored (default 0.05)
        stick_duration: Stick input duration (seconds) (default 0.03)
        max_stick_value: Maximum stick mapping value (0.0~1.0) (default 1.0)
    """
    
    def __init__(self) -> None:
        self._gamepad = None  # vg.VX360Gamepad instance or None
        self._lock = threading.Lock()
        self._connected = False
        self._init_attempted = False
        
        # Adjustable parameters
        self.sensitivity: float = 1.0
        self.deadzone: float = 0.05
        self.stick_duration: float = 0.008
        self.max_stick_value: float = 1.0
        
        # 統計
        self._move_count: int = 0
        self._error_count: int = 0
        self._last_error: str = ""
    
    def is_available(self) -> bool:
        """檢查 vgamepad 套件是否存在（不實際連線）"""
        try:
            import importlib.util
            return importlib.util.find_spec("vgamepad") is not None
        except Exception:
            return False
    
    def is_connected(self) -> bool:
        """檢查虛擬手把是否已連線"""
        return self._connected and self._gamepad is not None
    
    def connect(self) -> bool:
        """建立虛擬 Xbox 360 控制器

        Returns:
            是否成功建立
        """
        if not _import_vgamepad():
            return False

        with self._lock:
            if self._connected and self._gamepad is not None:
                return True

            # ── Pre-flight checks ──
            if not _is_vigem_installed_in_registry():
                print("[Xbox] WARNING: ViGEmBus driver not found in registry.")
                print("[Xbox] Virtual controller creation will likely fail.")
                print("[Xbox] Install from: https://github.com/nefarius/ViGEmBus/releases")
            if not _is_admin():
                logger.info(
                    "[Xbox] Process is NOT running as Administrator. "
                    "Some games require admin to accept virtual controller input."
                )

            try:
                # ── Snapshot connected XInput slots BEFORE creation ──
                # After the virtual device appears we diff the slot lists
                # to figure out which XInput index it occupies, then tell
                # gamepad_input to exclude that slot from physical reads.
                from .gamepad_input import (
                    snapshot_connected_slots,
                    set_virtual_slot,
                )
                pre_slots = snapshot_connected_slots()

                self._gamepad = vg.VX360Gamepad()
                self._connected = True
                self._init_attempted = True
                self._error_count = 0

                # ── Detect the virtual controller's XInput slot ──
                # Windows needs a moment to enumerate the new device.
                # We retry a few times because 150 ms is sometimes not
                # enough on slower machines or under heavy load.
                new_slots: set = set()
                for _attempt in range(4):
                    time.sleep(0.15)
                    post_slots = snapshot_connected_slots()
                    new_slots = post_slots - pre_slots
                    if new_slots:
                        break

                if new_slots:
                    vslot = min(new_slots)
                    set_virtual_slot(vslot)
                    logger.info(
                        "[Xbox] Virtual controller on XInput slot %d "
                        "(physical slots: %s)", vslot,
                        sorted(post_slots - new_slots) or "none detected",
                    )
                else:
                    # Could not diff — fall back to heuristic: if only one
                    # physical controller was present before creation, the
                    # virtual device is probably on a different slot.  As a
                    # safe default, exclude the highest connected slot (the
                    # virtual device is appended after existing ones).
                    if post_slots and len(post_slots) >= 2:
                        fallback = max(post_slots)
                        set_virtual_slot(fallback)
                        logger.warning(
                            "[Xbox] Could not diff virtual slot; "
                            "assuming slot %d (heuristic)", fallback,
                        )
                    else:
                        logger.warning(
                            "[Xbox] Could not determine virtual slot; "
                            "left-stick passthrough may read from virtual device",
                        )

                logger.info("[Xbox] 虛擬 Xbox 360 控制器已建立")
                print("[Xbox] 虛擬 Xbox 360 控制器已建立")

                # ── Double-input conflict check ──
                # Warn if both physical and virtual controllers are visible
                # and the physical one sits on a lower slot (game likely
                # ignores the virtual device in that case).
                try:
                    conflict = check_double_input_conflict()
                    if conflict["conflict"]:
                        print(
                            "[Xbox] WARNING: double-input conflict detected!\n"
                            f"  {conflict['recommendation']}"
                        )
                        logger.warning(
                            "[Xbox] Double-input conflict: %s",
                            conflict["recommendation"],
                        )
                except Exception:
                    pass  # non-critical

                # ── Anti-cheat warning ──
                try:
                    ac_hits = detect_anti_cheat()
                    if ac_hits:
                        names = ", ".join(h["description"] for h in ac_hits)
                        print(
                            f"[Xbox] WARNING: detected running anti-cheat: {names}\n"
                            "[Xbox] These may block virtual controller input.  "
                            "If the game ignores the virtual controller, consider "
                            "switching mouse_move_method to an alternative."
                        )
                        logger.warning("[Xbox] Anti-cheat detected: %s", names)
                except Exception:
                    pass  # non-critical

                return True
            except Exception as e:
                self._last_error = f"建立虛擬手把失敗: {e}"
                self._connected = False
                self._gamepad = None
                self._init_attempted = True
                logger.error(f"[Xbox] {self._last_error}")
                print(f"[Xbox] {self._last_error}")

                # Only run the installer + exit if ViGEmBus is genuinely absent.
                # If it IS in the registry the driver was recently installed and
                # needs a reboot to become active — don't kill the app for that.
                if _is_vigem_error(e):
                    if _is_vigem_installed_in_registry():
                        print("[Xbox] ViGEmBus driver is installed but not yet active.")
                        print("[Xbox] Please reboot your computer to activate it.")
                    else:
                        _launch_vigem_installer_and_exit()

                print("[Xbox] 請確認已安裝 ViGEmBus 驅動: https://github.com/nefarius/ViGEmBus/releases")
                return False
    
    def disconnect(self) -> None:
        """斷開虛擬手把"""
        with self._lock:
            if self._gamepad is not None:
                try:
                    # 重置所有輸入
                    self._gamepad.reset()
                    self._gamepad.update()
                except Exception:
                    pass
                self._gamepad = None
            self._connected = False

            # Clear virtual-slot exclusion so physical-controller reads
            # revert to the default full-scan behaviour.
            try:
                from .gamepad_input import clear_virtual_slot
                clear_virtual_slot()
            except Exception:
                pass

            logger.info("[Xbox] 虛擬手把已斷開")
            print("[Xbox] 虛擬手把已斷開")
    
    def ensure_initialized(self) -> bool:
        """確保手把已初始化，如果未初始化則嘗試連線"""
        if self._connected and self._gamepad is not None:
            return True
        return self.connect()
    
    def move_left_stick(self, x: float, y: float) -> bool:
        """Set the left stick to a normalized (x, y) position.

        Maps movement axes — typically driven by the physical left stick or
        AI-computed movement values — to the virtual controller's left stick.

        Args:
            x: Horizontal axis in [-1.0, 1.0].  Positive = right.
            y: Vertical   axis in [-1.0, 1.0].  Positive = forward (up in
               XInput convention; vgamepad Y is already +up).

        Returns:
            True on success, False if the virtual gamepad is not ready.
        """
        if not self.ensure_initialized():
            return False

        with self._lock:
            try:
                # Clamp to valid range.
                # NOTE: do NOT apply a second deadzone here — the caller
                # (get_left_stick → apply_radial_deadzone) already performed
                # proper radial deadzone processing.  Adding a per-axis
                # deadzone on top would create a cross-shaped dead region
                # and suppress small diagonal movements that legitimately
                # passed the radial check.
                norm_x = max(-1.0, min(1.0, x))
                norm_y = max(-1.0, min(1.0, y))

                # vgamepad left_joystick_float: positive Y = up (same as XInput)
                self._gamepad.left_joystick_float(
                    x_value_float=norm_x,
                    y_value_float=norm_y,
                )
                self._gamepad.update()
                return True
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                if self._error_count <= 3:
                    logger.error(f"[Xbox] 左搖桿移動失敗: {e}")
                if self._error_count > 5:
                    self._connected = False
                    self._gamepad = None
                return False

    def move_right_stick(self, dx: float, dy: float) -> bool:
        """移動右搖桿
        
        將滑鼠移動 (dx, dy) 映射到右搖桿偏移量。
        值域為 -1.0 到 1.0，其中：
        - X 軸: 負=左, 正=右
        - Y 軸: 負=上, 正=下 (注意: vgamepad Y 軸負=上)
        
        Args:
            dx: 水平移動量 (像素)
            dy: 垂直移動量 (像素)
            
        Returns:
            是否成功
        """
        if not self.ensure_initialized():
            return False
        
        # --- Pure arithmetic: no lock needed ---
        scaled_x = dx * self.sensitivity
        scaled_y = dy * self.sensitivity

        BASE_PIXELS = 50.0
        norm_x = max(-1.0, min(1.0, scaled_x / BASE_PIXELS))
        norm_y = max(-1.0, min(1.0, scaled_y / BASE_PIXELS))

        norm_x *= self.max_stick_value
        norm_y *= self.max_stick_value

        if abs(norm_x) < self.deadzone:
            norm_x = 0.0
        if abs(norm_y) < self.deadzone:
            norm_y = 0.0

        if norm_x == 0.0 and norm_y == 0.0:
            return True

        # --- LOCK BLOCK 1: push the right-stick value ---
        # The lock is held only for the duration of the vgamepad API call
        # (microseconds).  Previously the sleep below was inside this block,
        # which held the lock for 30 ms on every aim correction.  At a 60 Hz
        # left-stick passthrough rate (16 ms period) the lock was continuously
        # occupied, starving move_left_stick() and freezing in-game movement
        # whenever the AI was actively tracking a target.
        with self._lock:
            try:
                # Y軸反轉: vgamepad 正Y = 上; 遊戲 dy>0 = 向下
                self._gamepad.right_joystick_float(
                    x_value_float=norm_x,
                    y_value_float=-norm_y,
                )
                self._gamepad.update()
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                if self._error_count <= 3:
                    logger.error(f"[Xbox] 右搖桿移動失敗: {e}")
                if self._error_count > 5:
                    self._connected = False
                    self._gamepad = None
                return False

        # --- Hold duration OUTSIDE the lock ---
        # move_left_stick() (passthrough thread) can now acquire the lock
        # freely during this sleep and forward physical left-stick values to
        # the virtual gamepad without being starved.
        if self.stick_duration > 0:
            time.sleep(self.stick_duration)

        # --- LOCK BLOCK 2: reset the right stick to center ---
        with self._lock:
            try:
                self._gamepad.right_joystick_float(
                    x_value_float=0.0,
                    y_value_float=0.0,
                )
                self._gamepad.update()
                self._move_count += 1
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                if self._error_count <= 3:
                    logger.error(f"[Xbox] 右搖桿回中失敗: {e}")
                if self._error_count > 5:
                    self._connected = False
                    self._gamepad = None
                return False

        return True
    
    def press_button(self, button) -> bool:
        """按下手把按鈕
        
        Args:
            button: vgamepad 按鈕常數 (例如 vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            
        Returns:
            是否成功
        """
        if not self.ensure_initialized():
            return False
        
        with self._lock:
            try:
                self._gamepad.press_button(button=button)
                self._gamepad.update()
                return True
            except Exception as e:
                logger.error(f"[Xbox] 按鈕按下失敗: {e}")
                return False
    
    def release_button(self, button) -> bool:
        """釋放手把按鈕"""
        if not self.ensure_initialized():
            return False
        
        with self._lock:
            try:
                self._gamepad.release_button(button=button)
                self._gamepad.update()
                return True
            except Exception as e:
                logger.error(f"[Xbox] 按鈕釋放失敗: {e}")
                return False
    
    def click_button(self, button, duration: float = 0.05) -> bool:
        """點擊手把按鈕 (按下 + 釋放)
        
        Args:
            button: 按鈕常數
            duration: 按住時間（秒）
        """
        if self.press_button(button):
            time.sleep(duration)
            return self.release_button(button)
        return False
    
    def pull_right_trigger(self, value: float = 1.0) -> bool:
        """拉右扳機 (RT)
        
        Args:
            value: 0.0~1.0 (0=未按, 1=全按)
        """
        if not self.ensure_initialized():
            return False
        with self._lock:
            try:
                self._gamepad.right_trigger_float(value_float=value)
                self._gamepad.update()
                return True
            except Exception as e:
                logger.error(f"[Xbox] 右扳機失敗: {e}")
                return False
    
    def pull_left_trigger(self, value: float = 1.0) -> bool:
        """拉左扳機 (LT)"""
        if not self.ensure_initialized():
            return False
        with self._lock:
            try:
                self._gamepad.left_trigger_float(value_float=value)
                self._gamepad.update()
                return True
            except Exception as e:
                logger.error(f"[Xbox] 左扳機失敗: {e}")
                return False
    
    def reset(self) -> bool:
        """重置所有輸入"""
        if not self._connected or self._gamepad is None:
            return True
        with self._lock:
            try:
                self._gamepad.reset()
                self._gamepad.update()
                return True
            except Exception:
                return False
    
    def get_statistics(self) -> dict:
        """取得統計資料"""
        return {
            "connected": self._connected,
            "available": self.is_available(),
            "move_count": self._move_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "sensitivity": self.sensitivity,
            "deadzone": self.deadzone,
        }

    # ── Diagnostic helpers ──────────────────────────────────────────

    def diagnose(self) -> dict:
        """Return a diagnostic snapshot of the virtual controller state.

        Useful for out-of-game verification: call this from the console or
        a test script and check whether left-stick values are non-zero when
        the physical stick is deflected.

        Returns a dict with:
            connected       — bool, virtual controller alive
            report          — dict with stick/trigger values from the
                              vgamepad XUSB_REPORT (what the driver sees)
            virtual_slot    — int or None, the XInput slot we think belongs
                              to the virtual device
            xinput_slots    — dict[int, dict], XInput state for each of the
                              4 possible slots (so you can see which slots
                              are occupied and what values they report)
            vigem_registry  — bool, ViGEmBus found in the Windows registry
            admin           — bool, process running elevated
        """
        result: dict = {
            "connected": self._connected,
            "report": None,
            "virtual_slot": None,
            "xinput_slots": {},
            "vigem_registry": _is_vigem_installed_in_registry(),
            "admin": _is_admin(),
        }

        # Virtual controller report
        if self._gamepad is not None:
            try:
                r = self._gamepad.report
                result["report"] = {
                    "sThumbLX": r.sThumbLX,
                    "sThumbLY": r.sThumbLY,
                    "sThumbRX": r.sThumbRX,
                    "sThumbRY": r.sThumbRY,
                    "bLeftTrigger": r.bLeftTrigger,
                    "bRightTrigger": r.bRightTrigger,
                    "wButtons": r.wButtons,
                }
            except Exception as exc:
                result["report"] = f"error: {exc}"

        # Virtual slot
        try:
            from .gamepad_input import _virtual_slot
            result["virtual_slot"] = _virtual_slot
        except Exception:
            pass

        # XInput slot scan (all 4 slots, including virtual)
        try:
            from .gamepad_input import _load_xinput, _xinput, XINPUT_STATE
            import ctypes
            if _load_xinput():
                for slot in range(4):
                    state = XINPUT_STATE()
                    rc = _xinput.XInputGetState(slot, ctypes.byref(state))
                    if rc == 0:
                        gp = state.Gamepad
                        result["xinput_slots"][slot] = {
                            "sThumbLX": gp.sThumbLX,
                            "sThumbLY": gp.sThumbLY,
                            "sThumbRX": gp.sThumbRX,
                            "sThumbRY": gp.sThumbRY,
                            "bLeftTrigger": gp.bLeftTrigger,
                            "bRightTrigger": gp.bRightTrigger,
                            "wButtons": gp.wButtons,
                        }
                    else:
                        result["xinput_slots"][slot] = None  # not connected
        except Exception as exc:
            result["xinput_slots"] = f"error: {exc}"

        return result

    def diagnose_print(self) -> None:
        """Print a human-readable diagnostic to the console."""
        d = self.diagnose()
        print("=" * 60)
        print("  VIRTUAL CONTROLLER DIAGNOSTIC")
        print("=" * 60)
        print(f"  Connected:      {d['connected']}")
        print(f"  ViGEmBus reg:   {d['vigem_registry']}")
        print(f"  Admin elevated: {d['admin']}")
        print(f"  Virtual slot:   {d['virtual_slot']}")
        print()
        if isinstance(d["report"], dict):
            r = d["report"]
            print("  Virtual report (what the driver currently holds):")
            print(f"    Left  stick: ({r['sThumbLX']:+6d}, {r['sThumbLY']:+6d})")
            print(f"    Right stick: ({r['sThumbRX']:+6d}, {r['sThumbRY']:+6d})")
            print(f"    Triggers:    L={r['bLeftTrigger']}  R={r['bRightTrigger']}")
            print(f"    Buttons:     0x{r['wButtons']:04X}")
        else:
            print(f"  Virtual report: {d['report']}")
        print()
        print("  XInput slots (all 4, raw values):")
        if isinstance(d["xinput_slots"], dict):
            for slot in range(4):
                info = d["xinput_slots"].get(slot)
                if info is None:
                    tag = "(virtual)" if slot == d["virtual_slot"] else ""
                    print(f"    Slot {slot}: not connected {tag}")
                else:
                    tag = " ← VIRTUAL" if slot == d["virtual_slot"] else ""
                    print(
                        f"    Slot {slot}: LStick=({info['sThumbLX']:+6d},"
                        f"{info['sThumbLY']:+6d})  "
                        f"RStick=({info['sThumbRX']:+6d},"
                        f"{info['sThumbRY']:+6d}){tag}"
                    )
        else:
            print(f"    {d['xinput_slots']}")
        print("=" * 60)


# ===== 全域單例 =====
xbox_controller = XboxController()


# ===== 公開函數 =====

def send_mouse_move_xbox(dx: float, dy: float) -> None:
    """透過 Xbox 360 虛擬手把右搖桿發送移動

    與 send_mouse_move_sendinput / send_mouse_move_mouse_event 相同介面
    """
    xbox_controller.move_right_stick(dx, dy)


def send_movement_xbox(x: float, y: float) -> bool:
    """Send normalized movement axes via the virtual Xbox 360 left stick.

    Intended for games that read controller movement through the left stick.

    Args:
        x: Horizontal movement in [-1.0, 1.0].  Positive = strafe right.
        y: Vertical   movement in [-1.0, 1.0].  Positive = move forward.

    Returns:
        True on success, False if the virtual gamepad is not initialized.
    """
    return xbox_controller.move_left_stick(x, y)


def send_mouse_click_xbox(duration: float = 0.05) -> bool:
    """透過 Xbox 360 虛擬手把模擬射擊 (RT 扳機)"""
    if not xbox_controller.ensure_initialized():
        return False
    try:
        xbox_controller.pull_right_trigger(1.0)
        time.sleep(duration)
        xbox_controller.pull_right_trigger(0.0)
        return True
    except Exception:
        return False


def connect_xbox() -> bool:
    """連線虛擬 Xbox 360 手把"""
    return xbox_controller.connect()


def disconnect_xbox() -> None:
    """斷開虛擬 Xbox 360 手把"""
    xbox_controller.disconnect()


def is_xbox_connected() -> bool:
    """檢查虛擬手把是否已連線"""
    return xbox_controller.is_connected()


def is_xbox_available() -> bool:
    """檢查 vgamepad 是否可用"""
    return xbox_controller.is_available()


def set_xbox_sensitivity(value: float) -> None:
    """設定手把靈敏度"""
    xbox_controller.sensitivity = max(0.1, min(5.0, value))


def set_xbox_deadzone(value: float) -> None:
    """設定手把死區"""
    xbox_controller.deadzone = max(0.0, min(0.5, value))


def get_xbox_statistics() -> dict:
    """取得手把統計資料"""
    return xbox_controller.get_statistics()


def diagnose_xbox() -> dict:
    """Run the full virtual-controller diagnostic and return a dict.

    Also prints a human-readable summary to the console.
    """
    xbox_controller.diagnose_print()
    return xbox_controller.diagnose()


# =====================================================================
#  External Interference Detection
# =====================================================================

# Known anti-cheat process names (lowercase) that may block ViGEmBus
# virtual controller injection.  This is not exhaustive but covers the
# most common ones.
_KNOWN_ANTI_CHEAT_PROCESSES = {
    "vgc.exe":              "Vanguard (Riot / VALORANT)",
    "vgtray.exe":           "Vanguard Tray (Riot / VALORANT)",
    "easyanticheat.exe":    "Easy Anti-Cheat (EAC)",
    "eac_launcher.exe":     "Easy Anti-Cheat Launcher",
    "beclient.dll":         "BattlEye Client",
    "beservice.exe":        "BattlEye Service",
    "bedaisy.sys":          "BattlEye Kernel Driver",
    "atvi-ricochet.exe":    "Ricochet (Activision / COD)",
    "nprotect.exe":         "nProtect GameGuard",
    "gameguard.des":        "GameGuard",
    "xigncode.exe":         "XIGNCODE3",
    "uncheater.exe":        "Uncheater",
    "faceit.exe":           "FACEIT Anti-Cheat",
    "faceitclient.exe":     "FACEIT Client",
}


def detect_anti_cheat() -> list[dict]:
    """Scan running processes for known anti-cheat software.

    Returns a list of dicts with 'process' and 'description' keys.
    An empty list means no known anti-cheat was detected (does not
    guarantee none is running — kernel-level drivers are invisible
    to user-mode process enumeration).
    """
    import ctypes
    import ctypes.wintypes

    detected: list[dict] = []

    # Snapshot all running processes via the Toolhelp API
    TH32CS_SNAPPROCESS = 0x00000002
    try:
        kernel32 = ctypes.windll.kernel32

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.wintypes.DWORD),
                ("cntUsage", ctypes.wintypes.DWORD),
                ("th32ProcessID", ctypes.wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", ctypes.wintypes.DWORD),
                ("cntThreads", ctypes.wintypes.DWORD),
                ("th32ParentProcessID", ctypes.wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.wintypes.DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == -1:
            return detected

        pe = PROCESSENTRY32()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32)

        if kernel32.Process32First(snap, ctypes.byref(pe)):
            while True:
                name = pe.szExeFile.decode("utf-8", errors="ignore").lower()
                if name in _KNOWN_ANTI_CHEAT_PROCESSES:
                    detected.append({
                        "process": name,
                        "description": _KNOWN_ANTI_CHEAT_PROCESSES[name],
                    })
                if not kernel32.Process32Next(snap, ctypes.byref(pe)):
                    break

        kernel32.CloseHandle(snap)
    except Exception as exc:
        logger.warning("[Xbox] Anti-cheat scan failed: %s", exc)

    return detected


def check_double_input_conflict() -> dict:
    """Check whether the game might see both physical and virtual controllers.

    Many games bind to the first XInput device (slot 0).  If both the
    physical and virtual controllers are visible, the game may read only
    one of them — typically whichever occupies the lowest slot.

    Returns a dict:
        physical_slots   — list of slots occupied by physical controllers
        virtual_slot     — int or None
        conflict         — bool, True if both are visible and the physical
                           sits on a lower slot (game likely ignores virtual)
        recommendation   — str, plain-English guidance
    """
    from .gamepad_input import _virtual_slot, snapshot_connected_slots

    all_slots = snapshot_connected_slots()
    vs = _virtual_slot
    physical = sorted(all_slots - {vs}) if vs is not None else sorted(all_slots)

    conflict = False
    recommendation = "No conflict detected."

    if vs is not None and physical:
        if min(physical) < vs:
            # Game probably reads the physical controller (lower slot) and
            # ignores the virtual one.  Movement comes from the passthrough
            # writing to the virtual stick, but if the game never reads the
            # virtual device it will never see those values.
            conflict = True
            recommendation = (
                f"The physical controller is on slot {min(physical)} and the "
                f"virtual controller is on slot {vs}.  The game may only read "
                f"slot {min(physical)} (physical), ignoring the virtual device.  "
                "Options:\n"
                "  1. Use HidHide to hide the physical controller from the game "
                "     while keeping it visible to this app via XInput.\n"
                "  2. Unplug the physical controller and re-plug it AFTER the "
                "     virtual controller is created (so it gets a higher slot).\n"
                "  3. Switch mouse_move_method to a non-xbox method if controller "
                "     output is not required."
            )
        else:
            recommendation = (
                f"Virtual controller is on slot {vs} (lower than physical "
                f"slot {min(physical)}).  The game should prefer the virtual "
                "device.  If movement still does not work, the game may be "
                "ignoring ViGEmBus devices (anti-cheat or game-specific filter)."
            )

    return {
        "physical_slots": physical,
        "virtual_slot": vs,
        "conflict": conflict,
        "recommendation": recommendation,
    }
