# tests/test_gamepad_input.py
"""
手柄輸入模組單元測試

測試範圍：
1. is_gamepad_vk - 虛擬鍵碼範圍判斷
2. GP_VK 常數完整性
3. GP_VK_TRANSLATION_MAP 完整性
4. _BUTTON_FLAG_TO_GP_VK / _GP_VK_TO_BUTTON_FLAG 映射
"""

import pytest


class TestIsGamepadVk:
    """測試手柄虛擬鍵碼判斷"""

    def test_valid_range(self):
        from win_utils.gamepad_input import is_gamepad_vk, GP_VK_MIN, GP_VK_MAX
        for vk in range(GP_VK_MIN, GP_VK_MAX + 1):
            assert is_gamepad_vk(vk) is True

    def test_below_range(self):
        from win_utils.gamepad_input import is_gamepad_vk, GP_VK_MIN
        assert is_gamepad_vk(GP_VK_MIN - 1) is False
        assert is_gamepad_vk(0) is False
        assert is_gamepad_vk(0x01) is False  # Mouse Left

    def test_above_range(self):
        from win_utils.gamepad_input import is_gamepad_vk, GP_VK_MAX
        assert is_gamepad_vk(GP_VK_MAX + 1) is False
        assert is_gamepad_vk(0xFFFF) is False


class TestGamepadConstants:
    """測試手柄常數完整性"""

    def test_all_vk_constants_in_range(self):
        from win_utils.gamepad_input import (
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
            GP_VK_MIN, GP_VK_MAX, is_gamepad_vk,
        )
        all_vks = [
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
        ]
        assert len(all_vks) == 16
        for vk in all_vks:
            assert is_gamepad_vk(vk) is True

    def test_vk_values_unique(self):
        from win_utils.gamepad_input import (
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
        )
        all_vks = [
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
        ]
        assert len(set(all_vks)) == 16


class TestTranslationMap:
    """測試翻譯映射"""

    def test_all_buttons_have_translation_key(self):
        from win_utils.gamepad_input import (
            GP_VK_TRANSLATION_MAP,
            GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
            GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
            GP_VK_BACK, GP_VK_START,
            GP_VK_LSTICK, GP_VK_RSTICK,
            GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
            GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT,
        )
        for vk in [GP_VK_A, GP_VK_B, GP_VK_X, GP_VK_Y,
                    GP_VK_LB, GP_VK_RB, GP_VK_LT, GP_VK_RT,
                    GP_VK_BACK, GP_VK_START,
                    GP_VK_LSTICK, GP_VK_RSTICK,
                    GP_VK_DPAD_UP, GP_VK_DPAD_DOWN,
                    GP_VK_DPAD_LEFT, GP_VK_DPAD_RIGHT]:
            assert vk in GP_VK_TRANSLATION_MAP

    def test_translation_keys_prefix(self):
        from win_utils.gamepad_input import GP_VK_TRANSLATION_MAP
        for key, value in GP_VK_TRANSLATION_MAP.items():
            assert value.startswith("key_gp_")


class TestButtonFlagMapping:
    """測試 XInput 按鈕旗標映射"""

    def test_bidirectional_mapping(self):
        from win_utils.gamepad_input import _BUTTON_FLAG_TO_GP_VK, _GP_VK_TO_BUTTON_FLAG
        # 正向映射的每個值都應在反向映射中
        for flag, gp_vk in _BUTTON_FLAG_TO_GP_VK.items():
            assert gp_vk in _GP_VK_TO_BUTTON_FLAG
            assert _GP_VK_TO_BUTTON_FLAG[gp_vk] == flag

    def test_has_14_digital_buttons(self):
        """數位按鈕映射不含扳機（扳機用類比處理）"""
        from win_utils.gamepad_input import _BUTTON_FLAG_TO_GP_VK
        assert len(_BUTTON_FLAG_TO_GP_VK) == 14  # A,B,X,Y,LB,RB,Back,Start,LStick,RStick,DU,DD,DL,DR


class TestVirtualSlotExclusion:
    """Verify that set_virtual_slot / clear_virtual_slot properly gate reads."""

    def setup_method(self):
        from win_utils import gamepad_input
        # Reset module-level state between tests
        gamepad_input._virtual_slot = None

    def teardown_method(self):
        from win_utils import gamepad_input
        gamepad_input._virtual_slot = None

    def test_set_virtual_slot_stores_value(self):
        from win_utils.gamepad_input import set_virtual_slot, _virtual_slot
        import win_utils.gamepad_input as gi
        set_virtual_slot(2)
        assert gi._virtual_slot == 2

    def test_clear_virtual_slot_resets(self):
        from win_utils.gamepad_input import set_virtual_slot, clear_virtual_slot
        import win_utils.gamepad_input as gi
        set_virtual_slot(1)
        clear_virtual_slot()
        assert gi._virtual_slot is None

    def test_set_virtual_slot_invalidates_active_cache(self):
        """When the active slot IS the virtual slot, it must be invalidated."""
        import win_utils.gamepad_input as gi
        gi._active_slot = 0
        gi.set_virtual_slot(0)
        assert gi._active_slot != 0, (
            "_active_slot should be invalidated when it matches the virtual slot"
        )

    def test_set_virtual_slot_keeps_active_cache_if_different(self):
        """When the active slot differs from the virtual slot, leave it alone."""
        import win_utils.gamepad_input as gi
        gi._active_slot = 1
        gi.set_virtual_slot(0)
        assert gi._active_slot == 1


class TestRadialDeadzone:
    """Verify apply_radial_deadzone math."""

    def test_inside_deadzone_returns_zero(self):
        from win_utils.gamepad_input import apply_radial_deadzone
        x, y = apply_radial_deadzone(100, 100, 7849)
        assert x == 0.0
        assert y == 0.0

    def test_outside_deadzone_returns_nonzero(self):
        from win_utils.gamepad_input import apply_radial_deadzone
        x, y = apply_radial_deadzone(20000, 20000, 7849)
        assert x != 0.0
        assert y != 0.0

    def test_max_deflection_clamps_to_one(self):
        from win_utils.gamepad_input import apply_radial_deadzone
        x, y = apply_radial_deadzone(32767, 0, 7849)
        assert abs(x) <= 1.0
        assert y == 0.0

    def test_negative_values(self):
        from win_utils.gamepad_input import apply_radial_deadzone
        x, y = apply_radial_deadzone(-25000, -25000, 7849)
        assert x < 0.0
        assert y < 0.0
