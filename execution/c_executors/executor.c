#include <windows.h>
#include "executor.h"

int mouse_move(int x, int y) {
    SetCursorPos(x, y);
    return 0;
}

int mouse_click(int button) {
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    if (button == 0) { // Left click
        input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &input, sizeof(INPUT));
        Sleep(50);
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
    } else if (button == 1) { // Right click
        input.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
        SendInput(1, &input, sizeof(INPUT));
        Sleep(50);
        input.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
        SendInput(1, &input, sizeof(INPUT));
    }
    return 0;
}

int mouse_scroll(int amount) {
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_WHEEL;
    input.mi.mouseData = amount;
    SendInput(1, &input, sizeof(INPUT));
    return 0;
}

int keyboard_press_key(int vk_code) {
    INPUT input = {0};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = vk_code;
    SendInput(1, &input, sizeof(INPUT));
    return 0;
}

int keyboard_release_key(int vk_code) {
    INPUT input = {0};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = vk_code;
    input.ki.dwFlags = KEYEVENTF_KEYUP;
    SendInput(1, &input, sizeof(INPUT));
    return 0;
}

int keyboard_type_string(const char* text) {
    if (!text) return -1;
    for (int i = 0; text[i] != '\0'; ++i) {
        INPUT input = {0};
        input.type = INPUT_KEYBOARD;
        input.ki.wScan = text[i];
        input.ki.dwFlags = KEYEVENTF_UNICODE;
        SendInput(1, &input, sizeof(INPUT));
        Sleep(20);
        input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;
        SendInput(1, &input, sizeof(INPUT));
        Sleep(20);
    }
    return 0;
}