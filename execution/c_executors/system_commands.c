#include <windows.h>
#include <powrprof.h>

int system_volume_up() {
    keybd_event(VK_VOLUME_UP, 0, 0, 0);
    Sleep(50);
    keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_KEYUP, 0);
    return 0;
}

int system_volume_down() {
    keybd_event(VK_VOLUME_DOWN, 0, 0, 0);
    Sleep(50);
    keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_KEYUP, 0);
    return 0;
}

int system_volume_mute() {
    keybd_event(VK_VOLUME_MUTE, 0, 0, 0);
    Sleep(50);
    keybd_event(VK_VOLUME_MUTE, 0, KEYEVENTF_KEYUP, 0);
    return 0;
}

int system_brightness_up() {
    keybd_event(VK_CONTROL, 0, 0, 0);
    keybd_event(VK_F6, 0, 0, 0);
    Sleep(50);
    keybd_event(VK_F6, 0, KEYEVENTF_KEYUP, 0);
    keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0);
    return 0;
}

int system_brightness_down() {
    keybd_event(VK_CONTROL, 0, 0, 0);
    keybd_event(VK_F5, 0, 0, 0);
    Sleep(50);
    keybd_event(VK_F5, 0, KEYEVENTF_KEYUP, 0);
    keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0);
    return 0;
}

int system_sleep() {
    SetSuspendState(FALSE, FALSE, FALSE);
    return 0;
}

int system_shutdown() {
    HANDLE hToken;
    TOKEN_PRIVILEGES tkp;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        return -1;
    LookupPrivilegeValue(NULL, SE_SHUTDOWN_NAME, &tkp.Privileges[0].Luid);
    tkp.PrivilegeCount = 1;
    tkp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    AdjustTokenPrivileges(hToken, FALSE, &tkp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
    if (GetLastError() != ERROR_SUCCESS)
        return -1;
    ExitWindowsEx(EWX_SHUTDOWN | EWX_FORCE, 0);
    return 0;
}

int system_restart() {
    HANDLE hToken;
    TOKEN_PRIVILEGES tkp;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        return -1;
    LookupPrivilegeValue(NULL, SE_SHUTDOWN_NAME, &tkp.Privileges[0].Luid);
    tkp.PrivilegeCount = 1;
    tkp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    AdjustTokenPrivileges(hToken, FALSE, &tkp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
    if (GetLastError() != ERROR_SUCCESS)
        return -1;
    ExitWindowsEx(EWX_REBOOT | EWX_FORCE, 0);
    return 0;
}
