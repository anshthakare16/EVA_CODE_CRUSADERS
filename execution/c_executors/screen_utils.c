#include <windows.h>

int get_screen_width() {
    return GetSystemMetrics(SM_CXSCREEN);
}

int get_screen_height() {
    return GetSystemMetrics(SM_CYSCREEN);
}