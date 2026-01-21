#ifndef EXECUTOR_H
#define EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport) int mouse_move(int x, int y);
__declspec(dllexport) int mouse_click(int button);
__declspec(dllexport) int mouse_scroll(int amount);
__declspec(dllexport) int keyboard_press_key(int vk_code);
__declspec(dllexport) int keyboard_release_key(int vk_code);
__declspec(dllexport) int keyboard_type_string(const char* text);

#ifdef __cplusplus
}
#endif

#endif // EXECUTOR_H
