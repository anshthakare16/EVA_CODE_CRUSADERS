"""
Executor Bridge - Complete C Integration
Bridges Python to C executors with full function support
"""

import ctypes
import os
import platform
from pathlib import Path
from utils.logger import setup_logger
import pygetwindow as gw
import subprocess


class ExecutorBridge:
    """Bridge between Python and C executors"""
    
    def __init__(self):
        self.logger = setup_logger('ExecutorBridge')
        self.system_platform = platform.system()
        self.c_lib = None
        
        # Load C library
        try:
            lib_path = self._get_library_path()
            self.c_lib = ctypes.CDLL(lib_path)
            self._setup_functions()
            self.logger.info("C executor library loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load C library: {e}")
            raise Exception("C executor library is REQUIRED. Please compile C executors.")
    
    def _get_library_path(self):
        """Get path to compiled C library"""
        base_path = Path(__file__).parent / 'c_executors'
        
        if self.system_platform == 'Windows':
            lib_name = 'executor.dll'
        elif self.system_platform == 'Darwin':
            lib_name = 'executor.dylib'
        else:
            lib_name = 'executor.so'
        
        lib_path = base_path / lib_name
        
        if not lib_path.exists():
            raise FileNotFoundError(f"C library not found at {lib_path}")
        
        return str(lib_path.absolute())
    
    def _setup_functions(self):
        """Setup C function signatures"""
        # Mouse functions
        self.c_lib.mouse_move.argtypes = [ctypes.c_int, ctypes.c_int]
        self.c_lib.mouse_move.restype = ctypes.c_int
        
        self.c_lib.mouse_click.argtypes = [ctypes.c_int]
        self.c_lib.mouse_click.restype = ctypes.c_int
        
        self.c_lib.mouse_scroll.argtypes = [ctypes.c_int]
        self.c_lib.mouse_scroll.restype = ctypes.c_int
        
        # Keyboard functions
        self.c_lib.keyboard_press_key.argtypes = [ctypes.c_int]
        self.c_lib.keyboard_press_key.restype = ctypes.c_int

        self.c_lib.keyboard_release_key.argtypes = [ctypes.c_int]
        self.c_lib.keyboard_release_key.restype = ctypes.c_int
        
        self.c_lib.keyboard_type_string.argtypes = [ctypes.c_char_p]
        self.c_lib.keyboard_type_string.restype = ctypes.c_int
    
    def launch_application(self, app_name=None, url=None):
        """Launch application or open URL"""
        try:
            if url:
                self.logger.info(f"Opening URL: {url}")
                if self.system_platform == 'Windows':
                    os.startfile(url)
                else:
                    subprocess.Popen(['xdg-open', url])  # For Linux
                return {'success': True}
            elif app_name:
                self.logger.info(f"Launching application: {app_name}")
                # Press Windows key
                self.c_lib.keyboard_press_key(0x5B)  # VK_LWIN
                import time
                time.sleep(0.5)
                
                # Type app name
                app_bytes = app_name.encode('utf-8')
                self.c_lib.keyboard_type_string(app_bytes)
                time.sleep(0.5)
                
                # Press Enter
                self.c_lib.keyboard_press_key(0x0D)  # VK_RETURN
                
                return {'success': True}
            else:
                return {'success': False, 'error': "No application name or URL provided"}
        except Exception as e:
            self.logger.error(f"Launch error: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_action(self, action_type, coordinates, parameters):
        """Execute generic action"""
        try:
            if action_type == 'MOUSE_CLICK':
                x = coordinates.get('x', 0)
                y = coordinates.get('y', 0)
                self.c_lib.mouse_move(x, y)
                import time
                time.sleep(0.1)
                button = 0 if parameters.get('button') == 'left' else 1
                result = self.c_lib.mouse_click(button)
                return {'success': result == 0}
            
            elif action_type == 'TYPE_TEXT':
                text = parameters.get('text', '')
                text_bytes = text.encode('utf-8')
                result = self.c_lib.keyboard_type_string(text_bytes)
                return {'success': result == 0}
            
            elif action_type == 'PRESS_KEY':
                key = parameters.get('key', '')
                return self._press_key_combination(key)
            
            elif action_type == 'MOUSE_SCROLL':
                amount = parameters.get('amount', 0)
                result = self.c_lib.mouse_scroll(amount)
                return {'success': result == 0}
            
            else:
                return {'success': False, 'error': f'Unknown action: {action_type}'}
        
        except Exception as e:
            self.logger.error(f"Action execution error: {e}")
            return {'success': False, 'error': str(e)}

    def _press_key_combination(self, key_str):
        """Press a combination of keys"""
        keys = [k.strip() for k in key_str.split('+')]
        vk_codes = [self._key_to_vk(key) for key in keys]

        # Press keys
        for vk_code in vk_codes:
            if vk_code != 0x00:
                self.c_lib.keyboard_press_key(vk_code)

        # Add a small delay
        import time
        time.sleep(0.1)

        # Release keys in reverse order
        for vk_code in reversed(vk_codes):
            if vk_code != 0x00:
                self.c_lib.keyboard_release_key(vk_code)

        return {'success': True}
    
    def _key_to_vk(self, key):
        """Convert key string to virtual key code"""
        key_map = {
            'enter': 0x0D,
            'tab': 0x09,
            'escape': 0x1B,
            'space': 0x20,
            'backspace': 0x08,
            'delete': 0x2E,
            'win': 0x5B,
            'ctrl': 0x11,
            'alt': 0x12,
            'shift': 0x10,
            'up': 0x26,
            'down': 0x28,
            'left': 0x25,
            'right': 0x27,
            'home': 0x24,
            'end': 0x23,
            'pageup': 0x21,
            'pagedown': 0x22,
            'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73, 'f5': 0x74, 'f6': 0x75, 
            'f7': 0x76, 'f8': 0x77, 'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
            'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45, 'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 
            'j': 0x4A, 'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F, 'p': 0x50, 'q': 0x51, 'r': 0x52, 
            's': 0x53, 't': 0x54, 'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59, 'z': 0x5A,
            '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34, '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
            '/': 0xBF,
        }
        return key_map.get(key.lower(), 0x00)

    def focus_window_by_title(self, title):
        """Focus a window by its title using fuzzy matching"""
        try:
            from thefuzz import fuzz

            self.logger.info(f"Attempting to focus window with title: {title}")
            windows = gw.getAllWindows()
            
            if not windows:
                self.logger.warning("No open windows found.")
                return {'success': False, 'error': "No open windows found."}

            best_match = None
            highest_score = 0

            for window in windows:
                score = fuzz.partial_ratio(title.lower(), window.title.lower())
                if score > highest_score:
                    highest_score = score
                    best_match = window

            if best_match and highest_score > 80:
                self.logger.info(f"Best match: '{best_match.title}' with score {highest_score}")
                if best_match.isMinimized:
                    best_match.restore()
                best_match.activate()
                self.logger.info(f"Successfully focused window: {best_match.title}")
                return {'success': True}
            else:
                self.logger.warning(f"No suitable window found for title '{title}'. Best match: '{best_match.title if best_match else 'None'}' with score {highest_score}")
                return {'success': False, 'error': f"Window with title like '{title}' not found."}
        except Exception as e:
            self.logger.error(f"Failed to focus window: {e}")
            return {'success': False, 'error': str(e)}
