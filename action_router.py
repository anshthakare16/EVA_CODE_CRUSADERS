import logging
import time
from pynput.keyboard import Controller as PyKeyboardController, Key as PyKey
import pyautogui
from vision.edge_search_handler import EdgeSearchHandler

logger = logging.getLogger("ActionRouter")

class ActionRouter:
    """Action Router with integrated vision capabilities."""
    
    def __init__(self, system_executor, screenshot_handler, screen_analyzer, omniparser):
        self.system_executor = system_executor
        self.screenshot_handler = screenshot_handler
        self.screen_analyzer = screen_analyzer
        self.omniparser = omniparser
        self.py_keyboard = PyKeyboardController()
        
        # Initialize EdgeSearchHandler
        try:
            self.web_search_handler = EdgeSearchHandler()
            logger.info("✓ EdgeSearchHandler initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EdgeSearchHandler: {e}")
            import traceback
            traceback.print_exc()
            self.web_search_handler = None
        
        logger.info("✓ Action Router initialized with Vision and C Executor Bridge.")
    
    def execute(self, category, steps, entities, raw_command, classification):
        logger.info(f"Executing {len(steps)} steps for command: '{raw_command}'")
        print(f"[ACTIONROUTER] Starting execute with {len(steps)} steps")  # Direct print for threading
        
        if not steps:
            logger.error("No execution plan (steps) provided for command.")
            return {"success": False, "error": "No execution plan generated for the command."}
        
        try:
            for i, step in enumerate(steps):
                action_type = step.get('action_type', '').upper()
                params = step.get('parameters', {})
                description = step.get('description', 'No description')
                
                # Legacy support
                target = step.get('target', '')
                value = step.get('value', '')
                
                print(f"[ACTIONROUTER] Step {i+1}/{len(steps)}: action_type='{action_type}' | params={params}")  # Direct print
                logger.info(f" -> Step {i+1}/{len(steps)}: {action_type} - {description}")
                
                # ===== PRESS_KEY (for system commands) =====
                if action_type == "PRESS_KEY":
                    params = step.get('parameters', {})
                    key_str = params.get('key', '')
    
                    if key_str:
                        # Special handling for Windows key - use pyautogui
                        if 'LWIN' in key_str.upper() or 'WIN' in key_str.upper():
                            import pyautogui
                            if '+' in key_str:
                                # Combo like "LWIN+A" or "win+a"
                                keys = key_str.lower().replace('lwin', 'win').split('+')
                                pyautogui.hotkey(*keys)
                                logger.info(f" -> Pressed combo via pyautogui: {key_str}")
                            else:
                                # Just Windows key alone
                                pyautogui.press('win')
                                logger.info(f" -> Pressed Windows key via pyautogui")
                            time.sleep(0.5)  # Longer wait for Windows key
                        else:
                            # All other keys use C library
                            self.system_executor.executor.execute_action("PRESS_KEY", {}, {"key": key_str})
                            logger.info(f" -> Pressed key(s) '{key_str}' via ExecutorBridge")
                            time.sleep(0.1)
                
                                # ===== WAIT =====
                elif action_type == 'WAIT':
                    duration = params.get('duration', value or 0.5)
                    time.sleep(float(duration))
                    logger.info(f"⏳ Waited {duration}s")
                
                # ===== SET VOLUME =====
                elif action_type == 'SET_VOLUME':
                    level = params.get('level', value or 50)
                    try:
                        level = int(level)
                        result = self.system_executor.set_volume(level)
                        if not result.get('success'):
                            return result
                    except ValueError:
                        logger.error(f"Invalid volume level: {level}")
                        return {"success": False, "error": f"Invalid volume level: {level}"}
                
                # ===== SET BRIGHTNESS =====
                elif action_type == 'SET_BRIGHTNESS':
                    level = params.get('level', value or 50)
                    try:
                        level = int(level)
                        result = self.system_executor.set_brightness(level)
                        if not result.get('success'):
                            return result
                    except ValueError:
                        logger.error(f"Invalid brightness level: {level}")
                        return {"success": False, "error": f"Invalid brightness level: {level}"}
                
                # ===== SYSTEM_ACTION =====
                elif action_type == 'SYSTEM_ACTION':
                    action = params.get('action', target or raw_command)
                    result = self.system_executor.execute_system_command(action)
                    if not result.get('success'):
                        logger.error(f"System action failed: {result.get('error')}")
                        return result
                
                # ===== OPEN_APP =====
                elif action_type == 'OPEN_APP':
                    app_name = params.get('app_name', target)
                    if app_name:
                        try:
                            self.system_executor.executor.launch_application(app_name=app_name)
                            logger.info(f" -> Action successful: Launched application '{app_name}'")
                            time.sleep(1.5)
                        except Exception as e:
                            logger.error(f" -> Failed to launch application '{app_name}': {e}")
                
                # ===== TYPE_TEXT =====
                elif action_type == 'TYPE_TEXT':
                    text_to_type = params.get('text', value)
                    if text_to_type:
                        self.system_executor.executor.execute_action("TYPE_TEXT", {}, {"text": text_to_type})
                        logger.info(f" -> Action successful: Typed '{text_to_type}'")
                
                # ===== KEYBOARD_ACTION =====
                elif action_type == 'KEYBOARD_ACTION':
                    if target:
                        self._execute_keyboard_action(target, value)
                        time.sleep(0.3)
                
                # ===== MOUSE_ACTION or SCREEN_ANALYSIS (Vision-powered) =====
                elif action_type == 'MOUSE_ACTION' or action_type == 'MOUSE_CLICK' or action_type == 'SCREEN_ANALYSIS':
                    if target:
                        # Old method (without vision)
                        self._execute_mouse_action(target, value)
                        time.sleep(0.3)
                    else:
                        # Vision-powered click
                        target_description = description
                        logger.info(f" -> Vision: Looking for '{target_description}'")
                        
                        screenshot_path = self.screenshot_handler.capture()
                        if not screenshot_path:
                            logger.error(" -> Vision: Failed to capture screenshot")
                            continue
                        
                        parse_result = self.omniparser.parse_screen(screenshot_path, raw_command)
                        elements = parse_result.get('elements', []) if parse_result else []
                        
                        if not elements:
                            logger.warning(" -> Vision: No elements found, skipping")
                            continue
                        
                        logger.info(f" -> Vision: Found {len(elements)} elements")
                        
                        # Extract profile_name from step parameters (which contain the replaced {profile_name} value)
                        profile_name = step['parameters'].get('profile_name')
                        
                        try:
                            coordinate = self.screen_analyzer.select_coordinate(
                                elements, target_description, step, profile_name=profile_name
                            )
                        except Exception as e:
                            logger.error(f" -> Vision: Error in coordinate selection: {e}")
                            coordinate = None
                        
                        if coordinate and len(coordinate) == 2:
                            x, y = coordinate
                            screen_width, screen_height = pyautogui.size()
                            
                            if 0 <= x <= screen_width and 0 <= y <= screen_height:
                                self.system_executor.executor.execute_action(
                                    "MOUSE_CLICK",
                                    {'x': int(x), 'y': int(y)},
                                    {"button": params.get('button', 'left')}
                                )
                                logger.info(f" -> Action successful: Clicked at ({int(x)}, {int(y)})")
                                time.sleep(0.1)
                
                # ===== FOCUS_WINDOW =====
                elif action_type == 'FOCUS_WINDOW':
                    title = params.get('title', target)
                    if title:
                        try:
                            self.system_executor.executor.focus_window_by_title(title)
                            logger.info(f" -> Action successful: Focused window '{title}'")
                            time.sleep(0.2)
                        except Exception as e:
                            logger.error(f" -> Failed to focus window '{title}': {e}")
                
                # ===== OPEN_URL =====
                elif action_type == 'OPEN_URL':
                    url = params.get('url', target)
                    if url:
                        try:
                            self.system_executor.executor.launch_application(url=url)
                            logger.info(f" -> Action successful: Opened URL '{url}'")
                        except Exception as e:
                            logger.error(f" -> Failed to open URL '{url}': {e}")
                
                # ===== WINDOW_FOCUS =====
                elif action_type == 'WINDOW_FOCUS':
                    if target:
                        result = self._focus_window(target)
                        if not result.get('success'):
                            return result
                        time.sleep(0.5)
                
                # ===== VISION_CLICK =====
                elif action_type == 'VISION_CLICK':
                    if target:
                        result = self._vision_click(target)
                        if not result.get('success'):
                            return result
                        time.sleep(0.5)
                
                # ===== WEB_SEARCH_DIRECT (Selenium-based web search) =====
                elif action_type == 'WEB_SEARCH_DIRECT':
                    print(f"[ACTIONROUTER] 🔍 MATCHED WEB_SEARCH_DIRECT!")  # Direct print
                    website = params.get('website', 'google')
                    query = params.get('search_query', '')
                    
                    print(f"[ACTIONROUTER] Website: {website}, Query: {query}")  # Direct print
                    logger.info(f"🌐 WEB_SEARCH_DIRECT received:")
                    logger.info(f"   Website: {website}")
                    logger.info(f"   Query: {query}")
                    
                    if not query:
                        print(f"[ACTIONROUTER] ❌ No search query provided!")  # Direct print
                        logger.warning(" -> WEB_SEARCH_DIRECT: No search query provided")
                        continue
                    
                    if not self.web_search_handler:
                        print(f"[ACTIONROUTER] ❌ Handler not initialized!")  # Direct print
                        logger.error(" -> WEB_SEARCH_DIRECT: Handler not initialized!")
                        return {"success": False, "error": "EdgeSearchHandler not available"}
                    
                    print(f"[ACTIONROUTER] ✓ Handler exists, calling search()...")  # Direct print
                    logger.info(f" -> Opening {website} and searching for: {query}")
                    
                    try:
                        print(f"[ACTIONROUTER] Calling web_search_handler.search('{website}', '{query}')")  # Direct print
                        success = self.web_search_handler.search(website, query)
                        print(f"[ACTIONROUTER] Handler returned: {success}")  # Direct print
                        logger.info(f"   Handler returned: {success}")
                        
                        if not success:
                            print(f"[ACTIONROUTER] ❌ Search failed!")  # Direct print
                            logger.error(f" -> WEB_SEARCH_DIRECT failed for {website}: {query}")
                            return {"success": False, "error": f"Web search failed for {website}"}
                        
                        print(f"[ACTIONROUTER] ✅ Search successful!")  # Direct print
                        logger.info(f" -> Action successful: Searched {website} for '{query}'")
                    except Exception as e:
                        logger.error(f"❌ Exception in WEB_SEARCH_DIRECT: {e}")
                        import traceback
                        traceback.print_exc()
                        return {"success": False, "error": str(e)}
                
                # ===== WEB_OPEN_DIRECT (Open website without search) =====
                elif action_type == 'WEB_OPEN_DIRECT':
                    website = params.get('website', 'google')
                    
                    print(f"[ACTIONROUTER] 🌐 MATCHED WEB_OPEN_DIRECT!")  # Direct print
                    logger.info(f"🌐 WEB_OPEN_DIRECT received:")
                    logger.info(f"   Website: {website}")
                    
                    if not self.web_search_handler:
                        print(f"[ACTIONROUTER] ❌ Handler not initialized!")  # Direct print
                        logger.error(" -> WEB_OPEN_DIRECT: Handler not initialized!")
                        return {"success": False, "error": "EdgeSearchHandler not available"}
                    
                    print(f"[ACTIONROUTER] ✓ Handler exists, calling open_website()...")  # Direct print
                    logger.info(f" -> Opening website: {website}")
                    
                    try:
                        print(f"[ACTIONROUTER] Calling web_search_handler.open_website('{website}')")  # Direct print
                        success = self.web_search_handler.open_website(website)
                        print(f"[ACTIONROUTER] Handler returned: {success}")  # Direct print
                        logger.info(f"   Handler returned: {success}")
                        
                        if not success:
                            print(f"[ACTIONROUTER] ❌ Opening website failed!")  # Direct print
                            logger.error(f" -> WEB_OPEN_DIRECT failed for {website}")
                            return {"success": False, "error": f"Failed to open {website}"}
                        
                        print(f"[ACTIONROUTER] ✅ Website opened successfully!")  # Direct print
                        logger.info(f" -> Action successful: Opened {website}")
                    except Exception as e:
                        print(f"[ACTIONROUTER] ❌ Exception in WEB_OPEN_DIRECT: {e}")  # Direct print
                        logger.error(f"❌ Exception in WEB_OPEN_DIRECT: {e}")
                        import traceback
                        traceback.print_exc()
                        return {"success": False, "error": str(e)}
                
                else:
                    print(f"[ACTIONROUTER] ❌ UNKNOWN ACTION TYPE: '{action_type}'")  # Direct print
                    logger.warning(f" -> Unknown action_type: {action_type}")
            
            print(f"[ACTIONROUTER] ✅ All steps completed, returning success")  # Direct print
            return {"success": True, "message": "Command executed successfully"}
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_keyboard_action(self, action, value):
        """Execute keyboard-related actions"""
        action_lower = action.lower()
        
        if action_lower in ['type', 'write', 'input']:
            if value:
                pyautogui.write(value, interval=0.05)
                logger.info(f"⌨️ Typed: {value}")
        
        elif action_lower == 'press':
            if value:
                keys = value.lower().split('+')
                if len(keys) > 1:
                    pyautogui.hotkey(*keys)
                    logger.info(f"⌨️ Pressed: {' + '.join(keys)}")
                else:
                    pyautogui.press(value)
                    logger.info(f"⌨️ Pressed: {value}")
        
        elif action_lower in ['enter', 'return']:
            pyautogui.press('enter')
            logger.info("⌨️ Pressed Enter")
        
        elif action_lower == 'escape':
            pyautogui.press('escape')
            logger.info("⌨️ Pressed Escape")
        
        elif action_lower == 'tab':
            pyautogui.press('tab')
            logger.info("⌨️ Pressed Tab")
    
    def _execute_mouse_action(self, action, value):
        """Execute mouse-related actions"""
        action_lower = action.lower()
        
        if action_lower == 'click':
            pyautogui.click()
            logger.info("🖱️ Mouse clicked")
        
        elif action_lower == 'double_click':
            pyautogui.doubleClick()
            logger.info("🖱️ Double clicked")
        
        elif action_lower == 'right_click':
            pyautogui.rightClick()
            logger.info("🖱️ Right clicked")
        
        elif action_lower == 'move':
            if value:
                try:
                    coords = value.split(',')
                    x, y = int(coords[0]), int(coords[1])
                    pyautogui.moveTo(x, y, duration=0.3)
                    logger.info(f"🖱️ Moved to ({x}, {y})")
                except:
                    logger.error(f"Invalid coordinates: {value}")
    
    def _focus_window(self, window_title):
        """Focus on a specific window"""
        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(window_title)
            
            if windows:
                windows[0].activate()
                logger.info(f"🪟 Focused window: {window_title}")
                return {"success": True}
            else:
                logger.warning(f"Window not found: {window_title}")
                return {"success": False, "error": f"Window not found: {window_title}"}
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return {"success": False, "error": str(e)}
    
    def _vision_click(self, target_description):
        """Use vision AI to find and click an element"""
        try:
            logger.info(f"👁️ Vision click: {target_description}")
            
            screenshot_path = self.screenshot_handler.capture()
            result = self.omniparser.locate_element(screenshot_path, target_description)
            
            if result and 'coordinates' in result:
                x, y = result['coordinates']
                pyautogui.click(x, y)
                logger.info(f"✓ Vision clicked at ({x}, {y})")
                return {"success": True}
            else:
                logger.warning(f"Element not found: {target_description}")
                return {"success": False, "error": f"Element not found: {target_description}"}
                
        except Exception as e:
            logger.error(f"Vision click failed: {e}")
            return {"success": False, "error": str(e)}
