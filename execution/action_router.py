import logging
import time
from pynput.keyboard import Controller as PyKeyboardController, Key as PyKey
import pyautogui

logger = logging.getLogger("ActionRouter")

class ActionRouter:
    """Action Router with integrated vision capabilities."""
    
    def __init__(self, system_executor, screenshot_handler, screen_analyzer, omniparser):
        self.system_executor = system_executor
        self.screenshot_handler = screenshot_handler
        self.screen_analyzer = screen_analyzer
        self.omniparser = omniparser
        self.py_keyboard = PyKeyboardController()
        logger.info("✓ Action Router initialized with Vision and C Executor Bridge.")

    def execute(self, category, steps, entities, raw_command, classification):
        logger.info(f"Executing {len(steps)} steps for command: '{raw_command}'")
        if not steps:
            logger.error("No execution plan (steps) provided for command.")
            return {"success": False, "error": "No execution plan generated for the command."}
        
        try:
            for i, step in enumerate(steps):
                action_type = step.get('action_type')
                params = step.get('parameters', {})
                description = step.get('description', 'No description')

                logger.info(f"  -> Step {i+1}/{len(steps)}: {action_type} - {description}")

                if action_type == "PRESS_KEY":
                    key_str = params.get('key', '')
                    if key_str:
                        self.system_executor.executor.execute_action("PRESS_KEY", {}, {"key": key_str})
                        logger.info(f"  -> Action successful: Pressed key(s) '{key_str}'")
                
                elif action_type == "TYPE_TEXT":
                    text_to_type = params.get('text', '')
                    if text_to_type:
                        self.system_executor.executor.execute_action("TYPE_TEXT", {}, {"text": text_to_type})
                        logger.info(f"  -> Action successful: Typed '{text_to_type}'")
                
                elif action_type == "WAIT":
                    duration = params.get('duration', 0.5)
                    time.sleep(float(duration))

                elif action_type == "MOUSE_CLICK" or action_type == "SCREEN_ANALYSIS":
                    # Vision-powered click with improved accuracy
                    target_description = description  # Use the full description as the target
                    logger.info(f"  -> Vision: Looking for '{target_description}'")
                    
                    # Capture screenshot
                    screenshot_path = self.screenshot_handler.capture()
                    if not screenshot_path:
                        logger.error("  -> Vision: Failed to capture screenshot. Skipping step.")
                        continue

                    # Parse screen elements
                    parse_result = self.omniparser.parse_screen(screenshot_path, raw_command)
                    elements = parse_result.get('elements', []) if parse_result else []
                    
                    if not elements:
                        logger.error("  -> Vision: OmniParser found no elements on screen.")
                        logger.warning("  -> Skipping this step - no valid targets found")
                        continue

                    # Log found elements for debugging
                    logger.info(f"  -> Vision: Found {len(elements)} elements on screen")
                    
                    # Extract profile name from entities
                    profile_name = entities.get('profile_name') if entities else None
                    
                    # Use ScreenAnalyzer to select the best coordinate
                    try:
                        coordinate = self.screen_analyzer.select_coordinate(
                            elements, 
                            target_description, 
                            step, 
                            profile_name=profile_name
                        )
                    except Exception as e:
                        logger.error(f"  -> Vision: Error in coordinate selection: {e}", exc_info=True)
                        coordinate = None

                    if coordinate and len(coordinate) == 2:
                        x, y = coordinate
                        
                        # Validate coordinates are within screen bounds
                        screen_width, screen_height = pyautogui.size()
                        if 0 <= x <= screen_width and 0 <= y <= screen_height:
                            logger.info(f"  -> Vision: Selected coordinate ({x}, {y}) for '{target_description}'")
                            
                            # Execute click at the selected coordinate
                            click_result = self.system_executor.executor.execute_action(
                                "MOUSE_CLICK", 
                                {'x': int(x), 'y': int(y)}, 
                                {"button": params.get('button', 'left')}
                            )
                            
                            logger.info(f"  -> Action successful: Clicked at ({int(x)}, {int(y)})")
                            
                            # Small delay after click for UI response
                            time.sleep(0.1)
                        else:
                            logger.error(f"  -> Vision: Invalid coordinates ({x}, {y}) - out of screen bounds ({screen_width}x{screen_height})")
                            logger.warning("  -> Skipping click - coordinates out of bounds")
                    else:
                        logger.warning(f"  -> Vision: Could not determine valid coordinate for '{target_description}'")
                        logger.info(f"  -> Available elements: {[el.get('label', 'unknown') for el in elements[:5]]}")
                        logger.warning("  -> Skipping this step - no matching target found")

                elif action_type == "SYSTEM_ACTION":
                    action = params.get('action', '')
                    if action:
                        self.system_executor.execute_system_command(action)
                        logger.info(f"  -> Action successful: Executed system action '{action}'")
                    else:
                        logger.warning("  -> SYSTEM_ACTION: No action specified")

                elif action_type == "OPEN_APP":
                    app_name = params.get('app_name', '')
                    if app_name:
                        try:
                            self.system_executor.executor.launch_application(app_name=app_name)
                            logger.info(f"  -> Action successful: Launched application '{app_name}'")
                        except Exception as e:
                            logger.error(f"  -> Failed to launch application '{app_name}': {e}")
                    else:
                        logger.warning("  -> OPEN_APP: No app name specified")

                elif action_type == "FOCUS_WINDOW":
                    title = params.get('title', '')
                    if title:
                        try:
                            self.system_executor.executor.focus_window_by_title(title)
                            logger.info(f"  -> Action successful: Focused window with title '{title}'")
                            time.sleep(0.2)  # Give window time to focus
                        except Exception as e:
                            logger.error(f"  -> Failedev to focus window '{title}': {e}")
                    else:
                        logger.warning("  -> FOCUS_WINDOW: No title specified")

                elif action_type == "OPEN_URL":
                    url = params.get('url', '')
                    if url:
                        try:
                            self.system_executor.executor.launch_application(url=url)
                            logger.info(f"  -> Action successful: Opened URL '{url}'")
                        except Exception as e:
                            logger.error(f"  -> Failed to open URL '{url}': {e}")
                    else:
                        logger.warning("  -> OPEN_URL: No URL specified")

                else:
                    logger.warning(f"  -> Unknown action_type: {action_type}. Skipping step.")
            
            return {"success": True, "message": "All steps executed successfully."}

        except Exception as e:
            logger.error(f"❌ Execution error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}