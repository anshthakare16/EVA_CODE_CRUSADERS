import google.generativeai as genai
from execution.executor_bridge import ExecutorBridge
from vision.omniparser_executor import OmniParserExecutor
from vision.screenshot_handler import ScreenshotHandler
from utils.logger import setup_logger
import time
import config
import json

class ExecutionHandler:
    def __init__(self):
        self.logger = setup_logger('ExecutionHandler')
        self.executor_bridge = ExecutorBridge()
        self.omniparser_executor = OmniParserExecutor()
        self.screenshot_handler = ScreenshotHandler()
        genai.configure(api_key=config.GEMINI_API_KEY)

    def execute_steps(self, steps):
        """
        Executes a list of steps.
        """
        self.logger.info("Starting execution of steps...")
        for i, step in enumerate(steps):
            self.logger.info(f"Executing step {i+1}/{len(steps)}: {step['description']}")
            action_type = step.get('action_type')
            parameters = step.get('parameters', {})
            
            if action_type == "WAIT":
                duration = parameters.get('duration', 1)
                time.sleep(duration)
                continue

            if action_type == "SCREEN_ANALYSIS":
                self.logger.info("--- SCREEN ANALYSIS ---")
                target = parameters.get('target')
                if not target:
                    self.logger.warning("No target specified for screen analysis.")
                    continue

                self.logger.info(f"Target: {target}")
                screenshot_path = self.screenshot_handler.capture()

                if not screenshot_path:
                    self.logger.error("Failed to capture screenshot.")
                    continue

                self.logger.info(f"Screenshot captured: {screenshot_path}")
                elements = self.omniparser_executor.parse_screen(screenshot_path, f"Find the best match for {target}")

                if not elements or not elements.get('elements'):
                    self.logger.warning("OmniParser found no elements on the screen.")
                    continue

                self.logger.info(f"OmniParser found {len(elements['elements'])} elements.")

                best_match = self.find_best_match(target, elements['elements'])

                if not best_match:
                    self.logger.error("Could not determine the best match.")
                    continue

                self.logger.info(f"Best match determined: {best_match}")

                if 'x' not in best_match or 'y' not in best_match:
                    self.logger.error("Best match JSON does not contain 'x' and 'y' coordinates.")
                    continue

                self.logger.info(f"Executing click at ({best_match['x']}, {best_match['y']})")
                self.executor_bridge.execute_action(
                    action_type="MOUSE_CLICK",
                    coordinates=best_match,
                    parameters={'button': 'left'}
                )
                self.logger.info("--- SCREEN ANALYSIS COMPLETE ---")
                continue

            # For other actions, use the executor bridge
            self.executor_bridge.execute_action(
                action_type=action_type,
                coordinates={},
                parameters=parameters
            )

        self.logger.info("Finished executing steps.")

    def find_best_match(self, target, elements):
        """
        Finds the best matching element from the list of elements.
        """
        raw_response_text = ""
        try:
            model = genai.GenerativeModel(config.GEMINI_MODEL)
            prompt = f"Given the user's request to find '{target}', which of the following UI elements is the best match? Respond with only the JSON object of the best match, and nothing else. UI elements: {elements}"
            self.logger.info("Sending prompt to Gemini to find best match...")
            
            response = model.generate_content(prompt)
            raw_response_text = response.text
            self.logger.info(f"Received response from Gemini: {raw_response_text}")

            # Clean the response to ensure it's valid JSON
            if raw_response_text.strip().startswith("```json"):
                cleaned_response = raw_response_text.strip()[7:-3].strip()
            else:
                cleaned_response = raw_response_text.strip()

            self.logger.info(f"Cleaned response: {cleaned_response}")
            
            parsed_json = json.loads(cleaned_response)
            self.logger.info("Successfully parsed JSON response.")
            return parsed_json

        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError: Failed to parse Gemini response: {e}")
            self.logger.error(f"Raw response was: {raw_response_text}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in find_best_match: {e}")
            if raw_response_text:
                self.logger.error(f"Raw response was: {raw_response_text}")
            return None