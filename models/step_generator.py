import logging
import json
import google.generativeai as genai

logger = logging.getLogger("StepGenerator")

class StepGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        models_to_try = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-pro'
        ]
        for model_name in models_to_try:
            try:
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"✓ StepGenerator initialized with: {model_name}")
                break
            except Exception as e:
                logger.debug(f"Model {model_name} unavailable: {e}")
        if not hasattr(self, "model"):
            raise RuntimeError("No Gemini models available! Check API key and enable in Studio.")

    def generate(self, command_data):
        category = command_data['classification']['category']
        raw_command = command_data['raw_command']
        screen_summary = command_data.get('screen_summary', '')

        if category == 'SYSTEM_ACTION':
            logger.info("📍 SYSTEM_ACTION - no steps needed")
            return []
        if category == 'APP_LAUNCH':
            app_name = command_data['classification'].get('entities', {}).get('app_name', '')
            logger.info(f"📍 APP_LAUNCH - fast path for {app_name}")
            return [
                {"action": "press_key", "key": "win", "description": "Open Start menu"},
                {"action": "wait", "duration": 0.5},
                {"action": "type", "text": app_name},
                {"action": "press_key", "key": "enter"}
            ]
        close_keywords = ['close', 'exit', 'quit', 'shut', 'end', 'stop', 'kill']
        if any(keyword in raw_command.lower() for keyword in close_keywords):
            logger.info("🔴 Close command detected - generating close steps")
            return [
                {"action": "press_key", "key": "alt+f4", "description": "Close active window"}
            ]

        prompt = (
            "You are a Windows automation planner for a voice assistant. "
            f"Voice command: \"{raw_command}\"\n"
            f"Command category: {category}\n"
            f"Screen summary: \"{screen_summary}\".\n"
            "Generate an array (not an object, not code block) containing only JSON steps to automate the request. "
            "CRITICAL: Output only a JSON array of steps as raw text—no markdown, no code block, no explanation.\n"
            "Each step must be an object with: action (press_key, type, wait, open_app, ui_click, ui_type), parameters.\n"
            "Example output:\n"
            "[{\"action\": \"press_key\", \"key\": \"win\"},"
            "{\"action\": \"type\", \"text\": \"edge\"},"
            "{\"action\": \"press_key\", \"key\": \"enter\"}]"
            "\nNow generate the full steps for the user's request. Output *only* the JSON array."
        )

        response = self.model.generate_content(prompt)
        response_text = response.text.strip()

        # Universal step extraction for Gemini output
        # Remove any markdown/code blocks, only JSON array allowed
        if response_text.startswith("```"):
            response_text = response_text.replace("```json", '').replace("```")
        if response_text.startswith("["):
            json_str = response_text
        else:
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            json_str = response_text[start:end]
        try:
            steps = json.loads(json_str)
            logger.info(f"✓ Generated {len(steps)} steps from Gemini.")
            assert isinstance(steps, list)
            return steps
        except Exception as e:
            # Return empty if parsing fails
            logger.error(f"ERROR: Gemini response could not be parsed to list of steps! Response: {response_text}")
            raise RuntimeError(f"Gemini output not parseable as JSON array: {e}")

