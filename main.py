

import tkinter as tk
from tkinter import scrolledtext
import re
import threading
import time

import speech_recognition as sr

from dotenv import load_dotenv
import config
from execution.executor_bridge import ExecutorBridge
from execution.action_router import ActionRouter
from execution.system_executor import SystemExecutor
from vision.screenshot_handler import ScreenshotHandler
from vision.screen_analyzer import ScreenAnalyzer
from vision.omniparser_executor import OmniParserExecutor
from speech.wake_word_detector import WakeWordDetector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load environment variables from .env file
load_dotenv()

# ============================================================================ 
# EVA_TER LOGIC (INTEGRATED)
# ============================================================================ 

MODEL1_TRAINING_DATA = [
    ("open application", "OPEN_APP"), ("launch program", "OPEN_APP"), ("start software", "OPEN_APP"),
    ("run app", "OPEN_APP"), ("open app", "OPEN_APP"), ("open chrome", "OPEN_APP"), ("launch spotify", "OPEN_APP"),
    ("close application", "CLOSE_APP"), ("close this", "CLOSE_APP"), ("close window", "CLOSE_APP"),
    ("exit application", "CLOSE_APP"), ("quit app", "CLOSE_APP"),
    ("open file explorer", "OPEN_FILE_EXPLORER"), ("open file manager", "OPEN_FILE_EXPLORER"),
    ("search for file", "SEARCH_FILE"), ("find document", "SEARCH_FILE"),
    ("open documents", "OPEN_FOLDER"), ("open downloads", "OPEN_FOLDER"), ("open pictures", "OPEN_FOLDER"),
    ("type text", "TYPE_TEXT"), ("write something", "TYPE_TEXT"), ("enter text", "TYPE_TEXT"),
    ("click on something", "MOUSE_CLICK"), ("click here", "MOUSE_CLICK"),
    ("right click", "MOUSE_RIGHTCLICK"), ("double click", "MOUSE_DOUBLECLICK"),
    ("maximize window", "WINDOW_ACTION"), ("minimize window", "WINDOW_ACTION"), ("fullscreen mode", "WINDOW_ACTION"),
    ("take screenshot", "SYSTEM"), ("lock screen", "SYSTEM"),
    ("copy", "KEYBOARD"), ("paste", "KEYBOARD"), ("save", "KEYBOARD"), ("undo", "KEYBOARD"),
    ("open app and search", "APP_WITH_ACTION"), ("launch app and type", "APP_WITH_ACTION"),
    ("open app and play", "APP_WITH_ACTION"), ("start app and compose", "APP_WITH_ACTION"),
    ("play music", "MEDIA_CONTROL"), ("play video", "MEDIA_CONTROL"), ("stream music", "MEDIA_CONTROL"), ("stream video", "MEDIA_CONTROL"),
    ("open spotify and play", "MEDIA_CONTROL"), ("open youtube and play", "MEDIA_CONTROL"),
    ("send whatsapp to", "SEND_MESSAGE"), ("send message to", "SEND_MESSAGE"), ("whatsapp to", "SEND_MESSAGE"),
    ("email to", "SEND_MESSAGE"), ("post on social", "SEND_MESSAGE"), ("message to", "SEND_MESSAGE"),
    ("whatsapp mom", "SEND_MESSAGE"), ("email john", "SEND_MESSAGE"),
    ("search for something", "WEB_SEARCH"), ("google something", "WEB_SEARCH"), ("youtube search", "WEB_SEARCH"),
    ("open youtube", "WEB_SEARCH"), ("profile work search python", "WEB_SEARCH"), ("with profile personal search", "WEB_SEARCH"),
    ("chrome profile dev open youtube", "WEB_SEARCH"), ("open gmail", "WEB_SEARCH"), ("go to facebook", "WEB_SEARCH"),
    ("search amazon", "WEB_SEARCH"), 
]

STEP_TEMPLATES = {
    "open_app_windows": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
        {"action_type": "WAIT", "parameters": {"duration": 0.5}, "description": "Wait for menu"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{app_name}"}, "description": "Type: {app_name}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Launch {app_name}"},
        {"action_type": "WAIT", "parameters": {"duration": 5}, "description": "Wait for app to load"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait before focusing"},
        {"action_type": "FOCUS_WINDOW", "parameters": {"title": "{app_name}"}, "description": "Focus {app_name} window"},
    ],
    "search_file_explorer": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+e"}, "description": "Open File Explorer"},
        {"action_type": "WAIT", "parameters": {"duration": 1.5}, "description": "Wait for Explorer"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+f"}, "description": "Focus search box"},
        {"action_type": "WAIT", "parameters": {"duration": 0.5}, "description": "Wait for search box"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{search_target}"}, "description": "Search for: {search_target}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Execute search"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for search results"},
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"target": "{search_target}"}, "description": "Click on first result"},
    ],
    "chrome_with_profile": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
        {"action_type": "WAIT", "parameters": {"duration": 0.5}, "description": "Wait for menu"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "chrome"}, "description": "Type Chrome"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Launch Chrome"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for Chrome"},
        {"action_type": "FOCUS_WINDOW", "parameters": {"title": "Chrome"}, "description": "Focus Chrome window"},
        {"action_type": "WAIT", "parameters": {"duration": 0.5}, "description": "Wait for focus"},
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"profile_name": "{profile_name}"}, "description": "Select profile: {profile_name}"},
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Profile loaded"},
    ],
    "navigate_to_website": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+l"}, "description": "Focus address bar"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{website}"}, "description": "Go to: {website}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Navigate"},
        {"action_type": "WAIT", "parameters": {"duration": 2.5}, "description": "Wait for page load"},
    ],
    "search_on_page": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "/"}, "description": "Focus search"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for results"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{search_query}"}, "description": "Type: {search_query}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Search"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for results"},
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"target": "{search_query}"}, "description": "Click on result"},
    ],
    "whatsapp_open_chat": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+f"}, "description": "New chat"},
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait for search"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{recipient}"}, "description": "Search: {recipient}"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for search results"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Press Tab"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Press Enter"},
    ],
    "type_and_send_message": [
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait before typing"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{message_content}"}, "description": "Type message"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Send message"},
    ],
}


MODEL2_STEP_RULES = {
    "OPEN_APP": STEP_TEMPLATES["open_app_windows"],
    "CLOSE_APP": [{"action_type": "PRESS_KEY", "parameters": {"key": "alt+f4"}, "description": "Close window"}],
    "OPEN_FILE_EXPLORER": [{"action_type": "PRESS_KEY", "parameters": {"key": "win+e"}, "description": "Open File Explorer"}],
    "SEARCH_FILE": [*STEP_TEMPLATES["search_file_explorer"]],
    "OPEN_FOLDER": [{"action_type": "PRESS_KEY", "parameters": {"key": "win+e"}, "description": "Open File Explorer"}, {"action_type": "WAIT", "parameters": {"duration": 1.5}, "description": "Wait for Explorer"}, {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+l"}, "description": "Focus address bar"}, {"action_type": "TYPE_TEXT", "parameters": {"text": "{file_path}"}, "description": "Navigate to folder"}, {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Open folder"}],
    "WEB_SEARCH": [
        *STEP_TEMPLATES["chrome_with_profile"],
        *STEP_TEMPLATES["navigate_to_website"],
        {"action_type": "CONDITIONAL", "parameters": {"condition": "search_query_exists"}, "description": "Check search needed"},
        *STEP_TEMPLATES["search_on_page"],
    ],
    "TYPE_TEXT": [{"action_type": "TYPE_TEXT", "parameters": {"text": "{text_content}"}, "description": "Type: {text_content}"}],
    "FOCUS_WINDOW": [{"action_type": "FOCUS_WINDOW", "parameters": {"title": "{app_name}"}, "description": "Focus: {app_name}"}],
    "MOUSE_CLICK": [{"action_type": "SCREEN_ANALYSIS", "parameters": {"target": "{action_target}"}, "description": "Click: {action_target}"}],
    "MOUSE_RIGHTCLICK": [{"action_type": "MOUSE_RIGHTCLICK", "parameters": {}, "description": "Right click"}],
    "MOUSE_DOUBLECLICK": [{"action_type": "MOUSE_DOUBLECLICK", "parameters": {}, "description": "Double click"}],
    "WINDOW_ACTION": [{"action_type": "PRESS_KEY", "parameters": {"key": "win+up"}, "description": "Window action: {window_action}"}],
    "KEYBOARD": [{"action_type": "PRESS_KEY", "parameters": {"key": "{keyboard_shortcut}"}, "description": "Press: {keyboard_shortcut}"}],
    "SYSTEM": [{"action_type": "SYSTEM_ACTION", "parameters": {"action": "{system_action}"}, "description": "System: {system_action}"}],
    "APP_WITH_ACTION": [
        *STEP_TEMPLATES["open_app_windows"],
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait for app ready"},
        {"action_type": "CONDITIONAL", "parameters": {"condition": "has_search_query"}, "description": "Check action type"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+l"}, "description": "Focus search/input"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{action_content}"}, "description": "Enter: {action_content}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Execute"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for results"},
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"target": "{action_content}"}, "description": "Click on result"},
    ],
    "MEDIA_CONTROL": [
        *STEP_TEMPLATES["open_app_windows"],
        {"action_type": "WAIT", "parameters": {"duration": 4}, "description": "Wait for media app"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+k"}, "description": "Focus search"},
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait for search"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{media_query}"}, "description": "Search: {media_query}"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for search"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Play"},
    ],
    "SEND_MESSAGE": [
        *STEP_TEMPLATES["open_app_windows"],
        {"action_type": "WAIT", "parameters": {"duration": 3}, "description": "Wait for app to load"},
        *STEP_TEMPLATES["whatsapp_open_chat"],
    ],
    "SEND_MESSAGE_PHASE_2": [
        *STEP_TEMPLATES["type_and_send_message"],
    ],
}

class EvaGui:
    def __init__(self, root):
        self.root = root
        self.root.title("EVA - Integrated Logic Assistant")
        self.root.geometry("800x700")

        self.vision_enabled = False
        self.current_steps = []
        self.current_model1_result = None
        self.current_extracted_keywords = None
        self.action_router = None
        self.is_awake = False
        self.is_muted = False

        self.create_widgets()
        self.initialize_backend()

        # Start wake word listener
        wake_word_thread = threading.Thread(target=self.listen_for_wake_word)
        wake_word_thread.daemon = True
        wake_word_thread.start()

    def create_widgets(self):
        self.status_label = tk.Label(self.root, text="Listening for 'Jarvis'...", font=("Arial", 12), pady=10)
        self.status_label.pack()

        self.response_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, bg="#f0f0f0", fg="black")
        self.response_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.response_area.config(state=tk.DISABLED)

        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        self.command_entry = tk.Entry(input_frame, font=("Arial", 12))
        self.command_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.command_entry.bind("<Return>", self.process_text_command)

        self.submit_button = tk.Button(input_frame, text="Submit", command=self.process_text_command)
        self.submit_button.pack(side=tk.RIGHT, padx=5)

    def process_text_command(self, event=None):
        prompt = self.command_entry.get()
        if prompt:
            self.command_entry.delete(0, tk.END)
            self.update_status(f"Recognized: {prompt}")
            thread = threading.Thread(target=self.run_eva_pipeline, args=(prompt,))
            thread.start()

    def initialize_backend(self):
        try:
            self.update_response_area("Initializing backend components...\n")
            self.executor_bridge = ExecutorBridge()
            self.system_executor = SystemExecutor(self.executor_bridge)
            self.screenshot_handler = ScreenshotHandler()
            self.update_response_area("✓ Execution engine loaded.\n")

            self.screen_analyzer = ScreenAnalyzer(config.GEMINI_API_KEY)
            self.omniparser = OmniParserExecutor()
            self.action_router = ActionRouter(self.system_executor, self.screenshot_handler, self.screen_analyzer, self.omniparser)
            self.vision_enabled = True
            self.update_response_area("✓ Vision system loaded successfully.\n")
            
            self.wake_word_detector = WakeWordDetector()
            self.update_response_area("✓ Wake word detector loaded successfully.\n")

            # Train command classifier
            self.vectorizer = TfidfVectorizer()
            self.classifier = LogisticRegression()
            X, y = zip(*MODEL1_TRAINING_DATA)
            X_vectorized = self.vectorizer.fit_transform(X)
            self.classifier.fit(X_vectorized, y)
            self.update_response_area("✓ Command classifier trained.\n")

            self.update_response_area("Ready to receive commands.\n")
        except Exception as e:
            error_message = f"❌ CRITICAL ERROR: Could not initialize backend.\n{e}\nVision features will be disabled.\nCheck your .env file for GEMINI_API_KEY and ensure all model weights are downloaded."
            self.update_response_area(error_message)
    def listen_for_wake_word(self):
        self.wake_word_detector.start()
        while True:
            if not self.is_awake:
                self.root.after_idle(self.update_status, "Listening for 'Jarvis'...")
                if self.wake_word_detector.listen():
                    self.is_awake = True
                    self.root.after_idle(self.update_status, "Wake word detected! Listening for command...")
            else:
                if self.is_muted:
                    self.root.after_idle(self.update_status, "Muted. Say 'unmute' to resume.")
                    # Still listen for 'unmute'
                    recognizer = sr.Recognizer()
                    with sr.Microphone() as source:
                        try:
                            audio = recognizer.listen(source, timeout=5, phrase_time_limit=2)
                            command = recognizer.recognize_google(audio).lower()
                            if "unmute" in command:
                                self.is_muted = False
                                self.root.after_idle(self.update_status, "Unmuted. Listening for command...")
                        except (sr.UnknownValueError, sr.RequestError):
                            pass # Ignore errors when muted
                    time.sleep(1) # Avoid busy-waiting
                    continue

                self.root.after_idle(self.update_status, "Listening for command...")
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    try:
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        prompt = recognizer.recognize_google(audio)
                        self.root.after_idle(self.update_status, f"Recognized: {prompt}")

                        if "go to sleep" in prompt.lower():
                            self.is_awake = False
                            self.root.after_idle(self.update_status, "Going to sleep. Listening for 'Jarvis'...")
                            continue
                        
                        if "mute" in prompt.lower():
                            self.is_muted = True
                            self.root.after_idle(self.update_status, "Muted.")
                            continue

                        # Run pipeline in a new thread to keep GUI responsive
                        thread = threading.Thread(target=self.run_eva_pipeline, args=(prompt,))
                        thread.start()
                    except sr.UnknownValueError:
                        self.root.after_idle(self.update_status, "Could not understand audio. Listening for command...")
                    except sr.RequestError as e:
                        self.root.after_idle(self.update_status, f"Could not request results; {e}. Listening for command...")
                    except Exception as e:
                        self.root.after_idle(self.update_status, f"An error occurred: {e}. Listening for command...")

    def run_eva_pipeline(self, prompt):
        self.root.after_idle(self.clear_response_area)
        self.root.after_idle(self.update_response_area, f"Processing command: \"{prompt}\"\n\n")

        self.current_model1_result = self.analyze_query_with_model(prompt)
        if not self.current_model1_result:
            self.root.after_idle(self.update_response_area, "⚠️ Error analyzing command!")
            return

        self.root.after_idle(self.display_classification_results)

        self.current_extracted_keywords = self.extract_keywords_by_command_type(self.current_model1_result['input'], self.current_model1_result['command_type'])
        self.root.after_idle(self.display_keyword_results)

        command_type = self.current_model1_result['command_type']
        
        if command_type == "SEND_MESSAGE":
            self.handle_interactive_messaging()
        else:
            self.current_steps = self.generate_steps_model2(command_type, self.current_extracted_keywords)
            self.root.after_idle(self.display_step_results)
            self.execute_steps() # Automatic execution

    def handle_interactive_messaging(self):
        if self.current_extracted_keywords.get('message_content'):
            # Message is already present, execute all steps at once
            self.root.after_idle(self.update_response_area, "STEP 3: Generated Steps\n" + "-"*40 + "\n")
            steps = self.generate_steps_model2("SEND_MESSAGE", self.current_extracted_keywords)
            steps.extend(self.generate_steps_model2("SEND_MESSAGE_PHASE_2", self.current_extracted_keywords))
            self.root.after_idle(self.display_steps, steps)
            self.current_steps = steps
            self.execute_steps()
        else:
            # Message not present, ask for it
            # Phase 1: Open the chat
            self.root.after_idle(self.update_response_area, "STEP 3: Generated Steps (Phase 1: Open Chat)\n" + "-"*40 + "\n")
            steps_phase1 = self.generate_steps_model2("SEND_MESSAGE", self.current_extracted_keywords)
            self.root.after_idle(self.display_steps, steps_phase1)
            
            # Execute phase 1
            self.current_steps = steps_phase1
            self.execute_steps()

            # Phase 2: Get message content via voice
            self.root.after_idle(self.update_status, "What message do you want to send?")
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    audio = recognizer.listen(source, timeout=7, phrase_time_limit=15)
                    message = recognizer.recognize_google(audio)
                    self.root.after_idle(self.update_status, f"Message: {message}")
                    self.current_extracted_keywords['message_content'] = message
                    
                    # Generate and execute phase 2
                    self.root.after_idle(self.update_response_area, "\nSTEP 4: Generated Steps (Phase 2: Send Message)\n" + "-"*40 + "\n")
                    steps_phase2 = self.generate_steps_model2("SEND_MESSAGE_PHASE_2", self.current_extracted_keywords)
                    self.root.after_idle(self.display_steps, steps_phase2, len(self.current_steps) + 1)
                    
                    self.current_steps = steps_phase2
                    self.execute_steps()

                except sr.UnknownValueError:
                    self.root.after_idle(self.update_status, "Could not understand message. Cancelling.")
                except sr.RequestError as e:
                    self.root.after_idle(self.update_status, f"Could not request results; {e}. Cancelling.")

    def execute_steps(self):
        if not self.current_steps or not self.action_router:
            return
        self.root.after_idle(self.update_response_area, "\nEXECUTING STEPS...\n" + "-"*40 + "\n")
        
        # Run execution in a separate thread to avoid blocking the GUI
        thread = threading.Thread(target=self._execute_action_router)
        thread.start()

    def _execute_action_router(self):
        result = self.action_router.execute(
            self.current_model1_result['command_type'],
            self.current_steps,
            self.current_extracted_keywords,
            self.current_model1_result['input'],
            self.current_model1_result
        )
        self.root.after_idle(self.display_execution_result, result)

    def display_classification_results(self):
        self.update_response_area("STEP 1: Command Classification\n" + "-"*40 + "\n")
        self.update_response_area(f"Input: \"{self.current_model1_result['input']}\"\n")
        self.update_response_area(f"Type: {self.current_model1_result['command_type']}\n\n")

    def display_keyword_results(self):
        self.update_response_area("STEP 2: Keyword Extraction\n" + "-"*40 + "\n")
        for key, value in self.current_extracted_keywords.items():
            if value:
                self.update_response_area(f" • {key.replace('_', ' ').title()}: {value}\n")
        self.update_response_area("\n")

    def display_step_results(self):
        self.update_response_area("STEP 3: Generated Steps\n" + "-"*40 + "\n")
        self.display_steps(self.current_steps)

    def display_steps(self, steps, start_index=1):
        step_count = start_index - 1
        for step in steps:
            if step['action_type'] == "CONDITIONAL":
                continue
            step_count += 1
            self.update_response_area(f"{step_count}. {step['description']}\n")
            self.update_response_area(f"   Action: {step['action_type']}\n")
            if step['parameters']:
                for k, v in step['parameters'].items():
                    if v:
                        self.update_response_area(f"   {k}: {v}\n")
            self.update_response_area("\n")

    def display_execution_result(self, result):
        if result.get('success'):
            self.update_response_area("✅ Command executed successfully!\n")
        else:
            self.update_response_area(f"❌ ERROR: {result.get('error', 'Unknown error')}\n")

    def update_response_area(self, text):
        self.response_area.config(state=tk.NORMAL)
        self.response_area.insert(tk.END, text)
        self.response_area.see(tk.END)
        self.response_area.config(state=tk.DISABLED)
    
    def clear_response_area(self):
        self.response_area.config(state=tk.NORMAL)
        self.response_area.delete(1.0, tk.END)
        self.response_area.config(state=tk.DISABLED)

    def update_status(self, text):
        self.status_label.config(text=text)

    def analyze_query_with_model(self, query):
        try:
            query_vectorized = self.vectorizer.transform([query.lower()])
            prediction = self.classifier.predict(query_vectorized)
            confidence = self.classifier.predict_proba(query_vectorized).max()
            return {
                "input": query,
                "command_type": prediction[0],
                "confidence": confidence,
                "training_pattern": "Local Model Analysis",
            }
        except Exception as e:
            print(f"Error analyzing query with model: {e}")
            return None



    def extract_keywords_by_command_type(self, raw_command, command_type):
        raw_command_lower = raw_command.lower().strip()
        words = raw_command_lower.split()
        extracted = {
            'app_name': None, 'search_query': None, 'text_content': None, 'action_target': None, 'keyboard_shortcut': None,
            'system_action': None, 'window_action': None, 'profile_name': None, 'website': None, 'media_query': None,
            'recipient': None, 'message_content': None, 'action_content': None, 'is_file_operation': False, 'file_path': None,
            'target_type': None, 'is_known_folder': False, 'needs_search': False, 'search_target': None, 'has_message_content': False,
        }
        if command_type == "OPEN_APP":
            trigger = ['open', 'launch', 'start', 'run']
            extracted['app_name'] = self.extract_app_name(words, trigger)
        elif command_type == "CLOSE_APP":
            trigger = ['close', 'exit', 'quit']
            extracted['app_name'] = self.extract_app_name(words, trigger) or 'current'
        elif command_type == "OPEN_FOLDER":
            is_file_op, target_name, target_type, is_known = self.extract_file_or_folder_path(words, raw_command_lower)
            if is_known:
                extracted['file_path'] = target_name
            else:
                extracted['search_target'] = target_name
        elif command_type == "SEARCH_FILE":
            is_file_op, target_name, target_type, is_known = self.extract_file_or_folder_path(words, raw_command_lower)
            extracted['search_target'] = target_name
        elif command_type == "WEB_SEARCH":
            extracted['profile_name'] = self.extract_profile_name(raw_command_lower)
            website, query = self.extract_website_and_action(raw_command_lower)
            extracted['website'] = website
            extracted['search_query'] = query
        elif command_type == "TYPE_TEXT":
            extracted['text_content'] = self.extract_text_after_keywords(raw_command.split(), ['type', 'write', 'enter'], {'text', 'message'})
        elif command_type in ["MOUSE_CLICK", "MOUSE_RIGHTCLICK", "MOUSE_DOUBLECLICK"]:
            skip = {'click', 'on', 'here', 'it', 'this', 'right', 'double'}
            extracted['action_target'] = ' '.join([w for w in raw_command.split() if w.lower() not in skip]) or 'current'
        elif command_type == "WINDOW_ACTION":
            extracted['window_action'] = 'maximize' if any(w in raw_command.lower().split() for w in ['maximize', 'fullscreen']) else 'minimize'
        elif command_type == "KEYBOARD":
            shortcuts = {'copy': 'ctrl+c', 'paste': 'ctrl+v', 'save': 'ctrl+s', 'undo': 'ctrl+z'}
            for word, shortcut in shortcuts.items():
                if word in raw_command.lower():
                    extracted['keyboard_shortcut'] = shortcut
                    break
        elif command_type == "SYSTEM":
            extracted['system_action'] = 'screenshot' if 'screenshot' in raw_command.lower() or 'capture' in raw_command.lower() else 'lock'
        elif command_type == "APP_WITH_ACTION":
            if 'and' in raw_command.lower():
                and_idx = raw_command.lower().split().index('and')
                extracted['app_name'] = self.extract_app_name(raw_command.split()[:and_idx], ['open', 'launch', 'start'])
                extracted['action_content'] = ' '.join([w for w in raw_command.split()[and_idx+1:] if w.lower() not in ['search', 'type', 'play']])
        elif command_type == "MEDIA_CONTROL":
            apps = ['spotify', 'netflix', 'youtube', 'vlc']
            app_name = next((app for app in apps if app in raw_command.lower()), 'spotify')
            extracted['app_name'] = app_name
            play_idx = -1
            if 'play' in raw_command.lower():
                play_idx = raw_command.lower().split().index('play')
            elif 'stream' in raw_command.lower():
                play_idx = raw_command.lower().split().index('stream')
            if play_idx != -1:
                extracted['media_query'] = ' '.join(raw_command.split()[play_idx+1:])
        elif command_type == "SEND_MESSAGE":
            apps_map = {
                'whatsapp': 'whatsapp', 'email': 'outlook', 'social': 'facebook', 'twitter': 'twitter',
                'instagram': 'instagram', 'telegram': 'telegram',
            }
            extracted['app_name'] = next((v for k, v in apps_map.items() if k in raw_command.lower()), 'whatsapp')

            # Pattern 1: send {message} to {recipient}
            match = re.search(r'send\s+(.*?)\s+to\s+(.*)', raw_command, re.IGNORECASE)
            if match:
                extracted['message_content'] = match.group(1)
                extracted['recipient'] = match.group(2)
                return extracted

            # Pattern 2: to {recipient} message {message}
            match = re.search(r'to\s+(.*?)\s+(?:message|saying|that)\s+(.*)', raw_command, re.IGNORECASE)
            if match:
                extracted['recipient'] = match.group(1)
                extracted['message_content'] = match.group(2)
                return extracted

            # Pattern 3: to {recipient} {message} (single word recipient)
            match = re.search(r'to\s+(\w+)\s+(.*)', raw_command, re.IGNORECASE)
            if match:
                extracted['recipient'] = match.group(1)
                extracted['message_content'] = match.group(2)
                return extracted

            # Fallback: to {recipient}
            if 'to' in raw_command.lower():
                to_idx = raw_command.lower().split().index('to')
                extracted['recipient'] = ' '.join(raw_command.split()[to_idx + 1:])

        return extracted

    def generate_steps_model2(self, command_type, extracted_keywords):
        if command_type not in MODEL2_STEP_RULES:
            return [{"action_type": "EXECUTE", "parameters": {}, "description": f"Execute: {command_type}"}]
        steps_template = MODEL2_STEP_RULES[command_type]
        generated_steps = []
        for step in steps_template:
            if step.get("action_type") == "CONDITIONAL":
                condition = step["parameters"].get("condition")
                if condition == "search_query_exists":
                    if not extracted_keywords.get('search_query'):
                        break
                    else:
                        continue
                if condition == "has_search_query":
                    if not extracted_keywords.get('action_content'):
                        break
                    else:
                        continue
                if condition == "has_message_content":
                    if not extracted_keywords.get('message_content'):
                        break
                    else:
                        continue
                continue
            step_copy = {"action_type": step["action_type"], "parameters": dict(step["parameters"]), "description": step["description"]}
            replacements = {
                "{app_name}": extracted_keywords.get('app_name', 'app'), "{website}": extracted_keywords.get('website', ''),
                "{profile_name}": extracted_keywords.get('profile_name', 'Default'), "{search_query}": extracted_keywords.get('search_query', ''),
                "{text_content}": extracted_keywords.get('text_content', ''), "{action_target}": extracted_keywords.get('action_target', 'target'),
                "{keyboard_shortcut}": extracted_keywords.get('keyboard_shortcut', ''), "{system_action}": extracted_keywords.get('system_action', ''),
                "{window_action}": extracted_keywords.get('window_action', ''), "{media_query}": extracted_keywords.get('media_query', ''),
                "{recipient}": extracted_keywords.get('recipient', ''), "{message_content}": extracted_keywords.get('message_content', ''),
                "{action_content}": extracted_keywords.get('action_content', ''), "{file_path}": extracted_keywords.get('file_path', ''),
            }
            for key, value in step_copy["parameters"].items():
                if isinstance(value, str):
                    for placeholder, replacement in replacements.items():
                        value = value.replace(placeholder, str(replacement or ''))
                    step_copy["parameters"][key] = value
            for placeholder, replacement in replacements.items():
                step_copy["description"] = step_copy["description"].replace(placeholder, str(replacement or ''))
            generated_steps.append(step_copy)
        return generated_steps

    def extract_profile_name(self, text):
        patterns = [
            r'with chrome profile ([\w\s]+?)(?:\s+(?:search|open|go|and))', r'chrome profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
            r'with profile ([\w\s]+?)(?:\s+(?:search|open|go|and))', r'use profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
            r'profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Default"

    def extract_website_and_action(self, text):
        websites = {
            'youtube': 'youtube.com', 'google': 'google.com', 'gmail': 'mail.google.com', 'facebook': 'facebook.com',
            'twitter': 'twitter.com', 'instagram': 'instagram.com', 'linkedin': 'linkedin.com', 'github': 'github.com',
            'reddit': 'reddit.com', 'amazon': 'amazon.com', 'netflix': 'netflix.com', 'spotify': 'open.spotify.com',
        }
        text_l = text.lower()
        website = None
        for keyword, url in websites.items():
            if keyword in text_l:
                website = url
                break
        query_text = text_l
        for pattern in [r'with chrome profile [\w\s]+', r'chrome profile [\w\s]+', r'profile [\w\s]+']:
            query_text = re.sub(pattern, '', query_text)
        skip_words = {'with', 'chrome', 'search', 'for', 'open', 'go', 'to', 'on', 'in', 'and', 'youtube', 'google', 'gmail', 'facebook', 'profile'}
        query_words = [w for w in query_text.split() if w not in skip_words and w.strip()]
        return website, ' '.join(query_words) if query_words else None

    def extract_app_name(self, words, trigger_words):
        for trigger in trigger_words:
            if trigger in words:
                idx = words.index(trigger)
                app_words = [w for w in words[idx+1:] if w not in ['app', 'application', 'program']]
                if app_words:
                    return ' '.join(app_words)
        return None

    def extract_text_after_keywords(self, words, keywords, skip_words):
        for keyword in keywords:
            if keyword in words:
                idx = words.index(keyword)
                text_words = [w for w in words[idx+1:] if w not in skip_words]
                if text_words:
                    return ' '.join(text_words)
        return None

    def extract_file_or_folder_path(self, words, raw_command):
        common_folders = {
            'documents': r'%USERPROFILE%\Documents', 'downloads': r'%USERPROFILE%\Downloads', 'desktop': r'%USERPROFILE%\Desktop',
            'pictures': r'%USERPROFILE%\Pictures', 'videos': r'%USERPROFILE%\Videos', 'music': r'%USERPROFILE%\Music',
        }
        for folder_keyword, folder_path in common_folders.items():
            if folder_keyword in words:
                return True, folder_path, 'folder', True
        skip_words = {'open', 'file', 'folder', 'document', 'my', 'the', 'launch', 'show', 'browse', 'to', 'for', 'find'}
        target_words = [w for w in words if w not in skip_words]
        if target_words:
            target_name = ' '.join(target_words)
            target_type = 'folder' if 'folder' in words or 'directory' in words else 'file'
            return True, target_name, target_type, False
        return False, None, None, False

if __name__ == "__main__":
    root = tk.Tk()
    app = EvaGui(root)
    root.mainloop()
