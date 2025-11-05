import tkinter as tk
from tkinter import scrolledtext
import re
import threading
import speech_recognition as sr
import pvporcupine
import pyaudio
import struct
from groq import Groq
import os
from dotenv import load_dotenv
from execution.execution_handler import ExecutionHandler

# Load environment variables from .env file
load_dotenv()

# ============================================================================ 
# LOGIC FROM EVA_TER.py
# ============================================================================ 

# ... (rest of the logic from EVA_TER.py remains the same) ...
MODEL1_TRAINING_DATA = [
    ("open application", "OPEN_APP"), ("launch program", "OPEN_APP"), ("start software", "OPEN_APP"),
    ("run app", "OPEN_APP"), ("open app", "OPEN_APP"), ("open chrome", "OPEN_APP"), ("launch spotify", "OPEN_APP"),
    ("close application", "CLOSE_APP"), ("close this", "CLOSE_APP"), ("close window", "CLOSE_APP"),
    ("exit application", "CLOSE_APP"), ("quit app", "CLOSE_APP"),
    ("open file", "FILE_FOLDER_OPERATION"), ("open folder", "FILE_FOLDER_OPERATION"), ("open document", "FILE_FOLDER_OPERATION"),
    ("launch file", "FILE_FOLDER_OPERATION"), ("open my documents", "FILE_FOLDER_OPERATION"), ("open downloads folder", "FILE_FOLDER_OPERATION"),
    ("open desktop", "FILE_FOLDER_OPERATION"), ("show file", "FILE_FOLDER_OPERATION"), ("browse to folder", "FILE_FOLDER_OPERATION"),
    ("open pictures", "FILE_FOLDER_OPERATION"), ("open videos folder", "FILE_FOLDER_OPERATION"), ("open music", "FILE_FOLDER_OPERATION"),
    ("show folder", "FILE_FOLDER_OPERATION"), ("browse file", "FILE_FOLDER_OPERATION"),
    ("type text", "TYPE_TEXT"), ("write something", "TYPE_TEXT"), ("enter text", "TYPE_TEXT"),
    ("click on something", "MOUSE_CLICK"), ("click here", "MOUSE_CLICK"),
    ("right click", "MOUSE_RIGHTCLICK"), ("double click", "MOUSE_DOUBLECLICK"),
    ("maximize window", "WINDOW_ACTION"), ("minimize window", "WINDOW_ACTION"), ("fullscreen mode", "WINDOW_ACTION"),
    ("take screenshot", "SYSTEM"), ("lock screen", "SYSTEM"),
    ("copy", "KEYBOARD"), ("paste", "KEYBOARD"), ("save", "KEYBOARD"), ("undo", "KEYBOARD"),
    ("open app and search", "APP_WITH_ACTION"), ("launch app and type", "APP_WITH_ACTION"),
    ("open app and play", "APP_WITH_ACTION"), ("start app and compose", "APP_WITH_ACTION"),
    ("play music", "MEDIA_CONTROL"), ("play video", "MEDIA_CONTROL"), ("stream music", "MEDIA_CONTROL"), ("stream video", "MEDIA_CONTROL"),
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
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for app to load"},
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
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{search_query}"}, "description": "Type: {search_query}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Search"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for results"},
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"target": "{search_query}"}, "description": "Click on result"},
    ],
    "whatsapp_open_chat": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+n"}, "description": "New chat"},
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait for search"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{recipient}"}, "description": "Search: {recipient}"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for search results"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Press Tab"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Press Tab"},
    ],
    "type_and_send_message": [
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{message_content}"}, "description": "Type message"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Send message"},
    ],
}

MODEL2_STEP_RULES = {
    "OPEN_APP": [
        {"action_type": "OPEN_APP", "parameters": {"app_name": "{app_name}"}, "description": "Open {app_name}"},
    ],
    "CLOSE_APP": [{"action_type": "PRESS_KEY", "parameters": {"key": "alt+f4"}, "description": "Close window"}],
    "FILE_FOLDER_OPERATION": [*STEP_TEMPLATES["search_file_explorer"]],
    "WEB_SEARCH": [
        {"action_type": "OPEN_URL", "parameters": {"url": "https://{website}{search_path}"}, "description": "Open {website} with search query"},
    ],
    "TYPE_TEXT": [{"action_type": "TYPE_TEXT", "parameters": {"text": "{text_content}"}, "description": "Type: {text_content}"}],
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
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for media app"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+l"}, "description": "Focus search"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{media_query}"}, "description": "Search: {media_query}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Play"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for results"},
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"target": "{media_query}"}, "description": "Click on media"},
    ],
    "SEND_MESSAGE": [
        *STEP_TEMPLATES["open_app_windows"],
        {"action_type": "WAIT", "parameters": {"duration": 3}, "description": "Wait for app to load"},
        *STEP_TEMPLATES["whatsapp_open_chat"],
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"target": "{recipient}"}, "description": "Click on contact"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Confirm contact"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for chat to open"},
        {"action_type": "CONDITIONAL", "parameters": {"condition": "has_message_content"}, "description": "Check if message provided"},
        *STEP_TEMPLATES["type_and_send_message"],
    ],
}

def extract_profile_name(text):
    patterns = [
        r'with chrome profile ([\w\s]+?)(?:\s+(?:search|open|go|and))', r'chrome profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'with profile ([\w\s]+?)(?:\s+(?:search|open|go|and))', r'use profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()
    return "Default"

def extract_website_and_action(text):
    websites = {
        'youtube': 'youtube.com', 'google': 'google.com', 'gmail': 'mail.google.com', 'facebook': 'facebook.com',
        'twitter': 'twitter.com', 'instagram': 'instagram.com', 'linkedin': 'linkedin.com', 'github': 'github.com',
        'reddit': 'reddit.com', 'amazon': 'amazon.com', 'netflix': 'netflix.com', 'spotify': 'open.spotify.com',
    }
    text_l = text.lower()
    website = 'google.com'
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

def extract_app_name(words, trigger_words):
    for trigger in trigger_words:
        if trigger in words:
            idx = words.index(trigger)
            app_words = [w for w in words[idx+1:] if w not in ['app', 'application', 'program']]
            if app_words:
                return ' '.join(app_words)
    return None

def extract_text_after_keywords(words, keywords, skip_words):
    for keyword in keywords:
        if keyword in words:
            idx = words.index(keyword)
            text_words = [w for w in words[idx+1:] if w not in skip_words]
            if text_words:
                return ' '.join(text_words)
    return None

def extract_file_or_folder_path(words, raw_command):
    file_indicators = ['file', 'document', 'doc', 'pdf', 'image', 'video', 'folder', 'directory']
    common_folders = {
        'documents': r'%USERPROFILE%\Documents', 'downloads': r'%USERPROFILE%\Downloads', 'desktop': r'%USERPROFILE%\Desktop',
        'pictures': r'%USERPROFILE%\Pictures', 'videos': r'%USERPROFILE%\Videos', 'music': r'%USERPROFILE%\Music',
    }
    is_file_operation = any(indicator in words for indicator in file_indicators)
    if not is_file_operation:
        return False, None, None, False
    for folder_keyword, folder_path in common_folders.items():
        if folder_keyword in words:
            return True, folder_path, 'folder', True
    skip_words = {'open', 'file', 'folder', 'document', 'my', 'the', 'launch', 'show', 'browse', 'to'}
    target_words = [w for w in words if w not in skip_words]
    if target_words:
        target_name = ' '.join(target_words)
        target_type = 'folder' if 'folder' in words or 'directory' in words else 'file'
        return True, target_name, target_type, False
    return False, None, None, False

def extract_keywords_by_command_type(raw_command, command_type):
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
        extracted['app_name'] = extract_app_name(words, trigger) or ('chrome' if 'chrome' in words else 'current')
    elif command_type == "CLOSE_APP":
        trigger = ['close', 'exit', 'quit']
        extracted['app_name'] = extract_app_name(words, trigger) or 'current'
    elif command_type == "FILE_FOLDER_OPERATION":
        is_file_op, target_name, target_type, is_known = extract_file_or_folder_path(words, raw_command_lower)
        extracted['is_file_operation'] = True
        extracted['is_known_folder'] = is_known
        extracted['needs_search'] = not is_known
        extracted['target_type'] = target_type
        if is_known:
            extracted['file_path'] = target_name
        else:
            extracted['search_target'] = target_name
    elif command_type == "WEB_SEARCH":
        extracted['profile_name'] = extract_profile_name(raw_command_lower)
        website, query = extract_website_and_action(raw_command_lower)
        extracted['website'] = website
        extracted['search_query'] = query
    elif command_type == "TYPE_TEXT":
        extracted['text_content'] = extract_text_after_keywords(words, ['type', 'write', 'enter'], {'text', 'message'})
    elif command_type in ["MOUSE_CLICK", "MOUSE_RIGHTCLICK", "MOUSE_DOUBLECLICK"]:
        skip = {'click', 'on', 'here', 'it', 'this', 'right', 'double'}
        extracted['action_target'] = ' '.join([w for w in words if w not in skip]) or 'current'
    elif command_type == "WINDOW_ACTION":
        extracted['window_action'] = 'maximize' if any(w in words for w in ['maximize', 'fullscreen']) else 'minimize'
    elif command_type == "KEYBOARD":
        shortcuts = {'copy': 'ctrl+c', 'paste': 'ctrl+v', 'save': 'ctrl+s', 'undo': 'ctrl+z'}
        for word, shortcut in shortcuts.items():
            if word in words:
                extracted['keyboard_shortcut'] = shortcut
                break
    elif command_type == "SYSTEM":
        extracted['system_action'] = 'screenshot' if 'screenshot' in words or 'capture' in words else 'lock'
    elif command_type == "APP_WITH_ACTION":
        if 'and' in words:
            and_idx = words.index('and')
            extracted['app_name'] = extract_app_name(words[:and_idx], ['open', 'launch', 'start'])
            extracted['action_content'] = ' '.join([w for w in words[and_idx+1:] if w not in ['search', 'type', 'play']])
    elif command_type == "MEDIA_CONTROL":
        apps = ['spotify', 'netflix', 'youtube', 'vlc']
        extracted['app_name'] = next((app for app in apps if app in words), 'spotify')
        extracted['media_query'] = ' '.join([w for w in words if w not in ['play', 'stream', 'music', 'video', extracted['app_name']]])
    elif command_type == "SEND_MESSAGE":
        apps_map = {
            'whatsapp': 'whatsapp', 'email': 'outlook', 'social': 'facebook', 'twitter': 'twitter',
            'instagram': 'instagram', 'telegram': 'telegram',
        }
        extracted['app_name'] = next((v for k, v in apps_map.items() if k in raw_command_lower), 'whatsapp')
        if 'to' in words:
            to_idx = words.index('to')
            recipient_words = words[to_idx + 1:]
            if recipient_words:
                extracted['recipient'] = ' '.join(recipient_words)
    return extracted

def generate_steps_model2(command_type, extracted_keywords):
    if command_type not in MODEL2_STEP_RULES:
        return [{"action_type": "EXECUTE", "parameters": {}, "description": f"Execute: {command_type}"}]
    steps_template = MODEL2_STEP_RULES[command_type]
    generated_steps = []
    for step in steps_template:
        if step.get("action_type") == "CONDITIONAL":
            condition = step["parameters"].get("condition")
            if condition == "is_known_folder":
                if not extracted_keywords.get('is_known_folder'):
                    continue
                else:
                    continue
            if condition == "needs_search":
                if not extracted_keywords.get('needs_search'):
                    break
                else:
                    continue
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

        # Handle OPEN_URL specific logic
        if step_copy["action_type"] == "OPEN_URL":
            website = extracted_keywords.get('website', 'google.com')
            search_query = extracted_keywords.get('search_query')
            search_path = f"/search?q={search_query}" if search_query else ""
            step_copy["parameters"]['url'] = f"https://{website}{search_path}"

        replacements = {
            "{app_name}": extracted_keywords.get('app_name', 'app'), "{website}": extracted_keywords.get('website', 'google.com'),
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

def analyze_query_with_groq(query):
    """
    Analyzes the user's query using the Groq API to determine the command type.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a command classifier. Classify the user's command into one of the following categories: OPEN_APP, CLOSE_APP, FILE_FOLDER_OPERATION, WEB_SEARCH, TYPE_TEXT, MOUSE_CLICK, MOUSE_RIGHTCLICK, MOUSE_DOUBLECLICK, WINDOW_ACTION, KEYBOARD, SYSTEM, APP_WITH_ACTION, MEDIA_CONTROL, SEND_MESSAGE. Respond with only the command type.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            model="llama3-8b-8192",
        )
        command_type = chat_completion.choices[0].message.content.strip()
        return {
            "input": query,
            "command_type": command_type,
            "confidence": 1.0,  # Groq doesn't provide a confidence score, so we'll use 1.0
            "training_pattern": "Groq Analysis",
        }
    except Exception as e:
        print(f"Error analyzing query with Groq: {e}")
        return None

# ============================================================================ 
# GUI SPECIFIC LOGIC
# ============================================================================ 

def run_eva_pipeline(prompt, response_widget, execution_handler):
    """
    This function runs the EVA pipeline and displays the output in the GUI.
    """
    response_widget.config(state=tk.NORMAL)
    response_widget.delete(1.0, tk.END) # Clear previous output

    # STEP 1: Model 1 - Classification (using Groq)
    model1_result = analyze_query_with_groq(prompt)
    if not model1_result:
        response_widget.insert(tk.END, "⚠️ Error analyzing your command with Groq!")
        response_widget.config(state=tk.DISABLED)
        return

    response_widget.insert(tk.END, "STEP 1: Command Classification (Groq)\n")
    response_widget.insert(tk.END, "------------------------------------\n")
    response_widget.insert(tk.END, f"Input: \"{model1_result['input']}\"\n")
    response_widget.insert(tk.END, f"Type: {model1_result['command_type']}\n\n")

    # STEP 2: Keyword Extraction
    extracted_keywords = extract_keywords_by_command_type(model1_result['input'], model1_result['command_type'])
    response_widget.insert(tk.END, "STEP 2: Keyword Extraction\n")
    response_widget.insert(tk.END, "------------------------------------\n")
    for key, value in extracted_keywords.items():
        if value:
            response_widget.insert(tk.END, f" • {key.replace('_', ' ').title()}: {value}\n")
    response_widget.insert(tk.END, "\n")


    # STEP 3: Step Generation
    steps = generate_steps_model2(model1_result['command_type'], extracted_keywords)
    response_widget.insert(tk.END, "STEP 3: Generated Steps\n")
    response_widget.insert(tk.END, "------------------------------------\n")
    step_count = 0
    for step in steps:
        if step['action_type'] == "CONDITIONAL":
            continue
        step_count += 1
        response_widget.insert(tk.END, f"{step_count}. {step['description']}\n")
        response_widget.insert(tk.END, f"   Action: {step['action_type']}\n")
        if step['parameters']:
            for k, v in step['parameters'].items():
                if v:
                    response_widget.insert(tk.END, f"   {k}: {v}\n")
        response_widget.insert(tk.END, "\n")

    response_widget.config(state=tk.DISABLED)

    # STEP 4: Automatic Execution
    execution_handler.execute_steps(steps)


def create_gui():
    """
    This function creates and runs the main GUI window.
    """
    window = tk.Tk()
    window.title("EVA - Integrated Logic Assistant")
    window.geometry("800x600")

    status_label = tk.Label(window, text="Listening for 'Jarvis'...", font=("Arial", 12), pady=10)
    status_label.pack()

    response_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=100, height=30, bg="#f0f0f0", fg="black")
    response_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    response_area.config(state=tk.DISABLED)

    execution_handler = ExecutionHandler()

    def listen_for_wake_word():
        porcupine = None
        pa = None
        audio_stream = None
        try:
            # Replace with your Picovoice Access Key
            access_key = os.environ.get("PICOVOICE_ACCESS_KEY") 
            porcupine = pvporcupine.create(access_key=access_key, keywords=["jarvis"])
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length,
            )

            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    status_label.config(text="Wake word detected! Listening for command...")
                    recognizer = sr.Recognizer()
                    with sr.Microphone() as source:
                        try:
                            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                            prompt = recognizer.recognize_google(audio)
                            status_label.config(text=f"Recognized: {prompt}")
                            thread = threading.Thread(target=run_eva_pipeline, args=(prompt, response_area, execution_handler))
                            thread.start()
                        except sr.UnknownValueError:
                            status_label.config(text="Could not understand audio. Listening for 'Jarvis'...")
                        except sr.RequestError as e:
                            status_label.config(text=f"Could not request results; {e}. Listening for 'Jarvis'...")
                        except Exception as e:
                            status_label.config(text=f"An error occurred: {e}. Listening for 'Jarvis'...")
                    status_label.config(text="Listening for 'Jarvis'...")


        finally:
            if audio_stream is not None:
                audio_stream.close()
            if pa is not None:
                pa.terminate()
            if porcupine is not None:
                porcupine.delete()

    wake_word_thread = threading.Thread(target=listen_for_wake_word)
    wake_word_thread.daemon = True
    wake_word_thread.start()

    window.mainloop()

if __name__ == "__main__":
    create_gui()