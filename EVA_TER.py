#!/usr/bin/env python3

"""
EVA Model Testing Platform - Terminal Version (REFACTORED & OPTIMIZED)
Enhanced Virtual Assistant - Command Processing & Step Generation
Automatic workflow: Raw Input → Model 1 → Keyword Extraction → Model 2 → Steps

OPTIMIZATIONS:
- Consolidated redundant command types
- Reusable step templates (open_app, launch_browser_with_profile)
- Simplified keyword extraction with helper functions
- All functionality preserved
"""

import os
import sys
import re

# Global variables
DEFAULT_CHROME_PROFILE = None
test_count = 0

def set_default_chrome_profile():
    """Set default Chrome profile once at startup"""
    global DEFAULT_CHROME_PROFILE
    print("\n🌐 CHROME PROFILE SETUP")
    print("-" * 80)
    profile_input = input("Enter your default Chrome profile name (or press Enter for 'Default'): ").strip()
    DEFAULT_CHROME_PROFILE = profile_input if profile_input else "Default"
    print(f"✅ Default Chrome profile set to: {DEFAULT_CHROME_PROFILE}")
    print("-" * 80)

# ============================================================================
# MODEL 1: TRAINING DATA (unchanged - keeping all patterns)
# ============================================================================

MODEL1_TRAINING_DATA = [
        # App launching/closing (kept separate)
    ("open application", "OPEN_APP"),
    ("launch program", "OPEN_APP"),
    ("start software", "OPEN_APP"),
    ("run app", "OPEN_APP"),
    ("open app", "OPEN_APP"),
    ("open chrome", "OPEN_APP"),
    ("launch spotify", "OPEN_APP"),
    
    ("close application", "CLOSE_APP"),
    ("close this", "CLOSE_APP"),
    ("close window", "CLOSE_APP"),
    ("exit application", "CLOSE_APP"),
    ("quit app", "CLOSE_APP"),
    
    # ✅ NEW: File and folder operations (separate command type)
    ("open file", "FILE_FOLDER_OPERATION"),
    ("open folder", "FILE_FOLDER_OPERATION"),
    ("open document", "FILE_FOLDER_OPERATION"),
    ("launch file", "FILE_FOLDER_OPERATION"),
    ("open my documents", "FILE_FOLDER_OPERATION"),
    ("open downloads folder", "FILE_FOLDER_OPERATION"),
    ("open desktop", "FILE_FOLDER_OPERATION"),
    ("show file", "FILE_FOLDER_OPERATION"),
    ("browse to folder", "FILE_FOLDER_OPERATION"),
    ("open pictures", "FILE_FOLDER_OPERATION"),
    ("open videos folder", "FILE_FOLDER_OPERATION"),
    ("open music", "FILE_FOLDER_OPERATION"),
    ("show folder", "FILE_FOLDER_OPERATION"),
    ("browse file", "FILE_FOLDER_OPERATION"),

    # Text & Input
    ("type text", "TYPE_TEXT"),
    ("write something", "TYPE_TEXT"),
    ("enter text", "TYPE_TEXT"),

    # Mouse actions
    ("click on something", "MOUSE_CLICK"),
    ("click here", "MOUSE_CLICK"),
    ("right click", "MOUSE_RIGHTCLICK"),
    ("double click", "MOUSE_DOUBLECLICK"),

    # Window & System
    ("maximize window", "WINDOW_ACTION"),
    ("minimize window", "WINDOW_ACTION"),
    ("fullscreen mode", "WINDOW_ACTION"),
    ("take screenshot", "SYSTEM"),
    ("lock screen", "SYSTEM"),

    # Keyboard shortcuts
    ("copy", "KEYBOARD"),
    ("paste", "KEYBOARD"),
    ("save", "KEYBOARD"),
    ("undo", "KEYBOARD"),

    # Complex commands (consolidated into fewer types)
    ("open app and search", "APP_WITH_ACTION"),
    ("launch app and type", "APP_WITH_ACTION"),
    ("open app and play", "APP_WITH_ACTION"),
    ("start app and compose", "APP_WITH_ACTION"),

    # Media commands (consolidated)
    ("play music", "MEDIA_CONTROL"),
    ("play video", "MEDIA_CONTROL"),
    ("stream music", "MEDIA_CONTROL"),
    ("stream video", "MEDIA_CONTROL"),

    # ADD NEW PATTERNS (simpler - no message content):
("send whatsapp to", "SEND_MESSAGE"),
("send message to", "SEND_MESSAGE"),
("whatsapp to", "SEND_MESSAGE"),
("email to", "SEND_MESSAGE"),
("post on social", "SEND_MESSAGE"),
("message to", "SEND_MESSAGE"),
("whatsapp mom", "SEND_MESSAGE"),
("email john", "SEND_MESSAGE"),

    # Web commands (unified)
    ("search for something", "WEB_SEARCH"),
    ("google something", "WEB_SEARCH"),
    ("youtube search", "WEB_SEARCH"),
    ("open youtube", "WEB_SEARCH"),
    ("profile work search python", "WEB_SEARCH"),
    ("with profile personal search", "WEB_SEARCH"),
    ("chrome profile dev open youtube", "WEB_SEARCH"),
    ("open gmail", "WEB_SEARCH"),
    ("go to facebook", "WEB_SEARCH"),
    ("search amazon", "WEB_SEARCH"),
]

# ============================================================================
# REUSABLE STEP TEMPLATES (DRY principle)
# ============================================================================

STEP_TEMPLATES = {
    # Reusable: Open any app via Windows search
    "open_app_windows": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
        {"action_type": "WAIT", "parameters": {"duration": 0.5}, "description": "Wait for menu"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{app_name}"}, "description": "Type: {app_name}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Launch {app_name}"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for app to load"},
    ],
        # Reusable: Open file/folder via File Explorer with search
   
    
    # Reusable: Search for file/folder in File Explorer
    "search_file_explorer": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+e"}, "description": "Open File Explorer"},
        {"action_type": "WAIT", "parameters": {"duration": 1.5}, "description": "Wait for Explorer"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+f"}, "description": "Focus search box"},
        {"action_type": "WAIT", "parameters": {"duration": 0.5}, "description": "Wait for search box"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{search_target}"}, "description": "Search for: {search_target}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Execute search"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for search results"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Open first result"},
    ],


    # Reusable: Chrome with profile selection
    "chrome_with_profile": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
        {"action_type": "WAIT", "parameters": {"duration": 0.5}, "description": "Wait for menu"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "chrome"}, "description": "Type Chrome"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Launch Chrome"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for Chrome"},
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"profile_name": "{profile_name}"}, "description": "Select profile: {profile_name}"},
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Profile loaded"},
    ],

    # Reusable: Navigate to website
    "navigate_to_website": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+l"}, "description": "Focus address bar"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{website}"}, "description": "Go to: {website}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Navigate"},
        {"action_type": "WAIT", "parameters": {"duration": 2.5}, "description": "Wait for page load"},
    ],

    # Reusable: Search on page
    "search_on_page": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "/"}, "description": "Focus search"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{search_query}"}, "description": "Type: {search_query}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Search"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for results"},
    ],
    "open_chat_contact": [
    {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+n"}, "description": "New chat"},
    {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait for search"},
    {"action_type": "TYPE_TEXT", "parameters": {"text": "{recipient}"}, "description": "Search: {recipient}"},
    {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait for results"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Open chat"},
    {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Chat opened"},
],

"type_and_send_message": [
    {"action_type": "TYPE_TEXT", "parameters": {"text": "{message_content}"}, "description": "Type message"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Send message"},
],
}

# ============================================================================
# MODEL 2: COMMAND TYPE RULES (using reusable templates)
# ============================================================================

MODEL2_STEP_RULES = {
        
        # ✅ SIMPLIFIED: Regular app operations (no file/folder logic)
    "OPEN_APP": STEP_TEMPLATES["open_app_windows"],
    
    "CLOSE_APP": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "alt+f4"}, "description": "Close window"},
    ],
    
    # ✅ NEW: Dedicated file/folder operation command
    "FILE_FOLDER_OPERATION": [
        
        
        # Unknown file/folder - need to search
        
        *STEP_TEMPLATES["search_file_explorer"],
    ],




    "WEB_SEARCH": [
        *STEP_TEMPLATES["chrome_with_profile"],
        *STEP_TEMPLATES["navigate_to_website"],
        {"action_type": "CONDITIONAL", "parameters": {"condition": "search_query_exists"}, "description": "Check search needed"},
        *STEP_TEMPLATES["search_on_page"],
    ],

    "TYPE_TEXT": [
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{text_content}"}, "description": "Type: {text_content}"},
    ],

    "MOUSE_CLICK": [
        {"action_type": "MOUSE_CLICK", "parameters": {"target": "{action_target}"}, "description": "Click: {action_target}"},
    ],

    "MOUSE_RIGHTCLICK": [
        {"action_type": "MOUSE_RIGHTCLICK", "parameters": {}, "description": "Right click"},
    ],

    "MOUSE_DOUBLECLICK": [
        {"action_type": "MOUSE_DOUBLECLICK", "parameters": {}, "description": "Double click"},
    ],

    "WINDOW_ACTION": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+up"}, "description": "Window action: {window_action}"},
    ],

    "KEYBOARD": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "{keyboard_shortcut}"}, "description": "Press: {keyboard_shortcut}"},
    ],

    "SYSTEM": [
        {"action_type": "SYSTEM_ACTION", "parameters": {"action": "{system_action}"}, "description": "System: {system_action}"},
    ],

    # Consolidated: App with action (search/type/play/compose)
    "APP_WITH_ACTION": [
        *STEP_TEMPLATES["open_app_windows"],
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait for app ready"},
        {"action_type": "CONDITIONAL", "parameters": {"condition": "has_search_query"}, "description": "Check action type"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+l"}, "description": "Focus search/input"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{action_content}"}, "description": "Enter: {action_content}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Execute"},
    ],

    # Consolidated: Media control
    "MEDIA_CONTROL": [
        *STEP_TEMPLATES["open_app_windows"],
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for media app"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+l"}, "description": "Focus search"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{media_query}"}, "description": "Search: {media_query}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Play"},
    ],

    # Consolidated: Messaging
    "SEND_MESSAGE": [
    # PHASE 1: Open app and navigate to contact's chat
    *STEP_TEMPLATES["open_app_windows"],
    {"action_type": "WAIT", "parameters": {"duration": 3}, "description": "Wait for app to load"},
    *STEP_TEMPLATES["open_chat_contact"],
    
    # PHASE 2: Type and send message (CONDITIONAL)
    {"action_type": "CONDITIONAL", "parameters": {"condition": "has_message_content"}, "description": "Check if message provided"},
    *STEP_TEMPLATES["type_and_send_message"],
],
}

# ============================================================================
# HELPER FUNCTIONS (reusable extraction logic)
# ============================================================================

def extract_profile_name(text):
    """Extract Chrome profile name from text"""
    patterns = [
        r'with chrome profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'chrome profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'with profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'use profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()

    return DEFAULT_CHROME_PROFILE or "Default"

def extract_website_and_action(text):
    """Extract website URL and search query from text"""
    websites = {
        'youtube': 'youtube.com', 'google': 'google.com', 'gmail': 'mail.google.com',
        'facebook': 'facebook.com', 'twitter': 'twitter.com', 'instagram': 'instagram.com',
        'linkedin': 'linkedin.com', 'github': 'github.com', 'reddit': 'reddit.com',
        'amazon': 'amazon.com', 'netflix': 'netflix.com', 'spotify': 'open.spotify.com',
    }

    text_l = text.lower()
    website = 'google.com'  # default

    # Detect website
    for keyword, url in websites.items():
        if keyword in text_l:
            website = url
            break

    # Extract query (remove profile, website names, command words)
    query_text = text_l
    for pattern in [r'with chrome profile [\w\s]+', r'chrome profile [\w\s]+', r'profile [\w\s]+']:
        query_text = re.sub(pattern, '', query_text)

    skip_words = {'with', 'chrome', 'search', 'for', 'open', 'go', 'to', 'on', 'in', 'and', 
                  'youtube', 'google', 'gmail', 'facebook', 'profile'}
    query_words = [w for w in query_text.split() if w not in skip_words and w.strip()]

    return website, ' '.join(query_words) if query_words else None

def extract_app_name(words, trigger_words):
    """Extract app name after trigger words"""
    for trigger in trigger_words:
        if trigger in words:
            idx = words.index(trigger)
            app_words = [w for w in words[idx+1:] if w not in ['app', 'application', 'program']]
            if app_words:
                return ' '.join(app_words)
    return None

def extract_text_after_keywords(words, keywords, skip_words):
    """Generic function to extract text after certain keywords"""
    for keyword in keywords:
        if keyword in words:
            idx = words.index(keyword)
            text_words = [w for w in words[idx+1:] if w not in skip_words]
            if text_words:
                return ' '.join(text_words)
    return None
def extract_file_or_folder_path(words, raw_command):
    """
    Detect if command is for file/folder and extract path/name.
    Returns: (is_file_operation, target_name, target_type, is_known_folder)
    """
    # File/folder indicators
    file_indicators = ['file', 'document', 'doc', 'pdf', 'image', 'video', 'folder', 'directory']
    
    # Common folder names with full paths
    common_folders = {
        'documents': r'%USERPROFILE%\Documents',
        'downloads': r'%USERPROFILE%\Downloads',
        'desktop': r'%USERPROFILE%\Desktop',
        'pictures': r'%USERPROFILE%\Pictures',
        'videos': r'%USERPROFILE%\Videos',
        'music': r'%USERPROFILE%\Music',
    }
    
    is_file_operation = any(indicator in words for indicator in file_indicators)
    
    if not is_file_operation:
        return False, None, None, False  # ✅ FIXED: Return 4 values
    
    # Check for common folder names
    for folder_keyword, folder_path in common_folders.items():
        if folder_keyword in words:
            return True, folder_path, 'folder', True  # ✅ Return 4 values: is_file_op, path, type, is_known
    
    # Extract custom file/folder name (needs search)
    skip_words = {'open', 'file', 'folder', 'document', 'my', 'the', 'launch', 'show', 'browse', 'to'}
    target_words = [w for w in words if w not in skip_words]
    
    if target_words:
        target_name = ' '.join(target_words)
        target_type = 'folder' if 'folder' in words or 'directory' in words else 'file'
        return True, target_name, target_type, False  # ✅ Return 4 values: is_file_op, name, type, NOT known (needs search)
    
    return False, None, None, False  # ✅ FIXED: Return 4 values



# ============================================================================
# KEYWORD EXTRACTION (simplified using helpers)
# ============================================================================

def extract_keywords_by_command_type(raw_command, command_type):
    """Unified keyword extraction using helper functions"""
    raw_command_lower = raw_command.lower().strip()
    words = raw_command_lower.split()

    extracted = {
    'app_name': None, 
    'search_query': None, 
    'text_content': None,
    'action_target': None, 
    'keyboard_shortcut': None, 
    'system_action': None,
    'window_action': None, 
    'profile_name': None, 
    'website': None,
    'media_query': None, 
    'recipient': None, 
    'message_content': None,
    'action_content': None,
    'is_file_operation': False,
    'file_path': None,
    'target_type': None,
    'is_known_folder': False,
    'needs_search': False,
    'search_target': None,
    # ✅ NEW:
    'has_message_content': False,
}


    # OPEN_APP / CLOSE_APP
        # OPEN_APP / CLOSE_APP (now handles files/folders too)
    
        # ✅ SIMPLIFIED: OPEN_APP (no file/folder logic)
    if command_type == "OPEN_APP":
        trigger = ['open', 'launch', 'start', 'run']
        extracted['app_name'] = extract_app_name(words, trigger) or ('chrome' if 'chrome' in words else 'current')
    
    # ✅ SIMPLIFIED: CLOSE_APP (no file/folder logic)
    elif command_type == "CLOSE_APP":
        trigger = ['close', 'exit', 'quit']
        extracted['app_name'] = extract_app_name(words, trigger) or 'current'
    
    # ✅ NEW: FILE_FOLDER_OPERATION (dedicated handler)
    elif command_type == "FILE_FOLDER_OPERATION":
        is_file_op, target_name, target_type, is_known = extract_file_or_folder_path(words, raw_command_lower)
        
        extracted['is_file_operation'] = True
        extracted['is_known_folder'] = is_known
        extracted['needs_search'] = not is_known
        extracted['target_type'] = target_type
        
        if is_known:
            # Known folder - use direct path
            extracted['file_path'] = target_name
        else:
            # Unknown file/folder - will search
            extracted['search_target'] = target_name
    


    # WEB_SEARCH
    elif command_type == "WEB_SEARCH":
        extracted['profile_name'] = extract_profile_name(raw_command_lower)
        website, query = extract_website_and_action(raw_command_lower)
        extracted['website'] = website
        extracted['search_query'] = query

    # TYPE_TEXT
    elif command_type == "TYPE_TEXT":
        extracted['text_content'] = extract_text_after_keywords(words, ['type', 'write', 'enter'], {'text', 'message'})

    # MOUSE_CLICK
    elif command_type in ["MOUSE_CLICK", "MOUSE_RIGHTCLICK", "MOUSE_DOUBLECLICK"]:
        skip = {'click', 'on', 'here', 'it', 'this', 'right', 'double'}
        extracted['action_target'] = ' '.join([w for w in words if w not in skip]) or 'current'

    # WINDOW_ACTION
    elif command_type == "WINDOW_ACTION":
        extracted['window_action'] = 'maximize' if any(w in words for w in ['maximize', 'fullscreen']) else 'minimize'

    # KEYBOARD
    elif command_type == "KEYBOARD":
        shortcuts = {'copy': 'ctrl+c', 'paste': 'ctrl+v', 'save': 'ctrl+s', 'undo': 'ctrl+z'}
        for word, shortcut in shortcuts.items():
            if word in words:
                extracted['keyboard_shortcut'] = shortcut
                break

    # SYSTEM
    elif command_type == "SYSTEM":
        extracted['system_action'] = 'screenshot' if 'screenshot' in words or 'capture' in words else 'lock'

    # APP_WITH_ACTION (consolidated: search/type/play/compose)
    elif command_type == "APP_WITH_ACTION":
        if 'and' in words:
            and_idx = words.index('and')
            extracted['app_name'] = extract_app_name(words[:and_idx], ['open', 'launch', 'start'])
            extracted['action_content'] = ' '.join([w for w in words[and_idx+1:] if w not in ['search', 'type', 'play']])

    # MEDIA_CONTROL (consolidated: music/video streaming)
    elif command_type == "MEDIA_CONTROL":
        apps = ['spotify', 'netflix', 'youtube', 'vlc']
        extracted['app_name'] = next((app for app in apps if app in words), 'spotify')
        extracted['media_query'] = ' '.join([w for w in words if w not in ['play', 'stream', 'music', 'video', extracted['app_name']]])

    # SEND_MESSAGE (consolidated: WhatsApp/email/social)
    elif command_type == "SEND_MESSAGE":
    # 1. Detect which messaging app
        apps_map = {
        'whatsapp': 'whatsapp', 
        'email': 'outlook', 
        'social': 'facebook',
        'twitter': 'twitter',
        'instagram': 'instagram',
        'telegram': 'telegram',
    }
        extracted['app_name'] = next((v for k, v in apps_map.items() if k in raw_command_lower), 'whatsapp')
    
    # 2. Extract FULL recipient (supports multi-word names like "john smith")
    if 'to' in words:
        to_idx = words.index('to')
        recipient_words = words[to_idx + 1:]
        if recipient_words:
            extracted['recipient'] = ' '.join(recipient_words)
    
    # 3. Message content e
    return extracted

# ============================================================================
# STEP GENERATION (handles conditional steps)
# ============================================================================

def generate_steps_model2(command_type, extracted_keywords):
    """Generate steps with placeholder replacement and conditional logic"""
    if command_type not in MODEL2_STEP_RULES:
        return [{"action_type": "EXECUTE", "parameters": {}, "description": f"Execute: {command_type}"}]

    steps_template = MODEL2_STEP_RULES[command_type]
    generated_steps = []

    for step in steps_template:
        # Handle conditional steps
                # Handle conditional steps
                # Handle conditional steps
        if step.get("action_type") == "CONDITIONAL":
            condition = step["parameters"].get("condition")
            
            # Known folder check (FILE_FOLDER_OPERATION only)
            if condition == "is_known_folder":
                if not extracted_keywords.get('is_known_folder'):
                    # Not a known folder, skip direct navigation
                    continue
                else:
                    # Known folder, use direct path
                    continue
            
            # Needs search check (FILE_FOLDER_OPERATION only)
            if condition == "needs_search":
                if not extracted_keywords.get('needs_search'):
                    # Doesn't need search, stop (already handled by known folder)
                    break
                else:
                    # Needs search, continue with search steps
                    continue
            
            # Search query check (WEB_SEARCH)
            if condition == "search_query_exists":
                if not extracted_keywords.get('search_query'):
                    break
                else:
                    continue
            
            # Action content check (APP_WITH_ACTION)
            if condition == "has_search_query":
                if not extracted_keywords.get('action_content'):
                    break
                else:
                    continue
                
            if condition == "has_message_content":
                if not extracted_keywords.get('message_content'):
                # No message - stop before typing and sending
                    break
                else:
                 # Message provided - continue with typing and sending
                    continue

            
            continue  # Skip conditional marker



        step_copy = {"action_type": step["action_type"], "parameters": dict(step["parameters"]), "description": step["description"]}

        
        replacements = {
            "{app_name}": extracted_keywords.get('app_name', 'app'),
            "{website}": extracted_keywords.get('website', 'google.com'),
            "{profile_name}": extracted_keywords.get('profile_name', 'Default'),
            "{search_query}": extracted_keywords.get('search_query', ''),
            "{text_content}": extracted_keywords.get('text_content', ''),
            "{action_target}": extracted_keywords.get('action_target', 'target'),
            "{keyboard_shortcut}": extracted_keywords.get('keyboard_shortcut', ''),
            "{system_action}": extracted_keywords.get('system_action', ''),
            "{window_action}": extracted_keywords.get('window_action', ''),
            "{media_query}": extracted_keywords.get('media_query', ''),
            "{recipient}": extracted_keywords.get('recipient', ''),
            "{message_content}": extracted_keywords.get('message_content', ''),
            "{action_content}": extracted_keywords.get('action_content', ''),
            # ✅ ADD THIS LINE:
            "{file_path}": extracted_keywords.get('file_path', ''),
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

# ============================================================================
# MODEL 1: TF-IDF SIMILARITY
# ============================================================================

def calculate_tfidf_similarity(str1, str2):
    """Calculate similarity between two strings"""
    words1, words2 = str1.lower().split(), str2.lower().split()
    matches = sum(1 for word in words1 if word in words2)
    similarity = matches / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
    order_bonus = sum(0.1 for i in range(min(len(words1), len(words2))) if words1[i] == words2[i])
    return min(1.0, similarity + order_bonus)

def process_command_model1(input_text):
    """Model 1: Command type classifier"""
    best_match, highest_similarity = None, 0

    for pattern, cmd_type in MODEL1_TRAINING_DATA:
        similarity = calculate_tfidf_similarity(input_text, pattern)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = (pattern, cmd_type)

    return {
        "input": input_text,
        "command_type": best_match[1],
        "confidence": highest_similarity,
        "training_pattern": best_match[0],
    } if best_match else None

# ============================================================================
# PIPELINE & UI
# ============================================================================


def test_unified_pipeline(input_text):
    """
    Test the complete pipeline with THREE-PHASE messaging:
    PHASE 1: Display and execute steps to open chat
    PHASE 2: Ask user for message
    PHASE 3: Display and execute steps to send message
    """
    global test_count
    
    print("\n" + "="*80)
    print("🤖 EVA PIPELINE".center(80))
    print("="*80)
    
    # STEP 1: Model 1 - Classification
    print("\n🔄 STEP 1: Command Classification...")
    print("-" * 80)
    model1_result = process_command_model1(input_text)
    if not model1_result:
        print("⚠️  No match found!")
        return
    
    print(f"✅ Input: \"{model1_result['input']}\"")
    print(f"✅ Type: {model1_result['command_type']}")
    print(f"✅ Confidence: {model1_result['confidence'] * 100:.2f}%")
    
    # STEP 2: Keyword Extraction
    print("\n🔄 STEP 2: Keyword Extraction...")
    print("-" * 80)
    extracted_keywords = extract_keywords_by_command_type(
        model1_result['input'], 
        model1_result['command_type']
    )
    print("✅ Extracted Keywords:")
    for key, value in extracted_keywords.items():
        if value:
            print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # ✅ FOR MESSAGING: Three-phase execution
    if model1_result['command_type'] == "SEND_MESSAGE":
        app_name = extracted_keywords.get('app_name', 'whatsapp')
        recipient = extracted_keywords.get('recipient', 'contact')
        
        # ============================================================
        # PHASE 1: Display and EXECUTE steps to open chat
        # ============================================================
        print("\n" + "="*80)
        print("📱 PHASE 1: OPENING CHAT".center(80))
        print("="*80)
        
        # Don't include message content for Phase 1
        extracted_keywords['has_message_content'] = False
        
        print("\n🔄 STEP 3: Step Generation - Opening Chat...")
        print("-" * 80)
        steps_phase1 = generate_steps_model2(model1_result['command_type'], extracted_keywords)
        
        # Print Phase 1 steps (all steps to open chat)
        print("✅ Steps to Open Chat:")
        step_count = 0
        for step in steps_phase1:
            # Skip conditional steps
            if step['action_type'] == "CONDITIONAL":
                continue
            
            step_count += 1
            print(f"\n{step_count}. {step['description']}")
            print(f"   Action: {step['action_type']}")
            if step['parameters']:
                for k, v in step['parameters'].items():
                    if v:
                        print(f"   {k}: {v}")
        
        # Show execution summary
        print("\n" + "-"*80)
        print("⏳ Executing steps...")
        print("-"*80)
        for step in steps_phase1:
            if step['action_type'] == "CONDITIONAL":
                continue
            if step['action_type'] == "WAIT":
                print(f"⏳ {step['description']}")
            else:
                print(f"✓ {step['description']}")
        
        print("\n✅ Chat opened successfully!")
        
        # ============================================================
        # PHASE 2: Ask user for message
        # ============================================================
        print("\n" + "="*80)
        print("💬 PHASE 2: MESSAGE INPUT".center(80))
        print("="*80)
        print(f"\n📱 App: {app_name.upper()}")
        print(f"👤 Chat with: {recipient}")
        print("-" * 80)
        
        message = input("\n📝 Enter your message (or press Enter to cancel): ").strip()
        
        if not message:
            print("\n❌ Message cancelled. Operation stopped.")
            return
        
        print(f"\n✅ Message to send: \"{message}\"")
        
        # ============================================================
        # PHASE 3: Display and EXECUTE steps to send message
        # ============================================================
        print("\n" + "="*80)
        print("📤 PHASE 3: SENDING MESSAGE".center(80))
        print("="*80)
        
        # Add message content for Phase 3
        extracted_keywords['message_content'] = message
        extracted_keywords['has_message_content'] = True
        
        print("\n🔄 STEP 4: Step Generation - Sending Message...")
        print("-" * 80)
        steps_phase3 = generate_steps_model2(model1_result['command_type'], extracted_keywords)
        
        # Extract only the send-related steps (last 2 steps: type + send)
        send_steps = [s for s in steps_phase3 if s['action_type'] != "CONDITIONAL"]
        send_steps = send_steps[-2:] if len(send_steps) > 2 else send_steps
        
        print("✅ Steps to Send Message:")
        for i, step in enumerate(send_steps, 1):
            print(f"\n{i}. {step['description']}")
            print(f"   Action: {step['action_type']}")
            if step['parameters']:
                for k, v in step['parameters'].items():
                    if v:
                        print(f"   {k}: {v}")
        
        # Show execution summary
        print("\n" + "-"*80)
        print("⏳ Executing steps...")
        print("-"*80)
        for step in send_steps:
            if step['action_type'] == "WAIT":
                print(f"⏳ {step['description']}")
            else:
                print(f"✓ {step['description']}")
        
        print("\n✅ Message sent successfully!")
    
    else:
        # ============================================================
        # NOT MESSAGING: Normal single-phase execution
        # ============================================================
        print("\n🔄 STEP 3: Step Generation...")
        print("-" * 80)
        steps = generate_steps_model2(model1_result['command_type'], extracted_keywords)
        print("✅ Generated Steps:")
        step_count = 0
        for step in steps:
            if step['action_type'] == "CONDITIONAL":
                continue
            
            step_count += 1
            print(f"\n{step_count}. {step['description']}")
            print(f"   Action: {step['action_type']}")
            if step['parameters']:
                for k, v in step['parameters'].items():
                    if v:
                        print(f"   {k}: {v}")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!".center(80))
    print("="*80)
    test_count += 1

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 80)
    print("🤖 EVA - OPTIMIZED TESTING PLATFORM".center(80))
    print("="*80)

def get_message_from_user(app_name, recipient):
    """
    Ask user for message content AFTER chat is opened.
    This is Phase 2 of the workflow.
    """
    print("\n" + "=" * 80)
    print("💬 MESSAGE INPUT".center(80))
    print("=" * 80)
    print(f"📱 App: {app_name.upper()}")
    print(f"👤 Chat with: {recipient}")
    print("-" * 80)
    
    message = input("\n📝 Enter your message (or press Enter to cancel): ").strip()
    
    if not message:
        print("\n❌ Message cancelle.")
        return None
    
    print(f"\n✅ Message will be sent: \"{message}\"")
    return message

def main():
    """Main loop"""
    global DEFAULT_CHROME_PROFILE
    clear_screen()
    print_header()
    set_default_chrome_profile()

    print("\n📊 STATUS")
    print("-" * 80)
    print(f"   Training Patterns: {len(MODEL1_TRAINING_DATA)}")
    print(f"   Command Types: {len(MODEL2_STEP_RULES)}")
    print(f"   Chrome Profile: {DEFAULT_CHROME_PROFILE}")
    print("-" * 80)

    while True:
        print("\n📋 MENU")
        print("-" * 80)
        print("   1. Test Pipeline")
        print("   2. Exit")
        print("-" * 80)

        choice = input("\nChoice: ").strip()

        if choice == '1':
            input_text = input("\nEnter command: ").strip()
            if input_text:
                test_unified_pipeline(input_text)
            input("\nPress Enter...")
        elif choice == '2':
            print("\n👋 Goodbye!\n")
            break
        else:
            print("\n⚠️  Invalid choice!")

        clear_screen()
        print_header()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
