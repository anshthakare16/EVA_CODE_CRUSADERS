import os
import sys
import re
import threading
import time
import warnings
import json
import random
import string
import hashlib
import smtplib
import ssl
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import speech_recognition as sr
from PySide6.QtGui import QTextCursor
from vision.edge_search_handler import EdgeSearchHandler

import config
from execution.executor_bridge import ExecutorBridge
from vision.face_auth import FaceAuthenticator
from execution.action_router import ActionRouter
from execution.system_executor import SystemExecutor
from vision.screenshot_handler import ScreenshotHandler
from vision.screen_analyzer import ScreenAnalyzer
from vision.omniparser_executor import OmniParserExecutor
from speech.wake_word_detector import WakeWordDetector
from mail import start_mail_composition

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# === Qt (PySide6) ===

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QIcon, QMovie, QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QPlainTextEdit,
    QVBoxLayout, QHBoxLayout, QFrame, QStackedWidget, QSizePolicy,
    QDialog, QMessageBox
)

# Hide that pkg_resources deprecation notice from dependencies
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

# Load environment variables from .env file
load_dotenv()

# ============================================================================ 
# EVA_TER LOGIC (INTEGRATED) — unchanged datasets and step templates
# ============================================================================ 

MODEL1_TRAINING_DATA = [
    ("open application", "OPEN_APP"), ("launch program", "OPEN_APP"), ("start software", "OPEN_APP"),
    ("run app", "OPEN_APP"), ("open app", "OPEN_APP"), ("open chrome", "OPEN_APP"), ("launch spotify", "OPEN_APP"),
    ("open word", "OPEN_APP"),
("open microsoft word", "OPEN_APP"),
("launch word", "OPEN_APP"),
("start word document", "OPEN_APP"),
("open ms word", "OPEN_APP"),
("open word processor", "OPEN_APP"),

("open spotify", "OPEN_APP"),
("launch spotify", "OPEN_APP"),
("start spotify", "OPEN_APP"),
("play spotify", "OPEN_APP"),
("open music on spotify", "OPEN_APP"),
("spotify open", "OPEN_APP"),

("open notepad", "OPEN_APP"),
("launch notepad", "OPEN_APP"),
("start notepad", "OPEN_APP"),
("open text editor", "OPEN_APP"),
("open note pad", "OPEN_APP"),

("open chrome", "OPEN_APP"),
("open google chrome", "OPEN_APP"),
("launch chrome", "OPEN_APP"),
("start chrome browser", "OPEN_APP"),
("open browser chrome", "OPEN_APP"),

("open vscode", "OPEN_APP"),
("open visual studio code", "OPEN_APP"),
("launch vscode", "OPEN_APP"),
("start code editor", "OPEN_APP"),
("open vs code", "OPEN_APP"),

("open task manager", "OPEN_APP"),
("launch task manager", "OPEN_APP"),
("start taskmgr", "OPEN_APP"),
("open task manager ctrl shift esc", "OPEN_APP"),

("open paint", "OPEN_APP"),
("launch paint", "OPEN_APP"),
("start mspaint", "OPEN_APP"),
("open ms paint", "OPEN_APP"),

("open powerpoint", "OPEN_APP"),
("open microsoft powerpoint", "OPEN_APP"),
("launch powerpoint", "OPEN_APP"),
("start ppt", "OPEN_APP"),

("open excel", "OPEN_APP"),
("open microsoft excel", "OPEN_APP"),
("launch excel", "OPEN_APP"),
("start spreadsheet", "OPEN_APP"),

("open edge", "OPEN_APP"),
("open microsoft edge", "OPEN_APP"),
("launch edge", "OPEN_APP"),
("start edge browser", "OPEN_APP"),

("open cmd", "OPEN_APP"),
("open command prompt", "OPEN_APP"),
("launch cmd", "OPEN_APP"),
("start command line", "OPEN_APP"),

("open settings", "OPEN_APP"),
("open windows settings", "OPEN_APP"),
("launch settings", "OPEN_APP"),
("start settings app", "OPEN_APP"),
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
 ("post on social", "SEND_MESSAGE"), ("message to", "SEND_MESSAGE"),
    ("whatsapp mom", "SEND_MESSAGE"), 
    ("text to", "SEND_MESSAGE"), ("text message to", "SEND_MESSAGE"),
("send a text to", "SEND_MESSAGE"), ("send a whatsapp to", "SEND_MESSAGE"),
("send an email to", "SEND_MESSAGE"), ("compose email to", "SEND_MESSAGE"),
 ("draft message to", "SEND_MESSAGE"),
("send sms to", "SEND_MESSAGE"), ("sms to", "SEND_MESSAGE"),

    # === WEB_SEARCH Commands (Dynamic - works with ANY website) ===
    ("search for something", "WEB_SEARCH"), ("google something", "WEB_SEARCH"), ("youtube search", "WEB_SEARCH"),
    ("open youtube", "WEB_SEARCH"), ("open gmail", "WEB_SEARCH"), ("go to facebook", "WEB_SEARCH"),
    ("search amazon", "WEB_SEARCH"), ("search python on google", "WEB_SEARCH"), ("search for machine learning", "WEB_SEARCH"),
    ("github search api documentation", "WEB_SEARCH"), ("wikipedia search artificial intelligence", "WEB_SEARCH"),
    ("stackoverflow search error handling", "WEB_SEARCH"), ("search reddit python tutorials", "WEB_SEARCH"),
    ("open github", "WEB_SEARCH"), ("search wikipedia", "WEB_SEARCH"), ("search stackoverflow for bug", "WEB_SEARCH"),
    ("bing search tutorial", "WEB_SEARCH"), ("duckduckgo search privacy", "WEB_SEARCH"), ("medium search article", "WEB_SEARCH"),
    ("twitter search news", "WEB_SEARCH"), ("instagram search photos", "WEB_SEARCH"), ("linkedin search jobs", "WEB_SEARCH"),
    ("pinterest search ideas", "WEB_SEARCH"), ("reddit search community", "WEB_SEARCH"), ("imdb search movie", "WEB_SEARCH"),
    ("ebay search products", "WEB_SEARCH"), ("quora search answers", "WEB_SEARCH"), ("netflix search shows", "WEB_SEARCH"),
    ("search on google for python programming", "WEB_SEARCH"), ("find information about ai on wikipedia", "WEB_SEARCH"),
    ("open stack overflow and search for javascript", "WEB_SEARCH"), ("search github for projects", "WEB_SEARCH"),
    ("look up tutorials on youtube", "WEB_SEARCH"), ("browse reddit for discussions", "WEB_SEARCH"),
    ("go to perplexity", "WEB_SEARCH"),
    ("go to gemini", "WEB_SEARCH"),
    ("go to grok", "WEB_SEARCH"),
    ("go to chatgpt", "WEB_SEARCH"),
    ("go to claude", "WEB_SEARCH"),
    ("go to google", "WEB_SEARCH"),
    ("go to youtube", "WEB_SEARCH"),
    ("go to github", "WEB_SEARCH"),
    ("go to stackoverflow", "WEB_SEARCH"),
    ("go to reddit", "WEB_SEARCH"),
    ("go to wikipedia", "WEB_SEARCH"),
    ("go to amazon", "WEB_SEARCH"),
    ("go to twitter", "WEB_SEARCH"),
    ("go to linkedin", "WEB_SEARCH"),
    ("go to flipkart", "WEB_SEARCH"),
    ("turn on wifi", "SYSTEM"),
    ("turn off wifi", "SYSTEM"),
    ("enable bluetooth", "SYSTEM"),
    ("disable bluetooth", "SYSTEM"),
    ("turn on flight mode", "SYSTEM"),
    ("turn off airplane mode", "SYSTEM"),
    ("enable night light", "SYSTEM"),
    ("turn on battery saver", "SYSTEM"),
    ("enable hotspot", "SYSTEM"),
    ("set volume to 50", "SYSTEM"),
    ("increase volume", "SYSTEM"),
    ("mute volume", "SYSTEM"),
    ("set brightness to 70", "SYSTEM"),
    ("shutdown computer", "SYSTEM"),
    ("restart system", "SYSTEM"),
    ("lock computer", "SYSTEM"),
    ("put computer to sleep", "SYSTEM"),
    ("send mail", "SENDMAIL"),
    ("email to shriya", "SENDMAIL"),
    ("compose a mail to anu", "SENDMAIL"),
    ("send an email", "SENDMAIL"),
    ("email to shriya", "SENDMAIL"),
    ("send mail", "SENDMAIL"),
    ("email to shriya", "SENDMAIL"),
    ("compose a mail to anu", "SENDMAIL"),
    ("send an email", "SENDMAIL"),
    ("send email", "SENDMAIL"),
    ("send an email", "SENDMAIL"),
    ("email to shriya", "SENDMAIL"),
    ("email to anu", "SENDMAIL"),
    ("compose a mail to anu", "SENDMAIL"),
    ("compose email", "SENDMAIL"),
    ("compose a mail", "SENDMAIL"),
    ("send mail to john", "SENDMAIL"),
    ("write an email", "SENDMAIL"),
    ("draft email", "SENDMAIL"),
    ("send a mail", "SENDMAIL"),
    ("mail compose", "SENDMAIL"),
    ("write email to", "SENDMAIL"),
    # === CALCULATOR Commands ===
    ("open calculator", "CALCULATOR"),
    ("launch calculator", "CALCULATOR"),
    ("calculator", "CALCULATOR"),
    ("calculate 25 plus 30", "CALCULATOR"),
    ("calculate 100 minus 50", "CALCULATOR"),
    ("multiply 12 by 8", "CALCULATOR"),
    ("divide 144 by 12", "CALCULATOR"),
    ("calculate square root of 144", "CALCULATOR"),
    ("calculate 25 percent of 200", "CALCULATOR"),
    ("clear calculator", "CALCULATOR"),
    ("calculator equals", "CALCULATOR"),
    ("switch calculator to scientific mode", "CALCULATOR"),
    ("switch calculator to standard mode", "CALCULATOR"),
    ("calculate sine of 45", "CALCULATOR"),
    ("calculate cosine of 90", "CALCULATOR"),
    ("calculate tangent of 30", "CALCULATOR"),
    ("calculator memory store", "CALCULATOR"),
    ("calculator memory recall", "CALCULATOR"),
    ("calculator power 2 to the 8", "CALCULATOR"),
    ("calculate factorial of 5", "CALCULATOR"),
    ("add 15 and 25", "CALCULATOR"),
    ("subtract 50 from 100", "CALCULATOR"),
    ("what is 12 times 9", "CALCULATOR"),
    ("divide 200 by 4", "CALCULATOR"),
    ("square of 15", "CALCULATOR"),
    # === CAMERA Commands ===
    ("open camera", "CAMERA"),
    ("launch camera", "CAMERA"),
    ("start camera", "CAMERA"),
    ("take photo", "CAMERA"),
    ("capture photo", "CAMERA"),
    ("snap a photo", "CAMERA"),
    ("take a picture", "CAMERA"),
    ("capture a picture", "CAMERA"),
    ("snap a picture", "CAMERA"),
    ("click a photo", "CAMERA"),
    ("click a picture", "CAMERA"),
    ("take selfie", "CAMERA"),
    ("capture selfie", "CAMERA"),
    ("snap selfie", "CAMERA"),
    ("take a selfie", "CAMERA"),
    ("camera on", "CAMERA"),
    ("activate camera", "CAMERA"),
    ("camera app", "CAMERA"),
    ("open camera app", "CAMERA"),
    ("launch camera app", "CAMERA"),
    ("camera front", "CAMERA"),
    ("switch to front camera", "CAMERA"),
    ("camera back", "CAMERA"),
    ("switch to back camera", "CAMERA"),
    ("take a video", "CAMERA"),
    ("record video", "CAMERA"),
    ("capture video", "CAMERA"),
    ("video recording", "CAMERA"),
    ("start recording", "CAMERA"),
    ("stop recording", "CAMERA"),
    ("flash on", "CAMERA"),
    ("flash off", "CAMERA"),
    ("enable flash", "CAMERA"),
    ("disable flash", "CAMERA"),
    ("night mode", "CAMERA"),
    ("portrait mode", "CAMERA"),
    ("panorama mode", "CAMERA"),
    ("zoom in camera", "CAMERA"),
    ("zoom out camera", "CAMERA"),
    ("camera zoom", "CAMERA"),
    ("focus on face", "CAMERA"),
    ("smile detection", "CAMERA"),
    ("take burst photo", "CAMERA"),
    ("picture perfect", "CAMERA"),
    ("shoot", "CAMERA"),
    ("music play pause", "SPOTIFY_CONTROL"),
    ("music pause", "SPOTIFY_CONTROL"),
    ("music play", "SPOTIFY_CONTROL"),
    ("music next song", "SPOTIFY_CONTROL"),
    ("music next", "SPOTIFY_CONTROL"),
    ("music previous song", "SPOTIFY_CONTROL"),
    ("music previous", "SPOTIFY_CONTROL"),
    ("music back", "SPOTIFY_CONTROL"),
    # === CLOCK ALARM Commands ===
    ("set alarm for 7 am", "CLOCK_ALARM"),
    ("set alarm at 8 30", "CLOCK_ALARM"),
    ("create alarm for 6 pm", "CLOCK_ALARM"),
    ("wake me up at 9 am", "CLOCK_ALARM"),
("alarm for 5 30 pm", "CLOCK_ALARM"),
("set alarm for 6:15 am", "CLOCK_ALARM"),
("set alarm 10 pm", "CLOCK_ALARM"),
("create alarm 7:45", "CLOCK_ALARM"),
("wake me at 5 am", "CLOCK_ALARM"),
("alarm at 11 30 pm", "CLOCK_ALARM"),
("set alarm for 12 pm", "CLOCK_ALARM"),
("set alarm for noon", "CLOCK_ALARM"),
("set alarm for midnight", "CLOCK_ALARM"),
("alarm for 8 in the morning", "CLOCK_ALARM"),
("set alarm 9 30 am", "CLOCK_ALARM"),
("create alarm at 6:00 pm", "CLOCK_ALARM"),
("wake me up 7:30 am", "CLOCK_ALARM"),
("set an alarm for 4 pm", "CLOCK_ALARM"),
("alarm 10:15 am", "CLOCK_ALARM"),
("set morning alarm at 6", "CLOCK_ALARM"),
("create alarm 5 45 pm", "CLOCK_ALARM"),
("alarm for 3:30 in the afternoon", "CLOCK_ALARM"),
("set alarm for 2 am", "CLOCK_ALARM"),
("wake up alarm 8 am", "CLOCK_ALARM"),
("set alarm for quarter past 7", "CLOCK_ALARM"),
("alarm for half past 9 am", "CLOCK_ALARM"),
("set alarm 11:00 pm", "CLOCK_ALARM"),
("create wake up alarm 6:30", "CLOCK_ALARM"),
("set daily alarm 7 am", "CLOCK_ALARM"),
("alarm for 4:20 pm", "CLOCK_ALARM"),
]

STEP_TEMPLATES = {
    "open_app_windows": [
    {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
    {"action_type": "WAIT", "parameters": {"duration": 0.5}, "description": "Wait for menu"},
    {"action_type": "TYPE_TEXT", "parameters": {"text": "{app_name}"}, "description": "Type: {app_name}"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Launch {app_name}"},
    {"action_type": "WAIT", "parameters": {"duration": 5}, "description": "Wait for app to load"},
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
        {"action_type": "CONDITIONAL", "parameters": {"condition": "is_default_profile"}, "description": "Check if default profile"},
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
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Press Enter"},
    ],
    "type_and_send_message": [
        {"action_type": "WAIT", "parameters": {"duration": 1}, "description": "Wait before typing"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{message_content}"}, "description": "Type message"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Send message"},
    ],
    "set_alarm": [
    {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
    {"action_type": "TYPE_TEXT", "parameters": {"text": "clock"}, "description": "Type clock"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Open Clock app"},
    {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for app to open"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "down"}, "description": "Navigate down"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "down"}, "description": "Navigate to Alarm section"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Open Alarms"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab once"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab twice"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab three times"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab four times"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab five times"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab six times"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Add new alarm"},
    {"action_type": "TYPE_TEXT", "parameters": {"text": "{hour}"}, "description": "Type hour"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Move to minute"},
    {"action_type": "TYPE_TEXT", "parameters": {"text": "{minute}"}, "description": "Type minute"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab 1"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab 2"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab 3"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab 4"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab 5"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "tab"}, "description": "Tab to Save button"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Save alarm"},
],

"open_camera": [
    {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
    {"action_type": "WAIT", "parameters": {"duration": 1.5}, "description": "Wait for menu"},
    {"action_type": "TYPE_TEXT", "parameters": {"text": "camera"}, "description": "Type camera"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Launch Camera app"},
    {"action_type": "WAIT", "parameters": {"duration": 5.0}, "description": "Wait for Camera to open"},
],
"spotify_playpause": [
    {"action_type": "PRESS_KEY", "parameters": {"key": "space"}, "description": "Play/Pause Spotify"},
],
"spotify_next": [
    {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+right"}, "description": "Next song"},
],
"spotify_previous": [
    {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+left"}, "description": "Previous song"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+left"}, "description": "Previous song"},
],


"open_calculator": [
    {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
    {"action_type": "WAIT", "parameters": {"duration": 1.0}, "description": "Wait for menu"},
    {"action_type": "TYPE_TEXT", "parameters": {"text": "calculator"}, "description": "Type calculator"},
    {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Launch Calculator"},
    {"action_type": "WAIT", "parameters": {"duration": 2.0}, "description": "Wait for app to load"},
],
}

MODEL2_STEP_RULES = {
    "OPEN_APP": STEP_TEMPLATES["open_app_windows"],
    "CLOSE_APP": [{"action_type": "PRESS_KEY", "parameters": {"key": "alt+f4"}, "description": "Close window"}],
    "OPEN_FILE_EXPLORER": [{"action_type": "PRESS_KEY", "parameters": {"key": "win+e"}, "description": "Open File Explorer"}],
    "SEARCH_FILE": [*STEP_TEMPLATES["search_file_explorer"]],
    "OPEN_FOLDER": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+e"}, "description": "Open File Explorer"},
        {"action_type": "WAIT", "parameters": {"duration": 1.5}, "description": "Wait for Explorer"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+l"}, "description": "Focus address bar"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{file_path}"}, "description": "Navigate to folder"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Open folder"},
    ],
    "WEB_SEARCH": [
        {"action_type": "WEB_SEARCH_DIRECT", "parameters": {"website": "{website}", "search_query": "{search_query}"}, "description": "Search {website} for: {search_query}"},
    ],
    "TYPE_TEXT": [{"action_type": "TYPE_TEXT", "parameters": {"text": "{text_content}"}, "description": "Type: {text_content}"}],
    "FOCUS_WINDOW": [{"action_type": "FOCUS_WINDOW", "parameters": {"title": "{app_name}"}, "description": "Focus: {app_name}"}],
    "MOUSE_CLICK": [{"action_type": "SCREEN_ANALYSIS", "parameters": {"target": "{action_target}"}, "description": "Click: {action_target}"}],
    "MOUSE_RIGHTCLICK": [{"action_type": "MOUSE_RIGHTCLICK", "parameters": {}, "description": "Right click"}],
    "MOUSE_DOUBLECLICK": [{"action_type": "MOUSE_DOUBLECLICK", "parameters": {}, "description": "Double click"}],
    "WINDOW_ACTION": [{"action_type": "PRESS_KEY", "parameters": {"key": "win+up"}, "description": "Window action: {window_action}"}],
    "KEYBOARD": [{"action_type": "PRESS_KEY", "parameters": {"key": "{keyboard_shortcut}"}, "description": "Press: {keyboard_shortcut}"}],
        "SYSTEM": {
    "WIFI": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+a"}, "description": "Open Quick Settings"},
        {"action_type": "WAIT", "parameters": {"duration": 0.8}, "description": "Wait for Quick Settings"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Toggle WiFi"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "escape"}, "description": "Close Quick Settings"},
    ],
    "BLUETOOTH": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+a"}, "description": "Open Quick Settings"},
        {"action_type": "WAIT", "parameters": {"duration": 0.8}, "description": "Wait for Quick Settings"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate to Bluetooth"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Toggle Bluetooth"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "escape"}, "description": "Close Quick Settings"},
    ],
    "FLIGHTMODE": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+a"}, "description": "Open Quick Settings"},
        {"action_type": "WAIT", "parameters": {"duration": 0.8}, "description": "Wait for Quick Settings"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate to Flight Mode"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Toggle Flight Mode"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "escape"}, "description": "Close Quick Settings"},
    ],
    "NIGHTLIGHT": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+a"}, "description": "Open Quick Settings"},
        {"action_type": "WAIT", "parameters": {"duration": 0.8}, "description": "Wait for Quick Settings"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate to Night Light"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Toggle Night Light"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "escape"}, "description": "Close Quick Settings"},
    ],
    "ENERGYSAVER": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+a"}, "description": "Open Quick Settings"},
        {"action_type": "WAIT", "parameters": {"duration": 0.8}, "description": "Wait for Quick Settings"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate to Energy Saver"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Toggle Energy Saver"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "escape"}, "description": "Close Quick Settings"},
    ],
    "MOBILEHOTSPOT": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "win+a"}, "description": "Open Quick Settings"},
        {"action_type": "WAIT", "parameters": {"duration": 0.8}, "description": "Wait for Quick Settings"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate right"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "right"}, "description": "Navigate to Mobile Hotspot"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Toggle Mobile Hotspot"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "escape"}, "description": "Close Quick Settings"},
    ],
    "VOLUME": [
        {"action_type": "SET_VOLUME", "parameters": {"level": "<percentage>"}, "description": "Set volume to <percentage>%"}
    ],
    "BRIGHTNESS": [
        {"action_type": "SET_BRIGHTNESS", "parameters": {"level": "<percentage>"}, "description": "Set brightness to <percentage>%"}
    ],
    "DEFAULT": [
        {"action_type": "SYSTEM_ACTION", "parameters": {"action": "<systemaction>"}, "description": "Execute <systemaction>"}
    ]

},


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
        {"action_type": "PRESS_KEY", "parameters": {"key": "shift+enter"}, "description": "Play"},
    ],
    "SEND_MESSAGE": [
        *STEP_TEMPLATES["open_app_windows"],
        {"action_type": "WAIT", "parameters": {"duration": 3}, "description": "Wait for app to load"},
        *STEP_TEMPLATES["whatsapp_open_chat"],
    ],
    "SEND_MESSAGE_PHASE_2": [
        *STEP_TEMPLATES["type_and_send_message"],
    ],
    "CALCULATOR": [],
}

# ----------------- Utility: asset path helper -----------------
def asset_path(name: str) -> str:
    base, ext = os.path.splitext(name)
    variants = {
        name,
        name.replace("/", os.sep).replace("\\", os.sep),
        base + ext.lower(),
        (base.capitalize() + ext.lower()),
        name.lower(),
        name.upper(),
    }
    folders = [
        os.getcwd(),
        os.path.join(os.getcwd(), "graphics"),
        os.path.join(os.getcwd(), "Frontend", "Graphics"),
    ]
    for folder in folders:
        for v in variants:
            p = os.path.join(folder, v)
            if os.path.exists(p):
                return p
    return name  # let QMovie try

# ----------------- Signals container -----------------
class Bus(QObject):
    log = Signal(str)
    status = Signal(str)
    steps_ready = Signal(list)
    exec_done = Signal(dict)

# ----------------- NEW: Passcode storage + OTP utilities (ADDED) -----------------
# These helpers and dialogs were added and do not remove or modify any of your original logic.
PASSCODE_FILE = os.path.join(os.getcwd(), "passcode_store.json")

def _hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode('utf-8')).hexdigest()

def load_stored_passcode():
    """Load hashed passcode from file. If not present, initialize with default 1304."""
    default_pin = "1304"
    if not os.path.exists(PASSCODE_FILE):
        data = {
            "hashed": _hash_pin(default_pin),
            "updated_at": datetime.utcnow().isoformat()
        }
        try:
            with open(PASSCODE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
        return data['hashed']
    try:
        with open(PASSCODE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if 'hashed' in data:
                return data['hashed']
    except Exception:
        pass
    return _hash_pin(default_pin)

def store_new_passcode(pin: str):
    hashed = _hash_pin(pin)
    data = {"hashed": hashed, "updated_at": datetime.utcnow().isoformat()}
    try:
        with open(PASSCODE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False

def generate_numeric_otp(length=6):
    return ''.join(random.choices(string.digits, k=length))

def send_email_otp(recipient_email: str, otp: str) -> (bool, str):
    """
    Attempts to send OTP via SMTP using config.* settings.
    Returns (success, message). If SMTP not configured, returns False with a message.
    """
    smtp_host = getattr(config, "SMTP_HOST", "")
    smtp_port = getattr(config, "SMTP_PORT", 587)
    smtp_user = getattr(config, "SMTP_USER", "")
    smtp_pass = getattr(config, "SMTP_PASSWORD", "")
    smtp_tls = getattr(config, "SMTP_USE_TLS", True)

    if not smtp_host or not smtp_user or not smtp_pass:
        return False, "SMTP not configured - development fallback will be used."

    try:
        port = int(smtp_port)
    except Exception:
        port = 587

    subject = "Your EVA Reset OTP"
    body = f"Your One Time Password for resetting EVA passcode is: {otp}\nThis code is valid for a short time."

    message = f"Subject: {subject}\nTo: {recipient_email}\nFrom: {smtp_user}\n\n{body}"

    try:
        if smtp_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, port, timeout=10) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipient_email, message)
        else:
            with smtplib.SMTP(smtp_host, port, timeout=10) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipient_email, message)
        return True, "OTP sent via SMTP"
    except Exception as e:
        return False, f"SMTP send failed: {e}"

# ----------------- NEW: Dialogs for Forgot / Reset Passcode -----------------
class EmailInputDialog:
    @staticmethod
    def get_email(parent=None, default_email="thakareansh3@gmail.com"):
        dlg = QDialog(parent)
        dlg.setWindowTitle("Reset Passcode - Enter Email")
        dlg.setWindowModality(Qt.WindowModal)
        layout = QVBoxLayout(dlg)
        label = QLabel("Enter your email to receive OTP:")
        layout.addWidget(label)
        email_input = QLineEdit()
        email_input.setPlaceholderText("you@example.com")
        email_input.setText(default_email)
        layout.addWidget(email_input)
        row = QHBoxLayout()
        ok_btn = QPushButton("Send OTP")
        cancel_btn = QPushButton("Cancel")
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

        ok = {"val": False}

        def do_ok():
            e = email_input.text().strip()
            if not e or "@" not in e:
                QMessageBox.warning(dlg, "Invalid", "Please enter a valid email address.")
                return
            ok["val"] = True
            dlg.accept()

        ok_btn.clicked.connect(do_ok)
        cancel_btn.clicked.connect(dlg.reject)
        email_input.returnPressed.connect(do_ok)

        res = dlg.exec()
        return email_input.text().strip(), ok["val"]

class OTPVerifyDialog:
    @staticmethod
    def verify_otp(parent, expected_otp, show_otp=False):
        dlg = QDialog(parent)
        dlg.setWindowTitle("Verify OTP")
        dlg.setWindowModality(Qt.WindowModal)
        layout = QVBoxLayout(dlg)
        label = QLabel("Enter the OTP sent to your email")
        layout.addWidget(label)
        otp_input = QLineEdit()
        otp_input.setPlaceholderText("6-digit OTP")
        otp_input.setMaxLength(len(expected_otp))
        layout.addWidget(otp_input)

        if show_otp:
            note = QLabel(f"(Development) OTP: {expected_otp}")
            note.setStyleSheet("color: #ffaa00;")
            layout.addWidget(note)

        row = QHBoxLayout()
        ok_btn = QPushButton("Verify")
        cancel_btn = QPushButton("Cancel")
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

        verified = {"val": False}

        def do_verify():
            if otp_input.text().strip() == expected_otp:
                verified["val"] = True
                dlg.accept()
            else:
                QMessageBox.warning(dlg, "Invalid", "OTP incorrect.")
        ok_btn.clicked.connect(do_verify)
        cancel_btn.clicked.connect(dlg.reject)
        otp_input.returnPressed.connect(do_verify)

        dlg.exec()
        return verified["val"]

class NewPasscodeDialog:
    @staticmethod
    def get_new_passcode(parent=None):
        dlg = QDialog(parent)
        dlg.setWindowTitle("Set New Passcode")
        dlg.setWindowModality(Qt.WindowModal)
        layout = QVBoxLayout(dlg)

        label = QLabel("Enter new 4-digit passcode")
        layout.addWidget(label)
        p1 = QLineEdit()
        p1.setEchoMode(QLineEdit.Password)
        p1.setMaxLength(4)
        p1.setPlaceholderText("New passcode")
        layout.addWidget(p1)

        label2 = QLabel("Confirm new passcode")
        layout.addWidget(label2)
        p2 = QLineEdit()
        p2.setEchoMode(QLineEdit.Password)
        p2.setMaxLength(4)
        p2.setPlaceholderText("Confirm passcode")
        layout.addWidget(p2)

        row = QHBoxLayout()
        ok_btn = QPushButton("Update")
        cancel_btn = QPushButton("Cancel")
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

        result = {"val": None}

        def do_update():
            a = p1.text().strip()
            b = p2.text().strip()
            if not a or not b:
                QMessageBox.warning(dlg, "Invalid", "Both fields required.")
                return
            if len(a) != 4 or not a.isdigit():
                QMessageBox.warning(dlg, "Invalid", "Passcode must be 4 digits.")
                return
            if a != b:
                QMessageBox.warning(dlg, "Mismatch", "Passcodes do not match.")
                return
            result["val"] = a
            dlg.accept()

        ok_btn.clicked.connect(do_update)
        cancel_btn.clicked.connect(dlg.reject)
        p2.returnPressed.connect(do_update)
        dlg.exec()
        return result["val"]

class PasscodeDialog(QDialog):
    """
    Full-screen passcode dialog. Default passcode is loaded from passcode_store.json (1304 if not present).
    Includes 'Forgot / Reset' flow that uses SMTP via config or a dev fallback showing OTP on screen.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setWindowState(Qt.WindowFullScreen)
        self.setStyleSheet("background-color: black;")
        self.failed = 0

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(40, 40, 40, 40)

        self.prompt = QLabel("Enter 4-digit passcode")
        self.prompt.setStyleSheet("color: white; font-size: 28px;")
        self.prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prompt)

        self.pin = QLineEdit()
        self.pin.setMaxLength(4)
        self.pin.setEchoMode(QLineEdit.Password)
        self.pin.setPlaceholderText("• • • •")
        self.pin.setFixedWidth(240)
        self.pin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pin.setStyleSheet("font-size: 36px; padding:10px; background:#111; color:#fff; border-radius:8px;")
        layout.addWidget(self.pin, alignment=Qt.AlignmentFlag.AlignCenter)

        btn_row = QHBoxLayout()
        self.unlock_btn = QPushButton("Unlock")
        self.unlock_btn.setFixedWidth(160)
        self.unlock_btn.setStyleSheet("background:#1e90ff; color:#fff; padding:8px; border:none; border-radius:8px;")
        btn_row.addWidget(self.unlock_btn)

        self.forgot_btn = QPushButton("Forgot / Reset")
        self.forgot_btn.setFixedWidth(180)
        self.forgot_btn.setStyleSheet("background:#444; color:#fff; padding:8px; border:none; border-radius:8px;")
        btn_row.addWidget(self.forgot_btn)

        # Face unlock button (added)
        self.face_btn = QPushButton("Face Unlock")
        self.face_btn.setFixedWidth(140)
        self.face_btn.setStyleSheet("background:#2b8f2b; color:#fff; padding:8px; border:none; border-radius:8px;")
        btn_row.addWidget(self.face_btn)

        layout.addLayout(btn_row)

        self.error = QLabel("")
        self.error.setStyleSheet("color: #ff6666; font-size: 14px;")
        self.error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.error)

        self.unlock_btn.clicked.connect(self.check)
        self.pin.returnPressed.connect(self.check)
        self.forgot_btn.clicked.connect(self._forgot_flow)
        self.face_btn.clicked.connect(self._try_face_unlock)

        # focus input
        QTimer.singleShot(200, self.pin.setFocus)

    def check(self):
        code = self.pin.text().strip()
        if not code:
            return
        stored_hash = load_stored_passcode()
        if _hash_pin(code) == stored_hash:
            self.accept()
        else:
            self.failed += 1
            self.error.setText("Incorrect passcode")
            self.pin.clear()
            self.pin.setStyleSheet("font-size: 36px; padding:10px; background:#220000; color:#fff; border-radius:8px;")
            QTimer.singleShot(120, lambda: self.pin.setStyleSheet("font-size: 36px; padding:10px; background:#111; color:#fff; border-radius:8px;"))
            if self.failed >= 5:
                QMessageBox.critical(self, "Locked", "Too many incorrect attempts. Exiting.")
                sys.exit(0)

    def _forgot_flow(self):
        """
        Ask for email -> attempt to send OTP via SMTP (config) -> verify OTP -> ask new passcode -> save
        If SMTP fails or not configured the OTP is shown on-screen as a dev fallback.
        """
        email, ok = EmailInputDialog.get_email(self, default_email="thakareansh3@gmail.com")
        if not ok:
            return

        otp = generate_numeric_otp(6)
        sent, msg = send_email_otp(email, otp)
        if sent:
            QMessageBox.information(self, "OTP Sent", f"An OTP was sent to {email}. Check your email.")
            show_otp = False
        else:
            QMessageBox.warning(self, "OTP Delivery (fallback)",
                                f"Could not send OTP via SMTP.\nReason: {msg}\n\nFor development, OTP is: {otp}")
            show_otp = True

        verified = OTPVerifyDialog.verify_otp(self, otp, show_otp)
        if not verified:
            QMessageBox.warning(self, "Failed", "OTP verification failed or cancelled.")
            return

        new_pin = NewPasscodeDialog.get_new_passcode(self)
        if not new_pin:
            QMessageBox.information(self, "Cancelled", "Password reset cancelled.")
            return

        success = store_new_passcode(new_pin)
        if success:
            QMessageBox.information(self, "Success", "Passcode updated successfully.")
        else:
            QMessageBox.critical(self, "Error", "Failed to save new passcode. Try again.")

    def _try_face_unlock(self):
        """
        Attempts local face unlock using vision.face_auth.FaceAuthenticator.
        Shows a message box on success/failure. This uses a short-lived FaceAuthenticator instance.
        """
        try:
            fa = FaceAuthenticator(known_faces_dir=os.path.join(os.getcwd(), "known_faces"))
        except Exception as e:
            QMessageBox.warning(self, "Face Unlock", f"FaceAuth not available: {e}")
            return

        # Inform user and give time to face camera
        QMessageBox.information(self, "Face Unlock", "Initializing camera. Please face the camera for a few seconds.")
        QApplication.processEvents()

        try:
            result = fa.authenticate(camera_index=0, timeout=8.0, required_matches=2)
            if result:
                name, dist = result
                QMessageBox.information(self, "Face Unlock", f"Welcome, {name}!")
                self.accept()
            else:
                QMessageBox.warning(self, "Face Unlock", "Could not recognize face. Try again or use passcode.")
        except Exception as e:
            QMessageBox.warning(self, "Face Unlock", f"Face unlock failed: {e}")

# ----------------- Main App (UNCHANGED) -----------------
class EvaGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EVA - Integrated Logic Assistant")
        self.resize(1100, 680)
        self.setStyleSheet("background-color: #000000; color: #e6e6e6;")

        self.bus = Bus()
        self.bus.log.connect(self.append_log)
        self.bus.status.connect(self.set_status)
        self.bus.exec_done.connect(self.display_execution_result)

        self._is_muted = False
        self._is_awake = False
        self._movie = None

        # Backend fields
        self.vision_enabled = False
        self.current_steps = []
        self.current_model1_result = None
        self.current_extracted_keywords = None
        self.action_router = None

        self._build_ui()
        self._init_backend_async()
        self._start_wake_word_thread()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Top bar with Home/Chats
        top = QHBoxLayout()
        top.setSpacing(8)

        self.btn_home = QPushButton("  Home")
        self.btn_home.setIcon(QIcon(asset_path(r"graphics\home.png")))
        self.btn_home.clicked.connect(lambda: self._switch_tab(0))
        self._style_tab_button(self.btn_home, active=True)

        self.btn_chats = QPushButton("  Chats")
        self.btn_chats.setIcon(QIcon(asset_path(r"graphics\Chats.png")))
        self.btn_chats.clicked.connect(lambda: self._switch_tab(1))
        self._style_tab_button(self.btn_chats, active=False)

        spacer = QFrame()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        top.addWidget(self.btn_home)
        top.addWidget(self.btn_chats)
        top.addWidget(spacer)
        root.addLayout(top)

        # Stacked pages
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        # --- Home page ---
        page_home = QWidget()
        ph_layout = QVBoxLayout(page_home)
        ph_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        ph_layout.setSpacing(16)

        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gif_label.setStyleSheet("background: transparent;")
        ph_layout.addWidget(self.gif_label, 0, Qt.AlignmentFlag.AlignHCenter)

        self.status_label = QLabel("Available...")
        self.status_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignHCenter)

        mic_row = QHBoxLayout()
        mic_row.setSpacing(8)
        mic_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_mic = QPushButton()
        self.btn_mic.setIcon(QIcon(asset_path(r"graphics\mic_on.png")))
        self.btn_mic.setIconSize(QSize(48, 48))
        self.btn_mic.setFixedSize(56, 56)
        self.btn_mic.setStyleSheet("border: none;")
        self.btn_mic.clicked.connect(self._toggle_mic)
        mic_row.addWidget(self.btn_mic)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText("Type a command and press Enter…")
        self.entry.setStyleSheet("background:#111; color:#fff; padding:10px; border:1px solid #222;")
        self.entry.returnPressed.connect(self._on_submit)
        mic_row.addWidget(self.entry)

        self.btn_submit = QPushButton("Submit")
        self.btn_submit.setStyleSheet("background:#1e90ff; border:none; padding:8px 14px; color:#fff;")
        self.btn_submit.clicked.connect(self._on_submit)
        mic_row.addWidget(self.btn_submit)

        ph_layout.addLayout(mic_row)

        # start GIF via QMovie
        self._start_movie(asset_path(r"graphics\jarvis.gif"))

        # --- Chats page ---
        page_chats = QWidget()
        pc_layout = QVBoxLayout(page_chats)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(8)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background:#0b0b0b; color:#e6e6e6; border:1px solid #222;")
        pc_layout.addWidget(self.log_view, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.entry2 = QLineEdit()
        self.entry2.setPlaceholderText("Type a command and press Enter…")
        self.entry2.setStyleSheet("background:#111; color:#fff; padding:10px; border:1px solid #222;")
        self.entry2.returnPressed.connect(self._on_submit_from_chats)
        input_row.addWidget(self.entry2, 1)

        self.btn_submit2 = QPushButton("Submit")
        self.btn_submit2.setStyleSheet("background:#1e90ff; border:none; padding:8px 14px; color:#fff;")
        self.btn_submit2.clicked.connect(self._on_submit_from_chats)
        input_row.addWidget(self.btn_submit2)
        pc_layout.addLayout(input_row)

        self.stack.addWidget(page_home)   # index 0
        self.stack.addWidget(page_chats)  # index 1

    def _style_tab_button(self, btn: QPushButton, active: bool):
        if active:
            btn.setStyleSheet("""
                QPushButton {
                    background:#ffffff; color:#000; border:none; padding:8px 12px; border-radius:8px; font-weight:600;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background:#222; color:#e6e6e6; border:none; padding:8px 12px; border-radius:8px;
                }
                QPushButton:hover { background:#333; }
            """)

    def _switch_tab(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self._style_tab_button(self.btn_home, active=(idx == 0))
        self._style_tab_button(self.btn_chats, active=(idx == 1))

    def _start_movie(self, path: str):
        self._movie = QMovie(path)
        # Fastest Qt can do is the GIF's own frame delay
        self._movie.setCacheMode(QMovie.CacheAll)
        self.gif_label.setMovie(self._movie)
        self._movie.start()

    # ---------- UI events ----------
    def _toggle_mic(self):
        self._is_muted = not self._is_muted
        self.btn_mic.setIcon(QIcon(asset_path(r"graphics\mic_off.png" if self._is_muted else r"graphics\mic_on.png")))
        self.bus.status.emit("Muted." if self._is_muted else "Listening for command...")

    def _on_submit(self):
        text = self.entry.text().strip()
        if text:
            self.entry.clear()
            self.bus.status.emit(f"Recognized: {text}")
            threading.Thread(target=self._run_eva_pipeline, args=(text,), daemon=True).start()
            # jump to Chats so user sees logs
            self._switch_tab(1)

    def _on_submit_from_chats(self):
        text = self.entry2.text().strip()
        if text:
            self.entry2.clear()
            self.bus.status.emit(f"Recognized: {text}")
            threading.Thread(target=self._run_eva_pipeline, args=(text,), daemon=True).start()

    # ---------- Logging / status ----------
    def append_log(self, text: str):
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.ensureCursorVisible()
    def set_status(self, text: str):
        self.status_label.setText(text)

    # ---------- Backend init ----------
    def _init_backend_async(self):
        def work():
            try:
                self.bus.log.emit("Initializing backend components...\n")
                self.executor_bridge = ExecutorBridge()
                self.system_executor = SystemExecutor(self.executor_bridge)
                self.screenshot_handler = ScreenshotHandler()
                self.bus.log.emit("✓ Execution engine loaded.\n")

                self.screen_analyzer = ScreenAnalyzer(config.GEMINI_API_KEY)
                self.omniparser = OmniParserExecutor()
                self.action_router = ActionRouter(
                    self.system_executor, self.screenshot_handler, self.screen_analyzer, self.omniparser
                )
                self.vision_enabled = True
                self.bus.log.emit("✓ Vision system loaded successfully.\n")

                self.wake_word_detector = WakeWordDetector()
                self.bus.log.emit("✓ Wake word detector loaded successfully.\n")

                # Face authentication (non-blocking load)
                try:
                    self.face_auth = FaceAuthenticator(
                        known_faces_dir=os.path.join(os.getcwd(), "known_faces"),
                    )
                    self.bus.log.emit("✓ FaceAuthenticator loaded.\n")
                except Exception as e:
                    self.face_auth = None
                    self.bus.log.emit(f"⚠️ FaceAuthenticator not available: {e}\n")


                # Train classifier
                self.vectorizer = TfidfVectorizer()
                self.classifier = LogisticRegression()
                X, y = zip(*MODEL1_TRAINING_DATA)
                X_vectorized = self.vectorizer.fit_transform(X)
                self.classifier.fit(X_vectorized, y)
                self.bus.log.emit("✓ Command classifier trained.\n")
                self.bus.log.emit("Ready to receive commands.\n")
            except Exception as e:
                msg = (
                    "❌ CRITICAL ERROR: Could not initialize backend.\n"
                    f"{e}\nVision features will be disabled.\n"
                    "Check your .env for GEMINI_API_KEY and ensure all model weights are downloaded.\n"
                )
                self.bus.log.emit(msg)
        threading.Thread(target=work, daemon=True).start()

    def _start_wake_word_thread(self):
        def work():
            # ensure backend wake_word_detector exists
            while not hasattr(self, "wake_word_detector"):
                time.sleep(0.2)
            self.wake_word_detector.start()
            while True:
                if not self._is_awake:
                    self.bus.status.emit("Listening for 'Jarvis'...")
                    if self.wake_word_detector.listen():
                        self._is_awake = True
                        self.bus.status.emit("Wake word detected! Listening for command...")
                else:
                    if self._is_muted:
                        self.bus.status.emit("Muted. Say 'unmute' to resume.")
                        recognizer = sr.Recognizer()
                        with sr.Microphone() as source:
                            try:
                                audio = recognizer.listen(source, timeout=5, phrase_time_limit=2)
                                command = recognizer.recognize_google(audio).lower()
                                if "unmute" in command:
                                    self._is_muted = False
                                    self.bus.status.emit("Unmuted. Listening for command...")
                                    # update mic icon
                                    self.btn_mic.setIcon(QIcon(asset_path(r"graphics\mic_on.png")))
                            except (sr.UnknownValueError, sr.RequestError):
                                pass
                        time.sleep(1)
                        continue

                    self.bus.status.emit("Listening for command...")
                    recognizer = sr.Recognizer()
                    with sr.Microphone() as source:
                        try:
                            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                            prompt = recognizer.recognize_google(audio)
                            self.bus.status.emit(f"Recognized: {prompt}")

                            if "go to sleep" in prompt.lower():
                                self._is_awake = False
                                self.bus.status.emit("Going to sleep. Listening for 'Jarvis'...")
                                continue

                            if "mute" in prompt.lower():
                                self._is_muted = True
                                self.bus.status.emit("Muted.")
                                self.btn_mic.setIcon(QIcon(asset_path(r"graphics\mic_off.png")))
                                continue

                            threading.Thread(target=self._run_eva_pipeline, args=(prompt,), daemon=True).start()
                            # switch to chats to show logs
                            self._switch_tab(1)
                        except sr.UnknownValueError:
                            self.bus.status.emit("Could not understand audio. Listening for command...")
                        except sr.RequestError as e:
                            self.bus.status.emit(f"Could not request results; {e}. Listening for command...")
                        except Exception as e:
                            self.bus.status.emit(f"An error occurred: {e}. Listening for command...")
        threading.Thread(target=work, daemon=True).start()

    # ---------- EVA pipeline (same logic, threaded) ----------
    def _run_eva_pipeline(self, prompt: str):
        self._clear_log()
        self.bus.log.emit(f"Processing command: \"{prompt}\"\n\n")

        self.current_model1_result = self._analyze_query_with_model(prompt)
        if not self.current_model1_result:
            self.bus.log.emit("⚠️ Error analyzing command!")
            return

        self._display_classification_results()

        self.current_extracted_keywords = self._extract_keywords_by_command_type(
            self.current_model1_result['input'], self.current_model1_result['command_type']
        )
        self._display_keyword_results()

        command_type = self.current_model1_result['command_type']
        
        if command_type == "SENDMAIL":
            self.bus.log.emit("Starting mail composition...")
            # Define mute callback to control assistant during mail composition
            def toggle_mute(should_mute):
                self._is_muted = should_mute
                status_msg = "🔇 Muted for mail composition" if should_mute else "🔊 Listening for commands"
                self.bus.status.emit(status_msg)
            
            # Call mail composition function with logging and mute callbacks
            success = start_mail_composition(log_callback=self.bus.log.emit, mute_callback_func=toggle_mute)
            if success:
                self.bus.log.emit("Mail sent successfully. Awaiting next command.")
            else:
                self.bus.log.emit("Mail composition cancelled or failed. Awaiting next command.")
            return
        if command_type == "SEND_MESSAGE":
            self._handle_interactive_messaging()
        else:
            if command_type == "CAMERA" or command_type == "SPOTIFY_CONTROL" or command_type == "CLOCK_ALARM":
                self.current_steps = self._generate_steps_model2(command_type, self.current_extracted_keywords, prompt)
            else:
                self.current_steps = self._generate_steps_model2(command_type, self.current_extracted_keywords)

            self._display_step_results()
            self._execute_steps()  # Automatic execution

    def _handle_interactive_messaging(self):
        if self.current_extracted_keywords.get('message_content'):
            self.bus.log.emit("STEP 3: Generated Steps\n" + "-"*40 + "\n")
            steps = self._generate_steps_model2("SEND_MESSAGE", self.current_extracted_keywords)
            steps.extend(self._generate_steps_model2("SEND_MESSAGE_PHASE_2", self.current_extracted_keywords))
            self._display_steps(steps)
            self.current_steps = steps
            self._execute_steps()
        else:
            self.bus.log.emit("STEP 3: Generated Steps (Phase 1: Open Chat)\n" + "-"*40 + "\n")
            steps_phase1 = self._generate_steps_model2("SEND_MESSAGE", self.current_extracted_keywords)
            self._display_steps(steps_phase1)

            self.current_steps = steps_phase1
            self._execute_steps()

            self.bus.status.emit("What message do you want to send?")
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    audio = recognizer.listen(source, timeout=7, phrase_time_limit=15)
                    message = recognizer.recognize_google(audio)
                    self.bus.status.emit(f"Message: {message}")
                    self.current_extracted_keywords['message_content'] = message

                    self.bus.log.emit("\nSTEP 4: Generated Steps (Phase 2: Send Message)\n" + "-"*40 + "\n")
                    steps_phase2 = self._generate_steps_model2("SEND_MESSAGE_PHASE_2", self.current_extracted_keywords)
                    self._display_steps(steps_phase2, start_index=len(self.current_steps) + 1)

                    self.current_steps = steps_phase2
                    self._execute_steps()

                except sr.UnknownValueError:
                    self.bus.status.emit("Could not understand message. Cancelling.")
                except sr.RequestError as e:
                    self.bus.status.emit(f"Could not request results; {e}. Cancelling.")

    def _execute_steps(self):
        if not self.current_steps or not self.action_router:
            return
        self.bus.log.emit("\nEXECUTING STEPS...\n" + "-"*40 + "\n")
        def work():
            try:
                # Debug: Show what we're executing
                for i, step in enumerate(self.current_steps, 1):
                    self.bus.log.emit(f"[DEBUG] Step {i}: {step.get('action_type')} - {step.get('description')}\n")
                
                self.bus.log.emit(f"[DEBUG] Calling action_router.execute with {len(self.current_steps)} steps...\n")
                
                result = self.action_router.execute(
                    self.current_model1_result['command_type'],
                    self.current_steps,
                    self.current_extracted_keywords,
                    self.current_model1_result['input'],
                    self.current_model1_result
                )
                
                self.bus.log.emit(f"[DEBUG] action_router returned: {result}\n")
                self.bus.exec_done.emit(result)
            except Exception as e:
                self.bus.log.emit(f"[ERROR] Exception in work thread: {e}\n")
                import traceback
                self.bus.log.emit(f"{traceback.format_exc()}\n")
                self.bus.exec_done.emit({"success": False, "error": str(e)})
        threading.Thread(target=work, daemon=True).start()

    def display_execution_result(self, result: dict):
        if result.get('success'):
            self.bus.log.emit("✅ Command executed successfully!\n")
        else:
            self.bus.log.emit(f"❌ ERROR: {result.get('error', 'Unknown error')}\n")

    # ---------- Log helpers ----------
    def _clear_log(self):
        self.log_view.setPlainText("")

    def _display_classification_results(self):
        self.bus.log.emit("STEP 1: Command Classification\n" + "-"*40 + "\n")
        self.bus.log.emit(f"Input: \"{self.current_model1_result['input']}\"\n")
        self.bus.log.emit(f"Type: {self.current_model1_result['command_type']}\n\n")

    def _display_keyword_results(self):
        self.bus.log.emit("STEP 2: Keyword Extraction\n" + "-"*40 + "\n")
        for key, value in self.current_extracted_keywords.items():
            if value:
                self.bus.log.emit(f" • {key.replace('_', ' ').title()}: {value}\n")
        self.bus.log.emit("\n")

    def _display_step_results(self):
        self.bus.log.emit("STEP 3: Generated Steps\n" + "-"*40 + "\n")
        self._display_steps(self.current_steps)

    def _display_steps(self, steps, start_index=1):
        step_count = start_index - 1
        for step in steps:
            if step.get('action_type') == "CONDITIONAL":
                continue
            step_count += 1
            self.bus.log.emit(f"{step_count}. {step['description']}\n")
            self.bus.log.emit(f"   Action: {step['action_type']}\n")
            if step['parameters']:
                for k, v in step['parameters'].items():
                    if v:
                        self.bus.log.emit(f"   {k}: {v}\n")
            self.bus.log.emit("\n")

    # ---------- Model & NLP ----------
    def _analyze_query_with_model(self, query):
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
    def _extract_keywords_by_command_type(self, raw_command, command_type):
        import re
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
            extracted['app_name'] = self._extract_app_name(words, trigger)
        elif command_type == "CLOSE_APP":
            trigger = ['close', 'exit', 'quit']
            extracted['app_name'] = self._extract_app_name(words, trigger) or 'current'
        elif command_type == "OPEN_FOLDER":
            is_file_op, target_name, target_type, is_known = self._extract_file_or_folder_path(words, raw_command_lower)
            if is_known:
                extracted['file_path'] = target_name
            else:
                extracted['search_target'] = target_name
        elif command_type == "SEARCH_FILE":
            is_file_op, target_name, target_type, is_known = self._extract_file_or_folder_path(words, raw_command_lower)
            extracted['search_target'] = target_name
        elif command_type == "WEB_SEARCH":
            extracted['profile_name'] = self._extract_profile_name(raw_command_lower)
            website, query, is_search = self._extract_website_and_action(raw_command_lower)
            extracted['website'] = website
            extracted['search_query'] = query
            extracted['is_search_query'] = is_search  # True if searching, False if just opening
        elif command_type == "TYPE_TEXT":
            extracted['text_content'] = self._extract_text_after_keywords(raw_command.split(), ['type', 'write', 'enter'], {'text', 'message'})
        elif command_type in ["MOUSE_CLICK", "MOUSE_RIGHTCLICK", "MOUSE_DOUBLECLICK"]:
            skip = {'click', 'on', 'here', 'it', 'this', 'right', 'double'}
            extracted['action_target'] = ' '.join([w for w in raw_command.split() if w.lower() not in skip]) or 'current'
        elif command_type == "WINDOW_ACTION":
            extracted['window_action'] = 'maximize' if any(w in raw_command.lower().split() for w in ['maximize', 'fullscreen']) else 'minimize'
        elif command_type == "KEYBOARD":
            shortcuts = {'copy': 'ctrl+c', 'paste': 'ctrl+v', 'save': 'ctrl+s', 'undo': 'ctrl+z'}
            for word, shortcut in shortcuts.items():
                if word in raw_command_lower:
                    extracted['keyboard_shortcut'] = shortcut
                    break
        elif command_type == "SYSTEM":
            # Extract control_type and percentage
            extracted['control_type'] = None
            extracted['percentage'] = None
            extracted['system_action'] = raw_command.lower()
            
            # WiFi
            if 'wifi' in words or 'wi-fi' in words or 'wi fi' in words:
                extracted['control_type'] = 'WIFI'
                extracted['system_action'] = 'WIFI'
            
            # Bluetooth
            elif 'bluetooth' in words:
                extracted['control_type'] = 'BLUETOOTH'
                extracted['system_action'] = 'BLUETOOTH'
            
            # Flight Mode
            elif 'flight' in words or 'airplane' in words:
                extracted['control_type'] = 'FLIGHTMODE'
                extracted['system_action'] = 'FLIGHTMODE'
            
            # Night Light
            elif 'night' in words and 'light' in words:
                extracted['control_type'] = 'NIGHTLIGHT'
                extracted['system_action'] = 'NIGHTLIGHT'
            
            # Energy Saver
            elif 'energy' in words or 'saver' in words or 'battery' in words:
                extracted['control_type'] = 'ENERGYSAVER'
                extracted['system_action'] = 'ENERGYSAVER'
            
            # Mobile Hotspot
            elif 'hotspot' in words or 'mobile' in words:
                extracted['control_type'] = 'MOBILEHOTSPOT'
                extracted['system_action'] = 'MOBILEHOTSPOT'
            
            # Volume
            elif 'volume' in words:
                extracted['control_type'] = 'VOLUME'
                extracted['system_action'] = 'VOLUME'
                # Extract percentage
                import re
                match = re.search(r'\d+', raw_command)
                if match:
                    extracted['percentage'] = int(match.group())
            
            # Brightness
            elif 'brightness' in words:
                extracted['control_type'] = 'BRIGHTNESS'
                extracted['system_action'] = 'BRIGHTNESS'
                # Extract percentage
                import re
                match = re.search(r'\d+', raw_command)
                if match:
                    extracted['percentage'] = int(match.group())

        elif command_type == "APP_WITH_ACTION":
            if 'and' in raw_command_lower:
                and_idx = raw_command_lower.split().index('and')
                extracted['app_name'] = self._extract_app_name(raw_command.split()[:and_idx], ['open', 'launch', 'start'])
                extracted['action_content'] = ' '.join([w for w in raw_command.split()[and_idx+1:] if w.lower() not in ['search', 'type', 'play']])
        elif command_type == "MEDIA_CONTROL":
            apps = ['spotify', 'netflix', 'youtube', 'vlc']
            app_name = next((app for app in apps if app in raw_command_lower), 'spotify')
            extracted['app_name'] = app_name
            play_idx = -1
            if 'play' in raw_command_lower:
                play_idx = raw_command_lower.split().index('play')
            elif 'stream' in raw_command_lower:
                play_idx = raw_command_lower.split().index('stream')
            if play_idx != -1:
                extracted['media_query'] = ' '.join(raw_command.split()[play_idx+1:])
        elif command_type == "SEND_MESSAGE":
            apps_map = {
                'whatsapp': 'whatsapp', 'email': 'outlook', 'social': 'facebook', 'twitter': 'twitter',
                'instagram': 'instagram', 'telegram': 'telegram',
            }
            extracted['app_name'] = next((v for k, v in apps_map.items() if k in raw_command_lower), 'whatsapp')

            match = re.search(r'send\s+(.*?)\s+to\s+(.*)', raw_command, re.IGNORECASE)
            if match:
                extracted['message_content'] = match.group(1)
                extracted['recipient'] = match.group(2)
                return extracted

            match = re.search(r'to\s+(.*?)\s+(?:message|saying|that)\s+(.*)', raw_command, re.IGNORECASE)
            if match:
                extracted['recipient'] = match.group(1)
                extracted['message_content'] = match.group(2)
                return extracted

            match = re.search(r'to\s+(\w+)\s+(.*)', raw_command, re.IGNORECASE)
            if match:
                extracted['recipient'] = match.group(1)
                extracted['message_content'] = match.group(2)
                return extracted

            if 'to' in raw_command_lower:
                to_idx = raw_command_lower.split().index('to')
                extracted['recipient'] = ' '.join(raw_command.split()[to_idx + 1:])
        elif command_type == "CALCULATOR":
            import re
            # Extract numbers and operation
            numbers = re.findall(r'\d+', raw_command)
            extracted['number1'] = numbers[0] if len(numbers) > 0 else ''
            extracted['number2'] = numbers[1] if len(numbers) > 1 else ''
    
            # Extract operation type
            if any(word in raw_command_lower for word in ['plus', 'add', '+']):
                extracted['operation'] = 'plus'
            elif any(word in raw_command_lower for word in ['minus', 'subtract', '-']):
                extracted['operation'] = 'minus'
            elif any(word in raw_command_lower for word in ['multiply', 'times', '*', 'by']):
                extracted['operation'] = 'multiply'
            elif any(word in raw_command_lower for word in ['divide', '/']):
                extracted['operation'] = 'divide'
            elif 'square root' in raw_command_lower or 'sqrt' in raw_command_lower:
                extracted['operation'] = 'square root'
            elif 'clear' in raw_command_lower:
                extracted['operation'] = 'clear'
            elif 'scientific' in raw_command_lower:
                extracted['operation'] = 'scientific'
            elif 'standard' in raw_command_lower:
                extracted['operation'] = 'standard'
        elif command_type == "CAMERA":
            extracted['action_content'] = 'take_photo'
            if "open" in raw_command_lower:
                extracted['action_content'] = 'open_camera'
        elif command_type == "CLOCK_ALARM":
            import re
            # Extract time from command
            # Patterns: "7 am", "8:30", "9 30 pm", etc.
    
            # Try matching HH:MM format
            match = re.search(r'(\d{1,2}):(\d{2})', raw_command)
            if match:
                extracted['hour'] = match.group(1)
                extracted['minute'] = match.group(2)
            else:
                # Try matching separate hour and minute
                numbers = re.findall(r'\d{1,2}', raw_command)
                if len(numbers) >= 1:
                    extracted['hour'] = numbers[0]
                    extracted['minute'] = numbers[1] if len(numbers) > 1 else '00'
    
            # Check for AM/PM
            if 'pm' in raw_command_lower and extracted.get('hour'):
                hour = int(extracted['hour'])
                if hour < 12:
                    extracted['hour'] = str(hour + 12)
            elif 'am' in raw_command_lower and extracted.get('hour'):
                hour = int(extracted['hour'])
                if hour == 12:
                    extracted['hour'] = '0'

        return extracted

    def _generate_steps_model2(self, command_type, extracted_keywords, raw_command=None):
    # Special handling for SYSTEM commands (nested dict)
        if command_type == "SYSTEM":
            control_type = extracted_keywords.get('control_type', '').upper()
            system_action = extracted_keywords.get('system_action', '').upper()
            percentage = extracted_keywords.get('percentage')
        
            rule = MODEL2_STEP_RULES.get(command_type, {})
        
            if not isinstance(rule, dict):
                return [{"action_type": "SYSTEM_ACTION", "parameters": {"action": system_action}, "description": f"Execute {system_action}"}]
        
            # Try to find specific steps
            steps = None
            if control_type in rule:
                steps = rule[control_type]
            elif system_action in rule:
                steps = rule[system_action]
            elif 'DEFAULT' in rule:
                steps = rule['DEFAULT']
        
            if not steps:
                return [{"action_type": "SYSTEM_ACTION", "parameters": {"action": system_action}, "description": f"Execute {system_action}"}]
        
            # Replace <percentage> placeholder
            generated_steps = []
            for step in steps:
                step_copy = {
                "action_type": step.get("action_type", ""),
                "parameters": dict(step.get("parameters", {})),
                "description": step.get("description", "")
                }
            
                # Replace percentage placeholder
                if percentage:
                    for key, value in step_copy['parameters'].items():
                        if isinstance(value, str) and '<percentage>' in value:
                            step_copy['parameters'][key] = percentage
                
                    if '<percentage>' in step_copy['description']:
                        step_copy['description'] = step_copy['description'].replace('<percentage>', str(percentage))
            
                generated_steps.append(step_copy)
        
            return generated_steps
        elif command_type == "CLOCK_ALARM":
            steps = []
    
            hour = extracted_keywords.get('hour', '7')
            minute = extracted_keywords.get('minute', '00')
    
            # Get the template and replace placeholders
            for step in STEP_TEMPLATES["set_alarm"]:
                step_copy = {
            "action_type": step["action_type"],
            "parameters": dict(step["parameters"]),
            "description": step["description"]
                }
        
                # Replace hour and minute placeholders
                if "text" in step_copy["parameters"]:
                    step_copy["parameters"]["text"] = step_copy["parameters"]["text"].replace("{hour}", hour)
                    step_copy["parameters"]["text"] = step_copy["parameters"]["text"].replace("{minute}", minute)
        
                step_copy["description"] = step_copy["description"].replace("{hour}", hour)
                step_copy["description"] = step_copy["description"].replace("{minute}", minute)
        
                steps.append(step_copy)
    
            return steps



        elif command_type == "SPOTIFY_CONTROL":
            steps = []
            raw_lower = raw_command.lower() if raw_command else ""

            # Open Spotify first (reuse open_app_windows template by modifying text param)
            steps.extend([
                {"action_type": "PRESS_KEY", "parameters": {"key": "win"}, "description": "Open Start Menu"},
                {"action_type": "WAIT", "parameters": {"duration": 1.5}, "description": "Wait for Start menu"},
                {"action_type": "TYPE_TEXT", "parameters": {"text": "spotify"}, "description": "Type spotify"},
                {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Launch Spotify"},
                {"action_type": "WAIT", "parameters": {"duration": 5}, "description": "Wait for Spotify to open"},
            ])

            if "play pause" in raw_lower or raw_lower.strip().endswith("pause") or raw_lower.strip().endswith("play"):
                steps.extend(STEP_TEMPLATES["spotify_playpause"])
            elif "next" in raw_lower:
                steps.extend(STEP_TEMPLATES["spotify_next"])
            elif "previous" in raw_lower or "go back" in raw_lower:
                steps.extend(STEP_TEMPLATES["spotify_previous"])

            return steps


        elif command_type == "CAMERA":
            steps = []

            if raw_command is not None:
                raw_lower = raw_command.lower()
            else:
                raw_lower = ""

            # Always open camera first
            steps.extend(STEP_TEMPLATES["open_camera"])
            steps.append({"action_type": "WAIT", "parameters": {"duration": 5}, "description": "Wait to open camera"})

            # Check if trigger words to take photo
            if any(word in raw_lower for word in ["photo", "take", "capture", "selfie"]):
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "space"}, "description": "Take photo"})

            return steps
    



        elif command_type == "CALCULATOR":
            steps = []
        
            # Extract operation details from command
            operation = extracted_keywords.get('operation', '').lower()
            num1 = extracted_keywords.get('number1', '')
            num2 = extracted_keywords.get('number2', '')
        
            # Open calculator first
            steps.extend(STEP_TEMPLATES["open_calculator"])
        
            # Handle different calculator operations
            if 'plus' in operation or 'add' in operation:
                for digit in str(num1):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "shift+="}, "description": "Press plus"})
                for digit in str(num2):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Calculate result"})
        
            elif 'minus' in operation or 'subtract' in operation:
                for digit in str(num1):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "-"}, "description": "Press minus"})
                for digit in str(num2):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Calculate result"})
        
            elif 'multiply' in operation or 'times' in operation:
                for digit in str(num1):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "shift+8"}, "description": "Press multiply"})
                for digit in str(num2):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Calculate result"})
        
            elif 'divide' in operation:
                for digit in str(num1):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "/"}, "description": "Press divide"})
                for digit in str(num2):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Calculate result"})
        
            elif 'square' in operation and 'root' in operation:
                for digit in str(num1):
                    steps.append({"action_type": "PRESS_KEY", "parameters": {"key": digit}, "description": f"Press {digit}"})
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "shift+2"}, "description": "Press square root"})
        
            elif 'clear' in operation:
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "escape"}, "description": "Clear calculator"})
        
            elif 'scientific' in operation:
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "alt+2"}, "description": "Switch to scientific mode"})
        
            elif 'standard' in operation:
                steps.append({"action_type": "PRESS_KEY", "parameters": {"key": "alt+1"}, "description": "Switch to standard mode"})
        
            return steps
    
        # Special handling for WEB_SEARCH (open vs search)
        if command_type == "WEB_SEARCH":
            is_search = extracted_keywords.get('is_search_query', True)
            if not is_search:
                # Just opening the website, not searching
                return [
                    {"action_type": "WEB_OPEN_DIRECT", "parameters": {"website": extracted_keywords.get('website', 'google')}, "description": f"Open {extracted_keywords.get('website', 'website')}"}
                ]
            # Otherwise fall through to normal search steps
        
        # For all other command types, use existing logic
        if command_type not in MODEL2_STEP_RULES:
            return [{"action_type": "EXECUTE", "parameters": {}, "description": f"Execute: {command_type}"}]
    
        steps_template = MODEL2_STEP_RULES[command_type]
        generated_steps = []
    
        for step in steps_template:
            if step.get("action_type") == "CONDITIONAL":
                condition = step["parameters"].get("condition")
                if condition == "search_query_exists":
                    search_query = extracted_keywords.get('search_query')
                    if not search_query:
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
        
            step_copy = {
            "action_type": step["action_type"], 
            "parameters": dict(step["parameters"]), 
            "description": step["description"]
            }
        
            replacements = {
            "{app_name}": extracted_keywords.get('app_name', 'app'), 
            "{website}": extracted_keywords.get('website', ''),
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


    # ---------- small helpers ----------
    def _extract_profile_name(self, text):
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

    def _extract_website_and_action(self, text):
        """
        Extract website and search query from command
        Improved to handle dynamic websites and properly exclude "for" keyword
        Also detects if it's just opening a website vs searching
        
        Patterns supported:
        - "search google for python" -> (website, query, True)
        - "search python on google" -> (website, query, True)
        - "google search for python" -> (website, query, True)
        - "search python programming" -> (website, query, True)
        - "open github" -> (website, None, False)
        - "go to netflix" -> (website, None, False)
        - "search stackoverflow for error message" -> (website, query, True)
        
        Returns:
            tuple: (website, query, is_search_action)
        """
        websites = {
    'youtube': 'youtube',
    'google': 'google',
    'gmail': 'gmail',
    'facebook': 'facebook',
    'twitter': 'twitter',
    'instagram': 'instagram',
    'linkedin': 'linkedin',
    'github': 'github',
    'reddit': 'reddit',
    'amazon': 'amazon',
    'netflix': 'netflix',
    'spotify': 'spotify',
    'stackoverflow': 'stackoverflow',
    'wikipedia': 'wikipedia',
    'bing': 'bing',
    'x': 'x',
    'pinterest': 'pinterest',
    'medium': 'medium',
    'quora': 'quora',
    'ebay': 'ebay',
    'imdb': 'imdb',
    'duckduckgo': 'duckduckgo',
    'chatgpt': 'chatgpt',
    'whatsapp': 'whatsapp',
    'bank': 'bank',
    'hotstar': 'hotstar',
    'flipkart': 'flipkart',
    'cricbuzz': 'cricbuzz',
    'paytm': 'paytm',
    'timesofindia': 'timesofindia',
    'justdial': 'justdial',
    'jiosaavn': 'jiosaavn',
    'hindustantimes': 'hindustantimes',
    'indianexpress': 'indianexpress',
    'espncricinfo': 'espncricinfo',
    'bookmyshow': 'bookmyshow',
    'zeenews': 'zeenews',
    'snapchat': 'snapchat',
    'telegram': 'telegram',
    'magicbricks': 'magicbricks',
    'naukri': 'naukri',
    'shaadi': 'shaadi',
    'policybazaar': 'policybazaar',
    'myntra': 'myntra',
    'zomato': 'zomato',
    'swiggy': 'swiggy',
    'aajtak': 'aajtak',
    'ndtv': 'ndtv',
    'moneycontrol': 'moneycontrol',
    'gaana': 'gaana',
    'yatra': 'yatra',
    '99acres': '99acres',
    'cleartrip': 'cleartrip',
    'makemytrip': 'makemytrip',
    'olx': 'olx',
    'carwale': 'carwale',
    'bikedekho': 'bikedekho',
    'housing': 'housing',
    'bigbasket': 'bigbasket',
    'grofers': 'grofers',
    'reliancedigital': 'reliancedigital',
    'vijaysales': 'vijaysales',
    'croma': 'croma',
    'indiamart': 'indiamart',
    'tradeindia': 'tradeindia',
    'sulekha': 'sulekha',
    'mouthshut': 'mouthshut',
    'whatmobile': 'whatmobile',
    'hindisahityadarpan': 'hindisahityadarpan',
    'abhiyojana': 'abhiyojana',
    'pib': 'pib',
    'incometaxindia': 'incometaxindia',
    'upi': 'upi',
    'bhimgov': 'bhimgov',
    'umang': 'umang',
    'digilocker': 'digilocker',
    'mygov': 'mygov',
    'india': 'india',
    'epfindia': 'epfindia',
    'esi': 'esi',
    'passportindia': 'passportindia',
    'irctc': 'irctc',
    'indianrail': 'indianrail',
    'airindia': 'airindia',
    'raaga': 'raaga',
    'hungama': 'hungama',
    'wynk': 'wynk',
    'jio': 'jio',
    'airtel': 'airtel',
    'vi': 'vi',
    'bsnl': 'bsnl',
    'icicibank': 'icicibank',
    'hdfcbank': 'hdfcbank',
    'sbibank': 'sbibank',
    'axisbank': 'axisbank',
    'kotak': 'kotak',
    'yesbank': 'yesbank',
    'federalbank': 'federalbank',
    'ucobank': 'ucobank',
    'canarabank': 'canarabank',
    'iifl': 'iifl',
    'bajajfinserv': 'bajajfinserv',
    'tatacapital': 'tatacapital',
    'lendenclub': 'lendenclub',
    'groww': 'groww',
    'zerodha': 'zerodha',
    'upstox': 'upstox',
    'angelone': 'angelone',
    '5paisa': '5paisa',
    'dhan': 'dhan',
    'icicidirect': 'icicidirect',
    'sharekhan': 'sharekhan',
    'motilaloswal': 'motilaloswal',
    'choiceindia': 'choiceindia',
    'rkglobal': 'rkglobal',
    'indmoney': 'indmoney',
    'smallcase': 'smallcase',
    'fisdom': 'fisdom',
    'bajajbroking': 'bajajbroking',
    'hdfcsec': 'hdfcsec',
    'kfintech': 'kfintech',
    'nseindia': 'nseindia',
    'bseindia': 'bseindia',
    'mcxindia': 'mcxindia',
    'moneybhai': 'moneybhai',
    'tickertape': 'tickertape',
    'stockedge': 'stockedge',
    'screener': 'screener',
    'trendlyne': 'trendlyne',
    'tradingview': 'tradingview',
    'investing': 'investing',
    'yahoofinance': 'yahoofinance',
    'economictimes': 'economictimes',
    'livemint': 'livemint',
    'financialexpress': 'financialexpress',
    'businesstoday': 'businesstoday',
    'business-standard': 'businessstandard',
    'firstpost': 'firstpost',
    'scroll': 'scroll',
    'theprint': 'theprint',
    'opindia': 'opindia',
    'republicworld': 'republicworld',
    'news18': 'news18',
    'tv9telugu': 'tv9telugu',
    'sakshi': 'sakshi',
    'andhrajyothy': 'andhrajyothy',
    'manatelangana': 'manatelangana',
    'greatandhra': 'greatandhra',
    'idlebrain': 'idlebrain',
    '123telugu': '123telugu',
    'filmibeat': 'filmibeat',
    'bollywoodhungama': 'bollywoodhungama',
    'pinkvilla': 'pinkvilla',
    'koimoi': 'koimoi',
    'indiaforums': 'indiaforums',
    'tellychakkar': 'tellychakkar',
    'iwmbuzz': 'iwmbuzz',
    'gossipaddict': 'gossipaddict',
    'missmalini': 'missmalini',
    'jagran': 'jagran',
    'dainikbhaskar': 'dainikbhaskar',
    'amarujala': 'amarujala',
    'navbharattimes': 'navbharattimes',
    'abhiyaan': 'abhiyaan',
    'livehindustan': 'livehindustan',
    'jansatta': 'jansatta',
    'rajasthanpatrika': 'rajasthanpatrika',
    'bhaskar': 'bhaskar',
    'divyabhaskar': 'divyabhaskar',
    'inshorts': 'inshorts',
    'dailyhunt': 'dailyhunt',
    'newsdog': 'newsdog',
    'ucnews': 'ucnews',
    'operanews': 'operanews',
    'sharechat': 'sharechat',
    'mogo': 'mogo',
    'josh': 'josh',
    'public': 'public',
    'kooapp': 'kooapp',
    'mailyolo': 'mailyolo',
    'toffee': 'toffee',
    'voot': 'voot',
    'sonyliv': 'sonyliv',
    'zee5': 'zee5',
    'mxplayer': 'mxplayer',
    'erosnow': 'erosnow',
    'altbalaji': 'altbalaji',
    'hoichoi': 'hoichoi',
    'aha': 'aha',
    'perplexity': 'perplexity',
    'gemini': 'gemini',
    'grok': 'grok',
    'claude': 'claude'
        }
        text_l = text.lower()
        website = None
        query = None
        is_search_action = True  # Default to search
        
        # Step 0: Detect if it's just opening a website (no search action)
        # Patterns like: "open github", "go to netflix", "open perplexity"
        open_patterns = [
            r'\b(?:open|go\s+to|visit|browse)\s+(.+?)$',
        ]
        for pattern in open_patterns:
            match = re.search(pattern, text_l)
            if match:
                potential_website = match.group(1).strip()
                # Check if the potential website is in our websites list
                sorted_keywords = sorted(websites.keys(), key=len, reverse=True)
                for keyword in sorted_keywords:
                    word_pattern = rf'\b{re.escape(keyword)}\b'
                    if re.search(word_pattern, potential_website):
                        website = keyword
                        is_search_action = False  # Just opening, not searching
                        return website, None, is_search_action
        
        # Step 1: Extract website name from command using word boundaries
        # Sort by length (longest first) to match longer keywords before shorter ones
        # This prevents 'x' from matching inside 'perplexity'
        sorted_keywords = sorted(websites.keys(), key=len, reverse=True)
        for keyword in sorted_keywords:
            # Use word boundaries to match whole words only
            pattern = rf'\b{re.escape(keyword)}\b'
            if re.search(pattern, text_l):
                website = keyword
                break
        
        # Step 2: Extract search query - IMPROVED to exclude "for" keyword
        
        # Pattern 1: "search [query] on/for/at [website]"
        website_keywords_pattern = '|'.join(list(websites.keys()))
        pattern1 = rf'search\s+(.+?)\s+(?:on|for|at)\s+(?:{website_keywords_pattern})'
        match1 = re.search(pattern1, text_l)
        
        if match1:
            query = match1.group(1).strip()
        else:
            # Pattern 2: "[website] search [query]"
            pattern2 = rf'(?:{website_keywords_pattern})\s+(?:search|google)\s+(?:for\s+)?(.+?)$'
            match2 = re.search(pattern2, text_l)
            
            if match2:
                query = match2.group(1).strip()
            else:
                # Pattern 3: "search [query]" (generic search)
                pattern3 = r'search\s+(?:for\s+)?(.+?)(?:\s+on|\s+at|\s+in)?\s*$'
                match3 = re.search(pattern3, text_l)
                
                if match3:
                    query = match3.group(1).strip()
                    # Remove website names from end if present
                    for keyword in websites.keys():
                        query = re.sub(rf'\s+{keyword}\s*$', '', query)
                else:
                    # Pattern 4: Extract remaining words (fallback)
                    query_text = text_l
                    
                    # Remove action words and profile patterns
                    skip_patterns = [
                        r'with\s+chrome\s+profile\s+\w+',
                        r'chrome\s+profile\s+\w+',
                        r'with\s+profile\s+\w+',
                        r'use\s+profile\s+\w+',
                        r'profile\s+\w+',
                    ]
                    
                    for pattern in skip_patterns:
                        query_text = re.sub(pattern, '', query_text)
                    
                    # Remove structural keywords
                    skip_words = {
                        'search', 'open', 'go', 'to', 'on', 'in', 'at', 'for', 'and', 'the', 'a', 'an',
                        'with', 'use', 'profile', 'chrome', 'edge', 'browser', 'query'
                    }
                    skip_words.update(list(websites.keys()))
                    
                    query_words = [w for w in query_text.split() if w not in skip_words and w.strip()]
                    query = ' '.join(query_words) if query_words else None
        
        # Step 3: Clean up query - remove "for" if it was included
        if query:
            query = re.sub(r'^\s*for\s+', '', query)  # Remove "for" at the start
            query = re.sub(r'\s+for\s*$', '', query)  # Remove "for" at the end
        
        # Default to google if no website detected
        if not website:
            website = 'google'
        
        return website, query, is_search_action

    def _extract_app_name(self, words, trigger_words):
        for trigger in trigger_words:
            if trigger in words:
                idx = words.index(trigger)
                app_words = [w for w in words[idx+1:] if w not in ['app', 'application', 'program']]
                if app_words:
                    return ' '.join(app_words)
        return None

    def _extract_text_after_keywords(self, words, keywords, skip_words):
        for keyword in keywords:
            if keyword in words:
                idx = words.index(keyword)
                text_words = [w for w in words[idx+1:] if w not in skip_words]
                if text_words:
                    return ' '.join(text_words)
        return None

    def _extract_file_or_folder_path(self, words, raw_command):
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


def main():
    # show passcode dialog first
    app = QApplication(sys.argv)

    # Ensure passcode file exists (initializes default 1304 if not present)
    load_stored_passcode()

    # Quick face unlock attempt before showing passcode dialog
    unlocked_by_face = False
    try:
        fa = FaceAuthenticator(known_faces_dir=os.path.join(os.getcwd(), "known_faces"))
        res = fa.authenticate(camera_index=0, timeout=6.0, required_matches=2)
        if res:
            name, dist = res
            print(f"Face unlock success: {name} (dist={dist:.3f})")
            unlocked_by_face = True
    except Exception as e:
        # silent fallback to passcode if camera fails or face_auth import fails
        print(f"Face auth attempt failed: {e}")

    if not unlocked_by_face:
        passcode_dialog = PasscodeDialog()
        if passcode_dialog.exec() != QDialog.Accepted:
            sys.exit(0)

    ui = EvaGui()
    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
