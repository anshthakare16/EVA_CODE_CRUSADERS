"""Text-to-speech for EVA responses"""

import pyttsx3
from utils.logger import setup_logger

class TextToSpeech:
    """Convert text to speech"""
    
    def __init__(self):
        self.logger = setup_logger('TextToSpeech')
        self.engine = pyttsx3.init()
        
        # Configure voice properties
        voices = self.engine.getProperty('voices')
        # Set female voice if available (JARVIS-like)
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        self.engine.setProperty('rate', 175)  # Speed
        self.engine.setProperty('volume', 0.9)  # Volume
        
        self.logger.info("Text-to-speech initialized")
    
    def speak(self, text):
        """Speak the given text"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
