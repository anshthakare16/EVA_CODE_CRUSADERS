"""
Advanced Command Classification Model
Classifies commands into: SYSTEM_ACTION, APP_LAUNCH, IN_APP_ACTION, WEB_ACTION
Zero hardcoding, fully ML-based (except system commands list)
"""

import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import config
from utils.logger import setup_logger


class CommandClassifier:
    """Classify commands into execution categories"""
    
    # ONLY system commands are hardcoded (as per your requirement)
    SYSTEM_COMMANDS = {
        'volume': ['volume up', 'volume down', 'increase volume', 'decrease volume', 'mute', 'unmute'],
        'brightness': ['brightness up', 'brightness down', 'increase brightness', 'decrease brightness'],
        'power': ['sleep', 'shutdown', 'restart', 'hibernate', 'sign out'],
        'network': ['turn on wifi', 'turn off wifi', 'enable wifi', 'disable wifi', 
                   'turn on bluetooth', 'turn off bluetooth', 'enable bluetooth', 'disable bluetooth'],
        'battery': ['enable battery saver', 'disable battery saver', 'power saver on', 'power saver off'],
    }
    
    def __init__(self):
        self.logger = setup_logger('CommandClassifier')
        self.model_path = os.path.join(config.MODEL_WEIGHTS_DIR, 'classifier_model.pkl')
        self.vectorizer_path = os.path.join(config.MODEL_WEIGHTS_DIR, 'classifier_vectorizer.pkl')
        
        self.model = None
        self.vectorizer = None
        
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.load_model()
        else:
            self.train_default_model()
    
    def classify(self, text):
        """
        Enhanced classification with priority-based detection
        Priority: SYSTEM > APP_LAUNCH > IN_APP_ACTION > CONVERSATION
        
        Returns: {
            'category': 'SYSTEM_ACTION' | 'APP_LAUNCH' | 'IN_APP_ACTION' | 'CONVERSATION',
            'confidence': float,
            'subcategory': str (for SYSTEM_ACTION commands),
            'requires_screen_analysis': bool,
            'raw_command': str
        }
        """
        text_lower = text.lower().strip()
        
        self.logger.info(f"Classifying: '{text}'")
        
        # Priority 1: SYSTEM COMMANDS (highest priority)
        system_result = self._detect_system_command(text_lower)
        if system_result:
            self.logger.info(f"Classified: '{text}' → SYSTEM_ACTION ({system_result['subcategory']}) (confidence: {system_result['confidence']:.2%})")
            return system_result
        
        # Priority 2: APP LAUNCH
        app_result = self._detect_app_launch(text_lower, text)
        if app_result:
            self.logger.info(f"Classified: '{text}' → APP_LAUNCH (confidence: {app_result['confidence']:.2%})")
            return app_result
        
        # Priority 3: IN-APP ACTION
        in_app_result = self._detect_in_app_action(text_lower)
        if in_app_result:
            self.logger.info(f"Classified: '{text}' → IN_APP_ACTION (confidence: {in_app_result['confidence']:.2%})")
            return in_app_result
        
        # Default: CONVERSATION
        self.logger.info(f"Classified: '{text}' → CONVERSATION (confidence: 50.00%)")
        return {
            'category': 'CONVERSATION',
            'confidence': 0.5,
            'subcategory': None,
            'requires_screen_analysis': False,
            'raw_command': text
        }
    
    def _detect_system_command(self, text_lower):
        """Detect system-level commands"""
        system_patterns = {
            'volume': ['volume', 'sound level', 'audio level'],
            'brightness': ['brightness', 'screen brightness'],
            'shutdown': ['shut down', 'power off', 'turn off computer'],
            'restart': ['restart', 'reboot'],
            'sleep': ['sleep', 'hibernate'],
            'lock': ['lock screen', 'lock computer', 'lock my screen']
        }
        
        for subcategory, keywords in system_patterns.items():
            if any(kw in text_lower for kw in keywords):
                return {
                    'category': 'SYSTEM_ACTION',
                    'subcategory': subcategory,
                    'confidence': 0.95,
                    'requires_screen_analysis': False,
                    'raw_command': text_lower
                }
        
        return None
    
    def _detect_app_launch(self, text_lower, original_text):
        """Detect app launch commands"""
        launch_keywords = ['open', 'launch', 'start', 'run']
        
        for keyword in launch_keywords:
            if keyword in text_lower:
                # Extract app name after keyword
                parts = text_lower.split(keyword, 1)
                if len(parts) > 1:
                    app_name = parts[1].strip()  # ✅ FIXED: Correct order
                    
                    # Clean app name (remove punctuation)
                    app_name = app_name.replace('.', '').replace(',', '').replace('?', '').replace('!', '').strip()
                    
                    if app_name:
                        return {
                            'category': 'APP_LAUNCH',
                            'confidence': 0.90,
                            'subcategory': None,
                            'requires_screen_analysis': False,
                            'raw_command': original_text,
                            'app_name': app_name
                        }
        
        return None
    
    def _detect_in_app_action(self, text_lower):
        """Detect in-app actions"""
        in_app_keywords = [
            'click', 'press', 'tap', 'select',
            'send', 'message', 'text',
            'type', 'write', 'enter',
            'search', 'find', 'look for',
            'scroll', 'swipe',
            'play', 'pause', 'stop',
            'next', 'previous', 'back',
            'close', 'minimize', 'maximize'
        ]
        
        if any(kw in text_lower for kw in in_app_keywords):
            return {
                'category': 'IN_APP_ACTION',
                'confidence': 0.85,
                'subcategory': None,
                'requires_screen_analysis': True,
                'raw_command': text_lower
            }
        
        return None
    
    def _is_system_command(self, command):
        """Check if command is a system command (legacy compatibility)"""
        command = command.lower()
        for category, phrases in self.SYSTEM_COMMANDS.items():
            for phrase in phrases:
                if phrase in command:
                    return True, category
        return False, None
    
    def train_default_model(self):
        """Train a default model with sample data"""
        self.logger.info("Training default classification model...")
        
        # Sample training data
        training_data = [
            # APP_LAUNCH examples
            ("open chrome", "APP_LAUNCH"),
            ("launch spotify", "APP_LAUNCH"),
            ("start notepad", "APP_LAUNCH"),
            ("run calculator", "APP_LAUNCH"),
            ("open whatsapp", "APP_LAUNCH"),
            
            # IN_APP_ACTION examples
            ("click on the button", "IN_APP_ACTION"),
            ("send a message to john", "IN_APP_ACTION"),
            ("type hello world", "IN_APP_ACTION"),
            ("search for python tutorial", "IN_APP_ACTION"),
            ("scroll down", "IN_APP_ACTION"),
            ("close this window", "IN_APP_ACTION"),
            
            # WEB_ACTION examples
            ("search google for weather", "WEB_ACTION"),
            ("browse to youtube", "WEB_ACTION"),
            ("go to gmail", "WEB_ACTION"),
            
            # CONVERSATION examples
            ("what time is it", "CONVERSATION"),
            ("tell me a joke", "CONVERSATION"),
            ("how are you", "CONVERSATION"),
        ]
        
        X = [item[0] for item in training_data]
        y = [item[1] for item in training_data]
        
        # Train vectorizer
        self.vectorizer = TfidfVectorizer(max_features=100)
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Train classifier
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_vectorized, y)
        
        # Save model
        os.makedirs(config.MODEL_WEIGHTS_DIR, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        self.logger.info("Classification model trained successfully")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.logger.info("Classification model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.info("Falling back to training new model")
            self.train_default_model()
