"""
Command Processor - Uses Gemini for classification
WITH QUOTA HANDLING
"""

import logging
import json
import google.generativeai as genai
import time

logger = logging.getLogger("CommandProcessor")

class CommandProcessor:
    """Process commands using Gemini"""
    
    def __init__(self, api_key):
        """Initialize with Gemini"""
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            raise
        
        models_to_try = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-pro',
            'gemini-1.5-flash',
        ]
        
        self.model = None
        for model_name in models_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                self.model = test_model
                logger.info(f"✓ CommandProcessor initialized with: {model_name}")
                break
            except Exception as e:
                logger.debug(f"Model {model_name} unavailable: {e}")
                continue
        
        if not self.model:
            raise RuntimeError("No Gemini models available!")
    
    def process(self, text):
        """Process command using Gemini with retry on quota"""
        if not text or len(text.strip()) < 2:
            raise Exception("Text too short")
        
        prompt = f"""You are a command classifier for a voice assistant.

Analyze this command and classify it into ONE category.

COMMAND: "{text}"

Return ONLY JSON (no explanation):
{{
  "category": "SYSTEM_ACTION | APP_LAUNCH | IN_APP_ACTION | WEB_ACTION",
  "action": "specific action",
  "confidence": 95,
  "entities": {{
    "app_name": "if app launch",
    "parameter": "if system action"
  }}
}}

Examples:
- "open notepad" → category: APP_LAUNCH, action: launch, entities: {{"app_name": "notepad"}}
- "set volume to 50" → category: SYSTEM_ACTION, action: set_volume, entities: {{"parameter": "50"}}
- "send hi to john on whatsapp" → category: IN_APP_ACTION, action: send_message, entities: {{"recipient": "john", "message": "hi"}}
- "search for python tutorials" → category: WEB_ACTION, action: search, entities: {{"query": "python tutorials"}}

JSON:"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            
            result = json.loads(json_str)
            logger.info(f"✓ Classification: {result['category']} (conf: {result['confidence']}%)")
            
            return result
        
        except Exception as e:
            error_str = str(e)
            
            # ✅ QUOTA EXCEEDED - Return fallback classification
            if "429" in error_str or "quota" in error_str.lower():
                logger.warning(f"⚠️ QUOTA EXCEEDED - Using fallback classification")
                logger.warning(f"Retry after: Check error message for retry timing")
                
                # Return safe fallback based on keywords
                text_lower = text.lower()
                
                if any(word in text_lower for word in ['open', 'launch', 'start']):
                    return {
                        "category": "APP_LAUNCH",
                        "action": "launch",
                        "confidence": 50,
                        "entities": {"app_name": "unknown"}
                    }
                elif any(word in text_lower for word in ['search', 'google', 'bing']):
                    return {
                        "category": "WEB_ACTION",
                        "action": "search",
                        "confidence": 50,
                        "entities": {"query": text}
                    }
                elif any(word in text_lower for word in ['send', 'message', 'whatsapp']):
                    return {
                        "category": "IN_APP_ACTION",
                        "action": "send_message",
                        "confidence": 50,
                        "entities": {}
                    }
                else:
                    return {
                        "category": "IN_APP_ACTION",
                        "action": "unknown",
                        "confidence": 30,
                        "entities": {}
                    }
            
            # Any other error - raise it
            logger.error(f"Classification error: {e}")
            raise
