"""
Screen Understanding using Gemini (Vision)
Methodology: Produces 1-2 line screen summary for step generation
"""

import logging
import json
import re
import google.generativeai as genai
from difflib import SequenceMatcher

logger = logging.getLogger("ScreenAnalyzer")


class ScreenAnalyzer:
    """Gemini-based screen understanding and coordinate selection"""
    
    def __init__(self, api_key):
        """Initialize Gemini for screen analysis"""
        self.logger = logging.getLogger("ScreenAnalyzer")
        
        try:
            genai.configure(api_key=api_key)
            
            # ✅ Try models in order - NO TEST CALL
            models_to_try = [
                'gemini-2.0-flash-exp',
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-pro'
            ]
            
            self.model = None
            for model_name in models_to_try:
                try:
                    # Just create the model object, don't test it
                    test_model = genai.GenerativeModel(model_name)
                    self.model = test_model
                    self.logger.info(f"✓ Gemini initialized: {model_name}")
                    break
                except Exception as e:
                    self.logger.debug(f"Model {model_name} unavailable: {e}")
                    continue
            
            if not self.model:
                self.logger.error("❌ NO GEMINI MODELS AVAILABLE - Check:")
                self.logger.error("1. GEMINI_API_KEY is valid in .env")
                self.logger.error("2. API key has quota remaining")
                self.logger.error("3. Models are enabled in Google AI Studio")
                raise Exception("No Gemini model available - check GEMINI_API_KEY")
        
        except Exception as e:
            self.logger.error(f"❌ Gemini initialization failed: {e}")
            raise
    
    def get_screen_summary(self, screenshot_path):
        """
        Get 1-2 line screen summary (Methodology requirement)
        
        Args:
            screenshot_path: Path to PNG screenshot
        
        Returns:
            str: Brief screen state description
        """
        try:
            with open(screenshot_path, 'rb') as f:
                image_data = f.read()
            
            image_part = {
                "mime_type": "image/png",
                "data": image_data
            }
            
            prompt = """Analyze this screenshot and provide a concise 1-2 line summary describing:

1. What application is open
2. Current screen state
3. Visible UI elements

Keep it brief and factual. Example: "Chrome browser is open showing YouTube homepage with search bar visible at top."

Summary:"""
            
            self.logger.info("Requesting screen summary from Gemini...")
            response = self.model.generate_content([prompt, image_part])
            
            # Extract text safely
            if hasattr(response, 'text'):
                summary = response.text.strip()
            else:
                summary = str(response).strip()
            
            self.logger.info(f"Screen summary: {summary[:100]}...")
            return summary
        
        except Exception as e:
            self.logger.error(f"Screen summary error: {e}")
            return "Unable to analyze screen"
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text strings (0-1)"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _fuzzy_match_element(self, target, elements, profile_name=None):
        """
        Fallback: Use fuzzy matching to find best element
        
        Returns:
            tuple: (x, y) or None
        """
        best_match = None
        best_score = 0.0
        
        # Prioritize profile_name if provided
        search_terms = []
        if profile_name:
            search_terms.append(profile_name.lower())
        search_terms.append(target.lower())
        
        for elem in elements:
            label = elem.get('label', '').lower()
            confidence = elem.get('confidence', 0)
            
            for term in search_terms:
                # Calculate similarity score
                similarity = self._calculate_text_similarity(term, label)
                
                # Boost score for exact substring matches
                if term in label or label in term:
                    similarity = max(similarity, 0.9)
                
                # Combine with element confidence
                combined_score = similarity * 0.7 + (confidence * 0.3)
                
                if combined_score > best_score and combined_score > 0.5:
                    best_score = combined_score
                    best_match = elem
        
        if best_match:
            self.logger.info(f"✓ Fuzzy match: '{best_match['label']}' (score: {best_score:.2f})")
            return (best_match['x'], best_match['y'])
        
        return None
    
    def filter_coordinates(self, omniparser_elements, step_description):
        """
        Filter OmniParser elements to find best match for step
        
        Methodology: "Gemini filters the OmniParser UI element list"
        
        Args:
            omniparser_elements: List of UI elements from OmniParser
            step_description: Current step description
        
        Returns:
            dict: {"x": int, "y": int, "operation": str, "confidence": float}
        """
        try:
            if not omniparser_elements:
                self.logger.warning("No elements to filter")
                return {"x": 0, "y": 0, "operation": "click", "confidence": 0}
            
            # Simplify elements for Gemini (top 30 by confidence)
            sorted_elements = sorted(
                omniparser_elements, 
                key=lambda e: e.get('confidence', 0), 
                reverse=True
            )[:30]
            
            simplified = []
            for e in sorted_elements:
                simplified.append({
                    'id': e.get('id'),
                    'label': e.get('label', ''),
                    'x': e.get('x'),
                    'y': e.get('y'),
                    'type': e.get('type', 'unknown'),
                    'confidence': round(e.get('confidence', 0), 2)
                })
            
            prompt = f"""You are filtering UI elements to execute this step: "{step_description}"

Available elements (sorted by confidence):
{json.dumps(simplified, indent=2)}

Return ONLY a valid JSON object with this exact structure:
{{
  "element_id": 5,
  "x": 100,
  "y": 200,
  "operation": "click",
  "confidence": 85
}}

Valid operations: click, double_click, right_click

Choose the element that best matches the step description. Consider:
- Label text similarity
- Element type appropriateness
- Element confidence score

JSON:"""
            
            self.logger.info(f"Filtering coordinates for: {step_description}")
            response = self.model.generate_content(prompt)
            
            # Extract and parse JSON safely
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            else:
                response_text = str(response).strip()
            
            # ✅ FIX: Handle markdown JSON properly
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                parts = response_text.split('```')
                for part in parts:
                    if '{' in part and '}' in part:
                        response_text = part.strip()
                        break
            
            # Extract JSON object
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Validate result structure
                if all(k in result for k in ['x', 'y', 'operation']):
                    self.logger.info(f"Selected: ({result.get('x')}, {result.get('y')}) confidence: {result.get('confidence', 0)}%")
                    return result
                else:
                    self.logger.warning(f"Invalid JSON structure: {result}")
                    return {"x": 0, "y": 0, "operation": "click", "confidence": 0}
            else:
                self.logger.warning("No JSON found in response")
                return {"x": 0, "y": 0, "operation": "click", "confidence": 0}
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return {"x": 0, "y": 0, "operation": "click", "confidence": 0}
        except Exception as e:
            self.logger.error(f"Coordinate filtering error: {e}")
            return {"x": 0, "y": 0, "operation": "click", "confidence": 0}
    
    def select_coordinate(self, elements, target_label, step_context, profile_name=None):
        """
        Use Gemini to select best coordinate from OmniParser elements
        
        Args:
            elements: List of {id, label, x, y, type, confidence} from OmniParser
            target_label: What we're looking for (e.g., "Code Crusaders", "Type a message")
            step_context: Full step dict with description
            profile_name: Optional profile name to look for.
        
        Returns:
            (x, y) tuple or None
        """
        if not elements:
            self.logger.warning("No elements to select from")
            return None
        
        # Sort elements by confidence for better selection
        sorted_elements = sorted(
            elements, 
            key=lambda e: e.get('confidence', 0), 
            reverse=True
        )
        
        # Format elements for Gemini (top 50 elements)
        element_list = []
        for idx, elem in enumerate(sorted_elements[:50]):
            element_list.append(
                f"{elem['id']}: '{elem['label']}' at ({elem['x']}, {elem['y']}) "
                f"[type: {elem.get('type', 'unknown')}, conf: {elem.get('confidence', 0):.2f}]"
            )
        
        action_description = step_context.get("description", "")
        if not action_description:
            action_description = target_label
        
        prompt_parts = [f'ACTION: "{action_description}"']
        
        if profile_name:
            prompt_parts.append(f'PROFILE_NAME: "{profile_name}" (PRIORITIZE THIS)')

        prompt = f"""You are a UI element selector for a Windows automation assistant.

TASK: Select the best UI element to click for this action.

{chr(10).join(prompt_parts)}

AVAILABLE UI ELEMENTS (sorted by confidence):
{chr(10).join(element_list)}

SELECTION RULES (in priority order):
1. If PROFILE_NAME is provided, PRIORITIZE exact matches for it over everything else
2. Match element labels to the action description (exact > partial > semantic)
3. Prefer elements with appropriate types (text, button, input) for the action
4. Prefer higher confidence scores when multiple matches exist
5. For text input actions, prefer elements with type 'text' or 'input'
6. For button clicks, prefer elements with type 'button' or clickable text

RESPONSE FORMAT:
Return ONLY a JSON object: {{"id": N, "reason": "brief explanation"}}
- If no confident match found, return {{"id": -1, "reason": "No confident match"}}
- Do NOT include markdown formatting or code blocks
- Do NOT add any explanation outside the JSON

Examples:
- Target "Code Crusaders" + PROFILE_NAME "Code Crusaders" → Select element with exact "Code Crusaders" label
- Target "Type a message" → Select text/input element
- Target "Send button" → Select button element with "Send"

JSON:"""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Extract text safely
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            else:
                response_text = str(response).strip()
            
            self.logger.debug(f"Gemini response: {response_text[:200]}")
            
            # ✅ FIX: Handle markdown JSON properly
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                parts = response_text.split('```')
                for part in parts:
                    if '{' in part and '}' in part:
                        response_text = part.strip()
                        break
            
            # Try to find JSON object (non-greedy match)
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group(0))
                elem_id = result.get('id', -1)
                reason = result.get('reason', 'No reason provided')
                
                if elem_id == -1:
                    self.logger.warning(f"Gemini couldn't find match: {reason}")
                    # Try fuzzy matching as fallback
                    self.logger.info("Attempting fuzzy match fallback...")
                    return self._fuzzy_match_element(target_label, elements, profile_name)
                
                # Find element by id in original list
                for elem in elements:
                    if elem['id'] == elem_id:
                        x, y = elem['x'], elem['y']
                        self.logger.info(f"✓ Selected: '{elem['label']}' at ({x}, {y}) - {reason}")
                        return (x, y)
                
                self.logger.warning(f"Element ID {elem_id} not found in element list")
                # Try fuzzy matching as fallback
                return self._fuzzy_match_element(target_label, elements, profile_name)
            else:
                self.logger.warning(f"No JSON found in response: {response_text[:100]}")
                # Try fuzzy matching as fallback
                return self._fuzzy_match_element(target_label, elements, profile_name)
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.info("Attempting fuzzy match fallback...")
            return self._fuzzy_match_element(target_label, elements, profile_name)
        except Exception as e:
            self.logger.error(f"Coordinate selection error: {e}", exc_info=True)
            self.logger.info("Attempting fuzzy match fallback...")
            return self._fuzzy_match_element(target_label, elements, profile_name)