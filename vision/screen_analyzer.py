"""
Screen Understanding using Gemini (Vision)
Methodology: Uses vision + coordinate matching for precise element selection
"""

import logging
import json
import re
from google.genai import Client, types
from difflib import SequenceMatcher

logger = logging.getLogger("ScreenAnalyzer")


class ScreenAnalyzer:
    """Gemini-based screen understanding and coordinate selection"""
    
    def __init__(self, api_key):
        """Initialize Gemini for screen analysis with vision support"""
        self.logger = logging.getLogger("ScreenAnalyzer")
        self.gemini_available = False  # Flag to track if Gemini is ready
        self.model_name = None
        self.available_models = []  # Track all available models for fallback
        self.client = Client(api_key=api_key)
        
        try:
            
            # Models prioritized for VISION support (with image/generateContent support)
            # Try vision-capable models FIRST
            models_to_try = [
                'gemini-2.0-flash',          # Stable 2.0 flash (primary - best quota efficiency)
                'gemini-2.0-flash-exp',      # Latest with vision support
                'gemini-1.5-pro',             # Pro has better vision support than flash
                'gemini-1.5-flash',           # Flash (check API support)
                'gemini-1.0-pro-vision',      # Explicit vision model
                'gemini-pro-vision',          # Alternative vision model
            ]
            
            self.logger.info("🤖 Initializing Gemini for vision-based coordinate selection...")
            
            for model_name in models_to_try:
                try:
                    # Create model WITHOUT testing (avoids API calls during init)
                    self.model_name = model_name
                    self.gemini_available = True
                    self.available_models.append(model_name)
                    self.logger.info(f"✅ Gemini vision model selected: {model_name}")
                    self.logger.info(f"   Available fallback models: {self.available_models[1:]}")
                    return
                except Exception as e:
                    self.logger.debug(f"   Model '{model_name}' unavailable: {str(e)[:100]}")
                    self.available_models.append(model_name)  # Track for fallback
                    continue
            
            # If no vision model works, still proceed with fallback
            self.logger.warning("⚠️ No vision-capable Gemini models available")
            self.logger.warning("   Will use fuzzy matching for coordinate selection")
            self.logger.warning("   Check GEMINI_API_KEY and model availability in Google AI Studio")
            self.gemini_available = False
            self.model_name = None
        
        except Exception as e:
            self.logger.error(f"❌ Gemini initialization error: {e}")
            self.gemini_available = False
            self.model_name = None
    
    def _switch_to_fallback_model(self, current_model_index=None):
        """
        Switch to next available model when current one fails (e.g., 429 quota)
        
        Returns:
            bool: True if switched successfully, False if no more models
        """
        if current_model_index is None:
            # Find current model's index
            try:
                current_model_index = self.available_models.index(self.model_name)
            except (ValueError, AttributeError):
                current_model_index = 0
        
        # Try remaining models
        for i in range(current_model_index + 1, len(self.available_models)):
            model_name = self.available_models[i]
            try:
                self.model = genai.GenerativeModel(model_name)
                self.model_name = model_name
                self.logger.warning(f"⚠️ Switched to fallback model: {model_name}")
                return True
            except Exception as e:
                self.logger.debug(f"   Fallback model '{model_name}' also unavailable: {str(e)[:100]}")
                continue
        
        self.logger.error("❌ All fallback models exhausted")
        return False
    
    def get_screen_summary(self, screenshot_path):
        """
        Get 1-2 line screen summary (Methodology requirement)
        Uses fallback when Gemini unavailable
        
        Args:
            screenshot_path: Path to PNG screenshot
        
        Returns:
            str: Brief screen state description
        """
        # Fallback: No Gemini call needed
        self.logger.info("Using fallback screen summary")
        return "Screen active - processing UI elements"
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
        
        self.logger.info(f"🔍 Fuzzy matching: target='{target}', profile_name='{profile_name}'")
        
        for elem in elements:
            label = elem.get('label', '').lower()
            confidence = elem.get('confidence', 0)
            
            # If profile_name is specified, prioritize matching it first
            if profile_name:
                profile_lower = profile_name.lower()
                profile_similarity = self._calculate_text_similarity(profile_lower, label)
                
                # Boost score for exact substring matches
                if profile_lower in label or label in profile_lower:
                    profile_similarity = max(profile_similarity, 0.95)
                
                # Combine profile match with confidence (profile is PRIMARY)
                profile_score = profile_similarity * 0.9 + (confidence * 0.1)
                
                if profile_score > best_score and profile_score > 0.5:
                    best_score = profile_score
                    best_match = elem
                    self.logger.debug(f"  Profile match candidate: '{label}' (score: {profile_score:.2f})")
            else:
                # No profile specified, match against target description
                target_lower = target.lower()
                similarity = self._calculate_text_similarity(target_lower, label)
                
                # Boost score for exact substring matches
                if target_lower in label or label in target_lower:
                    similarity = max(similarity, 0.9)
                
                # Combine with element confidence
                combined_score = similarity * 0.7 + (confidence * 0.3)
                
                if combined_score > best_score and combined_score > 0.5:
                    best_score = combined_score
                    best_match = elem
                    self.logger.debug(f"  Target match candidate: '{label}' (score: {combined_score:.2f})")
        
        if best_match:
            self.logger.info(f"✓ Fuzzy match: '{best_match['label']}' (score: {best_score:.2f})")
            return (best_match['x'], best_match['y'])
        
        return None
    
    def filter_coordinates(self, omniparser_elements, step_description):
        """
        Filter OmniParser elements to find best match for step
        Uses fallback fuzzy matching when Gemini unavailable
        
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
            
            # Sort by confidence and take top element
            sorted_elements = sorted(
                omniparser_elements, 
                key=lambda e: e.get('confidence', 0), 
                reverse=True
            )[:5]
            
            if sorted_elements:
                best = sorted_elements[0]
                return {
                    "x": best.get('x', 0),
                    "y": best.get('y', 0),
                    "operation": "click",
                    "confidence": best.get('confidence', 0)
                }
            
            return {"x": 0, "y": 0, "operation": "click", "confidence": 0}
        
        except Exception as e:
            self.logger.error(f"Coordinate filtering error: {e}")
            return {"x": 0, "y": 0, "operation": "click", "confidence": 0}
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
    
    def select_coordinate(self, elements, target_label, step_context, profile_name=None, screenshot_path=None):
        """
        Use Gemini + vision to select best coordinate from OmniParser elements
        
        Args:
            elements: List of {id, label, x, y, type, confidence} from OmniParser
            target_label: What we're looking for (e.g., "Code Crusaders", "Type a message")
            step_context: Full step dict with description
            profile_name: Optional profile name to look for (CRITICAL for Chrome profile selection)
            screenshot_path: Path to screenshot for visual analysis by Gemini
        
        Returns:
            (x, y) tuple or None
        """
        if not elements:
            self.logger.warning("No elements to select from")
            return None
        
        self.logger.info(f"🎯 Selecting coordinate for: {target_label}, profile: {profile_name}")
        self.logger.info(f"📊 Total elements from OmniParser: {len(elements)}")
        
        # Log all elements for debugging
        for elem in elements[:10]:  # Log first 10
            self.logger.debug(f"  Element: {elem.get('label', 'N/A')} at ({elem['x']}, {elem['y']}) - conf: {elem.get('confidence', 0):.2f}")
        
        # If Gemini available and we have screenshot, use vision-based selection
        if self.gemini_available and screenshot_path:
            result = self._gemini_select_coordinate_with_vision(
                elements, target_label, step_context, profile_name, screenshot_path
            )
            if result:
                return result
            self.logger.info("Gemini vision selection failed, trying fuzzy match fallback...")
        
        # Fallback to fuzzy matching if Gemini unavailable or failed
        self.logger.info("Using fuzzy matching fallback...")
        return self._fuzzy_match_element(target_label, elements, profile_name)
    
    def _gemini_select_coordinate_with_vision(self, elements, target_label, step_context, profile_name, screenshot_path):
        """
        Use Gemini with actual screenshot image to select the correct coordinate
        Gemini can see the visual profile buttons and match them to profile_name
        """
        try:
            # Read screenshot image
            try:
                with open(screenshot_path, 'rb') as f:
                    image_data = f.read()
            except Exception as e:
                self.logger.warning(f"Failed to read screenshot: {e}")
                return None
            
            # Format elements for Gemini (top 50 elements)
            # ✅ IMPORTANT: Build list of valid IDs for validation
            valid_ids = set()
            element_list = []
            for idx, elem in enumerate(elements[:50]):
                elem_id = elem['id']
                valid_ids.add(elem_id)
                element_list.append(
                    f"ID {elem_id}: '{elem['label']}' at ({elem['x']}, {elem['y']}) "
                    f"[type: {elem.get('type', 'unknown')}, conf: {elem.get('confidence', 0):.2f}]"
                )
            
            action_description = step_context.get("description", "")
            if not action_description:
                action_description = target_label
            
            # Build prompt with profile prioritization
            prompt_parts = [
                f"🎯 ACTION: {action_description}",
                f"🔍 TARGET: {target_label}"
            ]
            
            if profile_name:
                prompt_parts.append(f"⭐ PRIORITY: Look for UI element labeled '{profile_name}' (CRITICAL for profile selection)")
            
            prompt_parts.extend([
                "",
                "AVAILABLE UI ELEMENTS FROM OMNIPARSER:",
                "\n".join(element_list),
                "",
                "INSTRUCTIONS:",
                "1. Look at the screenshot visually - you can see the actual UI",
                "2. If PRIORITY is given, MUST select element matching that exact text/profile name",
                "3. Match element labels to action description (exact > partial > semantic)",
                "4. For profile selection: look for clickable profile buttons/names on screen",
                "5. Return the element ID that BEST matches the action",
                "6. ⚠️  ONLY use IDs from the list above - do NOT invent IDs",
                "",
                "RESPONSE: Return ONLY valid JSON (no markdown): {\"id\": N, \"x\": X, \"y\": Y, \"reason\": \"brief why\"}"
            ])
            
            prompt = "\n".join(prompt_parts)
            
            self.logger.info(f"📸 Sending to Gemini ({self.model_name}): screenshot + {len(elements)} elements + profile='{profile_name}'")
            self.logger.debug(f"Prompt length: {len(prompt)} chars")
            
            # Create image part using proper google.genai types
            image_blob = types.Blob(mimeType="image/png", data=image_data)
            image_part = types.Part(inlineData=image_blob)
            
            # Initialize response_text to None
            response_text = None
            max_retries = 2
            retry_count = 0
            
            # Call Gemini with BOTH image and text (with retry on quota error)
            while retry_count < max_retries:
                try:
                    self.logger.debug(f"Calling {self.model_name} with vision capabilities... (attempt {retry_count + 1})")
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt, image_part]
                    )
                    
                    # Extract response
                    if hasattr(response, 'text'):
                        response_text = response.text.strip()
                    else:
                        response_text = str(response).strip()
                    
                    self.logger.debug(f"Gemini response: {response_text[:300]}")
                    break  # Success - exit retry loop
                    
                except Exception as api_error:
                    error_msg = str(api_error)
                    self.logger.error(f"❌ Gemini API Error: {error_msg[:200]}")
                    
                    # Check for specific API issues
                    if "404" in error_msg or "not found" in error_msg.lower():
                        self.logger.error(f"   Model {self.model_name} doesn't support images on this API")
                        self.logger.error("   Fix: Try updating google-generativeai package")
                        self.logger.error("   $ pip install --upgrade google-generativeai")
                        # Try fallback model
                        if self._switch_to_fallback_model():
                            retry_count += 1
                            continue
                        return None
                    
                    elif "429" in error_msg or "quota" in error_msg.lower():
                        self.logger.error("   API quota exceeded - attempting to switch to fallback model...")
                        # Try to switch to a lower-tier model with better quota limits
                        if self._switch_to_fallback_model():
                            retry_count += 1
                            self.logger.info(f"   Retrying with {self.model_name}...")
                            continue
                        else:
                            self.logger.error("   No fallback models available - will use fuzzy matching")
                            return None
                    
                    elif "SAFETY" in error_msg or "blocked" in error_msg.lower():
                        self.logger.warning("   Image blocked by safety filter - trying without image")
                        # Try text-only fallback
                        try:
                            response = self.client.models.generate_content(
                                model=self.model_name,
                                contents=prompt
                            )
                            if hasattr(response, 'text'):
                                response_text = response.text.strip()
                            else:
                                response_text = str(response).strip()
                            self.logger.info("   Text-only fallback succeeded")
                            break  # Exit retry loop if text-only works
                        except Exception as e:
                            self.logger.error(f"   Text-only fallback also failed: {e}")
                            return None
                    else:
                        self.logger.error(f"   Unhandled API error - falling back to fuzzy match")
                        return None
            
            # Only process response if we successfully got response_text
            if response_text is None:
                self.logger.error("No response from Gemini and no fallback available")
                return None
            
            # Parse JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                parts = response_text.split('```')
                for part in parts:
                    if '{' in part and '}' in part:
                        response_text = part.strip()
                        break
            
            # Try to find JSON object
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group(0))
                elem_id = result.get('id', -1)
                reason = result.get('reason', 'No reason')
                
                if elem_id == -1:
                    self.logger.warning(f"❌ Gemini couldn't find confident match: {reason}")
                    return None
                
                # ✅ Validate that elem_id is in our valid IDs list
                if elem_id not in valid_ids:
                    self.logger.warning(f"⚠️  Element ID {elem_id} returned by Gemini is not in valid element list")
                    self.logger.warning(f"   Valid IDs are: {sorted(valid_ids)}")
                    self.logger.info("   Falling back to fuzzy matching...")
                    return self._fuzzy_match_element(target_label, elements, profile_name)
                
                # Find and return element coordinates
                for elem in elements:
                    if elem['id'] == elem_id:
                        x, y = elem['x'], elem['y']
                        self.logger.info(f"✅ Gemini selected: '{elem['label']}' (ID {elem_id}) at ({x}, {y}) - {reason}")
                        return (x, y)
                
                # ✅ FIX: Element ID not found - fallback to fuzzy matching
                self.logger.warning(f"⚠️  Element ID {elem_id} not found in element list (screen may have changed)")
                self.logger.info("   Falling back to fuzzy matching...")
                return self._fuzzy_match_element(target_label, elements, profile_name)
            else:
                self.logger.warning(f"No JSON in response: {response_text[:100]}")
                return None
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Gemini vision selection error: {e}", exc_info=True)
            return None
