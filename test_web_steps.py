import re

# Simulate the step rules
STEP_TEMPLATES = {
    "chrome_with_profile": [
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"profile_name": "{profile_name}"}, "description": "Select profile: {profile_name}"},
    ],
    "navigate_to_website": [
        {"action_type": "NAVIGATE_URL", "parameters": {"url": "{website}"}, "description": "Navigate to {website}"},
        {"action_type": "WAIT", "parameters": {"duration": 2}, "description": "Wait for page load"},
    ],
    "search_on_page": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+f"}, "description": "Open search"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{search_query}"}, "description": "Type: {search_query}"},
        {"action_type": "PRESS_KEY", "parameters": {"key": "enter"}, "description": "Execute search"},
    ],
}

MODEL2_STEP_RULES = {
    "WEB_SEARCH": [
        *STEP_TEMPLATES["chrome_with_profile"],
        *STEP_TEMPLATES["navigate_to_website"],
        {"action_type": "CONDITIONAL", "parameters": {"condition": "search_query_exists"}, "description": "Check search needed"},
        *STEP_TEMPLATES["search_on_page"],
    ],
}

def generate_steps(command_type, extracted_keywords):
    steps_template = MODEL2_STEP_RULES[command_type]
    generated_steps = []
    
    for step in steps_template:
        if step.get("action_type") == "CONDITIONAL":
            condition = step["parameters"].get("condition")
            if condition == "search_query_exists":
                if not extracted_keywords.get('search_query'):
                    print(f"  -> CONDITIONAL: No search query found, breaking")
                    break
                else:
                    print(f"  -> CONDITIONAL: Search query exists, continuing")
                    continue
            continue
    
        step_copy = dict(step)
        replacements = {
            "{profile_name}": extracted_keywords.get('profile_name', 'Default'), 
            "{website}": extracted_keywords.get('website', ''),
            "{search_query}": extracted_keywords.get('search_query', ''),
        }
        
        # Replace in description
        if "description" in step_copy:
            for old, new in replacements.items():
                if new is not None:  # Only replace if value exists
                    step_copy["description"] = step_copy["description"].replace(old, str(new))
        
        # Replace in parameters
        if "parameters" in step_copy:
            for key, value in step_copy["parameters"].items():
                for old, new in replacements.items():
                    if isinstance(value, str) and new is not None:
                        step_copy["parameters"][key] = value.replace(old, str(new))
        
        generated_steps.append(step_copy)
    
    return generated_steps

# Test cases
test_cases = [
    {
        "name": "Web search WITH query",
        "keywords": {
            "profile_name": "work",
            "website": "google.com",
            "search_query": "python tutorials"
        }
    },
    {
        "name": "Web search WITHOUT query",
        "keywords": {
            "profile_name": "Default",
            "website": "youtube.com",
            "search_query": None
        }
    },
]

for test in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {test['name']}")
    print(f"Keywords: {test['keywords']}")
    print(f"{'='*60}")
    
    steps = generate_steps("WEB_SEARCH", test['keywords'])
    
    print(f"\nGenerated Steps ({len(steps)} total):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step['action_type']}: {step['description']}")
