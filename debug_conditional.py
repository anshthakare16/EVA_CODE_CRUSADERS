import sys
import re

# Test the exact extraction logic from main.py

websites = {
    'youtube': 'youtube.com', 'google': 'google.com', 'gmail': 'mail.google.com',
    'facebook': 'facebook.com', 'twitter': 'twitter.com', 'amazon': 'amazon.com',
    'linkedin': 'linkedin.com', 'reddit': 'reddit.com', 'github': 'github.com',
    'stackoverflow': 'stackoverflow.com',
}

def _extract_website_and_action(text):
    text_l = text.lower().strip()
    
    # Extract website
    website = None
    for key, value in websites.items():
        if key in text_l:
            website = value
            break
    
    if not website:
        return None, None
    
    query = None
    
    # Step 1: Look for explicit "search <query>" pattern
    website_keywords = '|'.join(list(websites.keys()))
    search_pattern = rf'(?:search|query)\s+(?:for\s+)?(.+?)(?:\s+(?:on|at|in)\s+(?:{website_keywords}))?$'
    search_match = re.search(search_pattern, text_l)
    
    if search_match:
        query = search_match.group(1).strip()
        query = re.sub(r'\s+(?:on|at|in)$', '', query)
        website_list = list(websites.keys())
        query_words = [w for w in query.split() if w not in website_list]
        query = ' '.join(query_words) if query_words else None
    
    # Step 2: If no search keyword found, try to extract remaining words
    if not query:
        query_text = text_l
        
        profile_patterns = [
            r'with\s+chrome\s+profile\s+\w+',
            r'chrome\s+profile\s+\w+',
            r'with\s+profile\s+\w+',
            r'use\s+profile\s+\w+',
            r'profile\s+\w+',
        ]
        
        for pattern in profile_patterns:
            query_text = re.sub(pattern, '', query_text)
        
        skip_words = {'with', 'chrome', 'for', 'open', 'go', 'to', 'on', 'in', 'and', 'profile', 'use', 'the', 'a', 'an', 'at', 'search', 'query'}
        skip_words.update(list(websites.keys()))
        
        query_words = [w for w in query_text.split() if w not in skip_words and w.strip()]
        query = ' '.join(query_words) if query_words else None
    
    return website, query

# Simulate the conditional logic
STEP_TEMPLATES = {
    "chrome_with_profile": [
        {"action_type": "SCREEN_ANALYSIS", "parameters": {"profile_name": "{profile_name}"}, "description": "Select profile"},
    ],
    "navigate_to_website": [
        {"action_type": "NAVIGATE_URL", "parameters": {"url": "{website}"}, "description": "Navigate to {website}"},
    ],
    "search_on_page": [
        {"action_type": "PRESS_KEY", "parameters": {"key": "ctrl+f"}, "description": "Open search"},
        {"action_type": "TYPE_TEXT", "parameters": {"text": "{search_query}"}, "description": "Type: {search_query}"},
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

# Test cases
test_commands = [
    "open youtube",
    "search python on google",
]

for cmd in test_commands:
    print(f"\n{'='*70}")
    print(f"Command: '{cmd}'")
    print('='*70)
    
    # Extract
    website, query = _extract_website_and_action(cmd)
    print(f"Extracted: website='{website}', query='{query}'")
    
    # Simulate step generation
    extracted_keywords = {
        'website': website,
        'search_query': query,
        'profile_name': 'Default'
    }
    
    steps_template = MODEL2_STEP_RULES["WEB_SEARCH"]
    generated_steps = []
    
    print(f"\nStep Template has {len(steps_template)} items")
    
    for i, step in enumerate(steps_template):
        print(f"\nProcessing step {i+1}/{len(steps_template)}: {step.get('action_type')}")
        
        if step.get("action_type") == "CONDITIONAL":
            condition = step["parameters"].get("condition")
            print(f"  -> This is a CONDITIONAL with condition: '{condition}'")
            
            if condition == "search_query_exists":
                search_query_value = extracted_keywords.get('search_query')
                print(f"  -> Checking: search_query = '{search_query_value}'")
                
                if not search_query_value:
                    print(f"  -> BREAKING - no search query")
                    break
                else:
                    print(f"  -> CONTINUING - search query exists")
                    continue
            continue
        
        print(f"  -> Adding {step['action_type']} step")
        generated_steps.append(step)
    
    print(f"\n✅ Generated {len(generated_steps)} steps:")
    for j, step in enumerate(generated_steps, 1):
        print(f"  {j}. {step['action_type']}: {step.get('description', 'N/A')}")
