import re

def _extract_website_and_action(text):
    websites = {
        'youtube': 'youtube.com', 'google': 'google.com', 'gmail': 'mail.google.com',
        'facebook': 'facebook.com', 'twitter': 'twitter.com', 'amazon': 'amazon.com',
        'linkedin': 'linkedin.com', 'reddit': 'reddit.com', 'github': 'github.com',
        'stackoverflow': 'stackoverflow.com',
    }
    
    text_l = text.lower().strip()
    
    # Extract website
    website = None
    for key, value in websites.items():
        if key in text_l:
            website = value
            break
    
    if not website:
        return None, None
    
    # Extract search query
    query = None
    
    # Step 1: Look for explicit "search <query>" pattern
    website_keywords = '|'.join(list(websites.keys()))
    search_pattern = rf'(?:search|query)\s+(?:for\s+)?(.+?)(?:\s+(?:on|at|in)\s+(?:{website_keywords}))?$'
    search_match = re.search(search_pattern, text_l)
    
    if search_match:
        query = search_match.group(1).strip()
        # Remove "on" / "at" / "in" if they appear at the end
        query = re.sub(r'\s+(?:on|at|in)$', '', query)
        
        # Remove website names from query
        website_list = list(websites.keys())
        query_words = [w for w in query.split() if w not in website_list]
        query = ' '.join(query_words) if query_words else None
    
    # Step 2: If no search keyword found, try to extract remaining words
    if not query:
        query_text = text_l
        
        # Remove profile patterns
        profile_patterns = [
            r'with\s+chrome\s+profile\s+\w+',
            r'chrome\s+profile\s+\w+',
            r'with\s+profile\s+\w+',
            r'use\s+profile\s+\w+',
            r'profile\s+\w+',
        ]
        
        for pattern in profile_patterns:
            query_text = re.sub(pattern, '', query_text)
        
        # Remove structural and website keywords
        skip_words = {'with', 'chrome', 'for', 'open', 'go', 'to', 'on', 'in', 'and', 'profile', 'use', 'the', 'a', 'an', 'at', 'search', 'query'}
        skip_words.update(list(websites.keys()))
        
        query_words = [w for w in query_text.split() if w not in skip_words and w.strip()]
        query = ' '.join(query_words) if query_words else None
    
    return website, query

# Test cases
test_commands = [
    ("open youtube", "Should NOT have search query"),
    ("open youtube search python", "Should have 'python' query"),
    ("search python on google", "Should have 'python' query"),
    ("go to facebook", "Should NOT have search query"),
    ("youtube", "Should NOT have search query"),
    ("open youtube and search cat videos", "Should have 'cat videos' query"),
    ("with chrome profile work open youtube", "Should NOT have search query"),
    ("with chrome profile work search machine learning on youtube", "Should have 'machine learning' query"),
]

print("Testing search query extraction:\n")
for cmd, description in test_commands:
    website, query = _extract_website_and_action(cmd)
    status = "✅" if query else "❌"
    print(f"{status} '{cmd}'")
    print(f"   Description: {description}")
    print(f"   Website: {website}, Query: {query}\n")
