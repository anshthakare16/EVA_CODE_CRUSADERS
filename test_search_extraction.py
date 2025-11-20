"""
Test script to verify search query extraction with custom profiles
"""

import re

def _extract_website_and_action(text):
    websites = {
        'youtube': 'youtube.com', 'google': 'google.com', 'gmail': 'mail.google.com', 'facebook': 'facebook.com',
        'twitter': 'twitter.com', 'instagram': 'instagram.com', 'linkedin': 'linkedin.com', 'github': 'github.com',
        'reddit': 'reddit.com', 'amazon': 'amazon.com', 'netflix': 'netflix.com', 'spotify': 'open.spotify.com',
    }
    text_l = text.lower()
    website = None
    for keyword, url in websites.items():
        if keyword in text_l:
            website = url
            break
    
    # ✅ FIXED: More intelligent search query extraction
    query = None
    
    # Step 1: Look for explicit "search <query>" pattern
    # Match: "search <query>" or "search for <query>" (stop at website keywords)
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
    "with chrome profile work search python on google",
    "chrome profile personal search machine learning youtube",
    "profile dev search coding tutorials on youtube",
    "search python on google",  # no profile
    "open youtube search for cat videos",  # no profile
    "with profile work go to amazon search for laptop",
]

print("=" * 80)
print("SEARCH QUERY EXTRACTION TEST")
print("=" * 80)

for cmd in test_commands:
    website, query = _extract_website_and_action(cmd)
    print(f"\n📝 Command: {cmd}")
    print(f"   🌐 Website: {website}")
    print(f"   🔍 Search Query: {query if query else '(None)'}")

print("\n" + "=" * 80)
