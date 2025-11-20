import re

def _extract_profile_name(text):
    patterns = [
        r'with chrome profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'chrome profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'with profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'use profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
        r'profile ([\w\s]+?)(?:\s+(?:search|open|go|and))',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Default"

# Test cases
test_cases = [
    "with chrome profile work search python on google",
    "chrome profile personal search machine learning youtube",
    "with profile dev search coding tutorials on youtube",
    "profile gaming search for laptop on amazon",
]

for test in test_cases:
    profile = _extract_profile_name(test)
    print(f"Input: '{test}'")
    print(f"Extracted Profile: '{profile}'")
    print()
