import sys
sys.path.insert(0, r'c:\Users\sabni\Desktop\EVA_CODE_CRUSADERS-main')

from main import CommandProcessor

processor = CommandProcessor()

# Test cases
test_commands = [
    "search for python on google",  # WITH search query
    "open youtube",                   # WITHOUT search query
    "with profile work search machine learning on youtube",  # WITH search query
    "chrome profile dev open gmail",  # WITHOUT search query
]

for cmd in test_commands:
    print(f"\n{'='*70}")
    print(f"Command: '{cmd}'")
    print('='*70)
    
    # Process command
    processor.process_command(cmd)
    
    # Get the extracted keywords
    extracted = processor.current_extracted_keywords
    print(f"Extracted Keywords:")
    print(f"  - profile_name: {extracted.get('profile_name')}")
    print(f"  - website: {extracted.get('website')}")
    print(f"  - search_query: {extracted.get('search_query')}")
    
    # Get the generated steps
    steps = processor.current_steps
    print(f"\nGenerated Steps ({len(steps)} total):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step['action_type']}: {step['description']}")
