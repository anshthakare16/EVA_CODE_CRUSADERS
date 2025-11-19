"""Helper utility functions"""

import os
import json
import config

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def cleanup_temp_files():
    """Clean up temporary files"""
    if os.path.exists(config.SCREENSHOT_TEMP_DIR):
        for file in os.listdir(config.SCREENSHOT_TEMP_DIR):
            filepath = os.path.join(config.SCREENSHOT_TEMP_DIR, file)
            try:
                os.remove(filepath)
            except Exception:
                pass
