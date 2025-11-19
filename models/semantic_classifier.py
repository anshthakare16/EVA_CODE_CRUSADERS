"""
Semantic Classifier - DISABLED (use CommandProcessor instead)
"""

import logging

logger = logging.getLogger("SemanticClassifier")

class SemanticClassifier:
    """Deprecated - Use CommandProcessor instead"""
    
    def __init__(self, *args, **kwargs):
        logger.warning("⚠️ SemanticClassifier deprecated - using CommandProcessor (Gemini)")
