"""
Session Manager - Manages active sessions with timeout logic
Methodology: 10-second timeout from LAST ACTIVITY, NOT session start
"""
import time
import logging

logger = logging.getLogger("SessionManager")

class SessionManager:
    """Manages EVA session lifecycle and timeout"""
    
    def __init__(self, timeout_seconds=10):
        """Initialize session manager"""
        self.timeout_seconds = timeout_seconds
        self.active = False
        self.last_activity_time = None
        self.commands_history = []
        
        logger.info(f"✓ Session manager initialized (timeout: {timeout_seconds}s)")
    
    def start_session(self):
        """Start a new session"""
        self.active = True
        self.last_activity_time = time.time()  # ✅ Reset on START
        self.commands_history = []
        logger.info("📍 Session started")
    
    def is_active(self):
        """Check if session is currently active"""
        return self.active
    
    def check_timeout(self):
        """Check if session has timed out (10 seconds since LAST activity)"""
        if not self.active:
            return False
        
        # ✅ Check time since LAST ACTIVITY, not session start
        time_since_activity = time.time() - self.last_activity_time
        
        if time_since_activity > self.timeout_seconds:
            logger.info(f"⏱️ Timeout: {time_since_activity:.1f}s > {self.timeout_seconds}s")
            return True
        
        return False
    
    def update_activity(self):
        """Update last activity time (called when command received)"""
        self.last_activity_time = time.time()
        logger.info(f"✓ Activity updated (reset timeout)")
    
    def add_command(self, command_text, result):
        """Add command to history"""
        self.commands_history.append({
            'command': command_text,
            'result': result,
            'timestamp': time.time()
        })
        self.update_activity()  # ✅ Reset timeout on new command
    
    def end_session(self):
        """End current session"""
        self.active = False
        logger.info("📴 Session ended")
    
    def should_end_session(self, command_text):
        """Check if command is goodbye phrase"""
        return "goodbye" in command_text.lower() and "jarvis" in command_text.lower()
