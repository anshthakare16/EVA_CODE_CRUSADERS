"""
System Executor - Execute system-level commands
Handles volume, brightness, WiFi, Bluetooth, Flight Mode, etc.
"""

import logging
import subprocess
import time
import os

logger = logging.getLogger("SystemExecutor")

class SystemExecutor:
    """Execute system commands directly via Windows API"""
    
    # Quick Settings Panel Options (Windows + A)
    QUICK_SETTINGS_OPTIONS = {
        'wifi': 0,              # No arrow presses needed
        'bluetooth': 1,         # 1 left arrow
        'flight_mode': 2,       # 2 left arrows
        'airplane_mode': 2,     # Same as flight mode
        'energy_saver': 3,      # 3 left arrows
        'night_light': 4,       # 4 left arrows
        'project': 5,           # 5 left arrows
        'accessibility': 6,     # 6 left arrows
        'live_captions': 7,     # 7 left arrows
        'mobile_hotspot': 8,    # 8 left arrows
        'nearby_sharing': 9,    # 9 left arrows
        'cast': 10              # 10 left arrows
    }
    
    def __init__(self, executor_bridge):
        """Initialize system executor"""
        self.executor = executor_bridge
        logger.info("✓ System executor initialized")

    def set_volume(self, level):
        """Set system volume (0-100) using keyboard - SIMPLE & WORKS!"""
        try:
            logger.info(f"🔊 Setting volume to {level}%")
            level = max(0, min(100, level))
    
            try:
                import time
                from pynput.keyboard import Controller, Key
        
                keyboard = Controller()
        
                # First, press volume down 50 times to go to minimum
                logger.info("Resetting to min volume...")
                for _ in range(50):
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    time.sleep(0.01)
        
                time.sleep(0.3)
        
                # Now press volume up to reach target level
                # Windows has ~50 volume steps, so we calculate presses needed
                target_presses = int((level / 100) * 50)
                logger.info(f"Setting to {level}% ({target_presses} presses)...")
        
                for _ in range(target_presses):
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    time.sleep(0.01)
        
                logger.info(f"✓ Volume set to {level}%")
                return {"success": True, "message": f"Volume set to {level}%"}
        
            except Exception as e:
                logger.error(f"Keyboard volume failed: {e}")
                return {"success": False, "error": str(e)}

        except Exception as e:
            logger.error(f"Failed to set volume: {e}")
            return {"success": False, "error": str(e)}




    def set_brightness(self, level):
        """Set screen brightness (0-100) via WMI"""
        try:
            logger.info(f"💡 Setting brightness to {level}%")
            level = max(0, min(100, level))
            
            try:
                import wmi
                c = wmi.WMI(namespace='wmi')
                methods = c.WmiMonitorBrightnessMethods()[0]
                methods.WmiSetBrightness(level, 0)
                logger.info(f"✓ Brightness set to {level}%")
                return {"success": True, "message": f"Brightness set to {level}%"}
            except Exception as wmi_err:
                logger.warning(f"WMI failed: {wmi_err}, trying PowerShell...")
                
                ps_command = f'(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})'
                subprocess.run(['powershell', '-Command', ps_command], 
                             check=True, capture_output=True, timeout=5)
                logger.info(f"✓ Brightness set to {level}% via PowerShell")
                return {"success": True, "message": f"Brightness set to {level}%"}
                
        except Exception as e:
            logger.error(f"❌ Brightness control failed: {e}")
            return {"success": False, "error": str(e)}

    def toggle_quick_setting(self, setting_name):
        """Toggle a Quick Settings option (WiFi, Bluetooth, etc.) via Windows + A"""
        try:
            setting_lower = setting_name.lower().replace(' ', '_')
            
            if setting_lower not in self.QUICK_SETTINGS_OPTIONS:
                logger.error(f"Unknown setting: {setting_name}")
                return {"success": False, "error": f"Unknown setting: {setting_name}"}
            
            arrow_count = self.QUICK_SETTINGS_OPTIONS[setting_lower]
            logger.info(f"🎛️ Toggling {setting_name} (arrows: {arrow_count})")
            
            # Open Quick Settings (Windows + A)
            self.executor.press_key_combination(['win', 'a'])
            time.sleep(0.8)  # Wait for panel to open
            
            # Navigate with left arrows
            for _ in range(arrow_count):
                self.executor.press_key('left')
                time.sleep(0.1)
            
            # Press Enter to toggle
            self.executor.press_key('enter')
            time.sleep(0.3)
            
            # Close panel (Escape)
            self.executor.press_key('escape')
            
            logger.info(f"✓ Toggled {setting_name}")
            return {"success": True, "message": f"{setting_name.title()} toggled"}
            
        except Exception as e:
            logger.error(f"❌ Failed to toggle {setting_name}: {e}")
            return {"success": False, "error": str(e)}

    def execute_system_command(self, action):
        """Execute various system commands"""
        try:
            cmd = action.lower()
            logger.info(f"⚙️ Executing system command: {cmd}")
            
            # Volume commands
            if 'volume' in cmd:
                if 'mute' in cmd or 'unmute' in cmd:
                    self.executor.press_key('volume_mute')
                    return {"success": True, "message": "Volume muted/unmuted"}
                elif 'up' in cmd or 'increase' in cmd:
                    for _ in range(5):
                        self.executor.press_key('volume_up')
                        time.sleep(0.02)
                    return {"success": True, "message": "Volume increased"}
                elif 'down' in cmd or 'decrease' in cmd:
                    for _ in range(5):
                        self.executor.press_key('volume_down')
                        time.sleep(0.02)
                    return {"success": True, "message": "Volume decreased"}
                else:
                    # Extract number
                    import re
                    match = re.search(r'\d+', cmd)
                    if match:
                        level = int(match.group())
                        return self.set_volume(level)
            
            # Brightness commands
            elif 'brightness' in cmd:
                import re
                match = re.search(r'\d+', cmd)
                if match:
                    level = int(match.group())
                    return self.set_brightness(level)
            
            # WiFi
            elif 'wifi' in cmd or 'wi-fi' in cmd or 'wi fi' in cmd:
                return self.toggle_quick_setting('wifi')
            
            # Bluetooth
            elif 'bluetooth' in cmd:
                return self.toggle_quick_setting('bluetooth')
            
            # Flight Mode / Airplane Mode
            elif 'flight' in cmd or 'airplane' in cmd:
                return self.toggle_quick_setting('flight_mode')
            
            # Night Light
            elif 'night' in cmd and 'light' in cmd:
                return self.toggle_quick_setting('night_light')
            
            # Energy Saver / Battery Saver
            elif 'energy' in cmd or 'battery' in cmd:
                return self.toggle_quick_setting('energy_saver')
            
            # Mobile Hotspot
            elif 'hotspot' in cmd:
                return self.toggle_quick_setting('mobile_hotspot')
            
            # Power commands
            elif 'shutdown' in cmd or 'shut down' in cmd:
                logger.warning("🔴 SHUTDOWN command received")
                subprocess.run(['shutdown', '/s', '/t', '0'])
                return {"success": True, "message": "Shutting down..."}
            
            elif 'restart' in cmd or 'reboot' in cmd:
                logger.warning("🔄 RESTART command received")
                subprocess.run(['shutdown', '/r', '/t', '0'])
                return {"success": True, "message": "Restarting..."}
            
            elif 'sleep' in cmd:
                logger.info("😴 SLEEP command received")
                subprocess.run(['rundll32.exe', 'powrprof.dll,SetSuspendState', '0,1,0'])
                return {"success": True, "message": "Entering sleep mode..."}
            
            elif 'lock' in cmd:
                logger.info("🔒 LOCK command received")
                subprocess.run(['rundll32.exe', 'user32.dll,LockWorkStation'])
                return {"success": True, "message": "Locking workstation..."}
            
            else:
                logger.warning(f"Unknown system command: {cmd}")
                return {"success": False, "error": f"Unknown system command: {cmd}"}
                
        except Exception as e:
            logger.error(f"❌ System command failed: {e}")
            return {"success": False, "error": str(e)}
