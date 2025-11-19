"""
Speech-to-text using Faster Whisper
Simple, robust recording without over-engineering
"""

from faster_whisper import WhisperModel
import pyaudio
import wave
import tempfile
import os
import logging
import config

logger = logging.getLogger("SpeechToText")

class SpeechToText:
    """Convert speech to text using Faster Whisper"""
    
    def __init__(self):
        self.logger = logging.getLogger("SpeechToText")
        
        # Initialize Faster Whisper model
        try:
            self.model = WhisperModel(
                "large",
                device="cpu",
                compute_type="int8"
            )
            self.logger.info(f"Faster Whisper model loaded: large")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper: {e}")
            raise
    
    def record_audio(self, duration=10):
        """
        ✅ SIMPLE RECORDING: Just record for fixed duration
        Whisper will handle silence filtering internally
        
        Args:
            duration: Recording duration in seconds (10 = good balance)
        
        Returns:
            str: Path to temporary WAV file
        """
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        self.logger.info(f"🎤 Recording (speak clearly, you have {duration} seconds)...")
        
        frames = []
        chunks = int(RATE / CHUNK * duration)
        
        for _ in range(chunks):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                self.logger.error(f"Recording error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        self.logger.info(f"✓ Recording saved ({len(frames) * CHUNK / RATE:.1f} seconds)")
        return temp_file.name
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file to text"""
        try:
            # ✅ Simple transcription with quality settings
            segments, info = self.model.transcribe(
                audio_path,
                language="en",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False
            )
            
            # Collect all segments
            text = " ".join([segment.text for segment in segments]).strip()
            
            # ✅ If empty or too short, return empty (not hallucination)
            if not text or len(text) < 2:
                self.logger.info("📴 No speech detected")
                return ""
            
            self.logger.info(f"Transcription: {text}")
            return text
        
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""
        
        finally:
            # Clean up temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    # ✅ COMPATIBILITY METHODS FOR main.py
    
    def record(self, duration=10):
        """
        ✅ NEW: Compatibility wrapper for record_audio()
        
        Args:
            duration: Recording duration in seconds
        
        Returns:
            str: Path to temporary WAV file
        """
        return self.record_audio(duration)
    
    def transcribe(self, audio_path):
        """
        ✅ NEW: Compatibility wrapper for transcribe_audio()
        
        Args:
            audio_path: Path to WAV file
        
        Returns:
            str: Transcribed text
        """
        return self.transcribe_audio(audio_path)
    
    def listen(self):
        """
        ✅ NEW: Combined record + transcribe (convenience)
        
        Returns:
            str: Transcribed text from 10-second recording
        """
        audio_path = self.record_audio(duration=10)
        return self.transcribe_audio(audio_path)
    
    def speak(self, text):
        """Placeholder for TTS (handled by TextToSpeech class)"""
        pass
