"""
Wake Word Detection using faster_whisper
"""

import logging
import pyaudio
import numpy as np
import time
from faster_whisper import WhisperModel

logger = logging.getLogger("WakeWordDetector")

class WakeWordDetector:
    """faster_whisper-based wake word detector"""

    def __init__(self, wake_word="jarvis", model_size="tiny.en"):
        """Initialize faster_whisper"""
        self.wake_word = wake_word.lower()
        self.model_size = model_size
        self.is_running = False

        try:
            logger.info(f"Initializing faster_whisper with model: '{self.model_size}'")
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info(f"✓ faster_whisper initialized (model: '{self.model_size}')")

            self.pa = pyaudio.PyAudio()
            self.audio_stream = None
            self.frames = []

        except Exception as e:
            logger.critical(f"❌ faster_whisper initialization FAILED: {e}")
            raise RuntimeError(f"faster_whisper MUST work. Error: {e}")

    def start(self):
        """Start faster_whisper listening"""
        try:
            logger.info("Opening audio stream...")
            self.audio_stream = self.pa.open(
                rate=16000,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=1024
            )
            self.is_running = True
            logger.info("🎤 faster_whisper listening for wake word...")
            logger.info(f"✓ Audio stream started (say '{self.wake_word.upper()}' clearly)")

        except Exception as e:
            logger.critical(f"❌ Failed to start audio: {e}")
            raise RuntimeError(f"Audio stream failed: {e}")

    def listen(self):
        """Listen for wake word using faster_whisper"""
        if not self.audio_stream:
            raise RuntimeError("Audio stream not started")

        try:
            data = self.audio_stream.read(1024, exception_on_overflow=False)
            self.frames.append(np.frombuffer(data, dtype=np.int16))

            # Process audio in chunks
            if len(self.frames) >= 50:  # Process every ~5 seconds of audio
                audio_data = np.concatenate(self.frames)
                self.frames = []

                # Normalize audio
                audio_data = audio_data.astype(np.float32) / 32768.0

                # Transcribe
                segments, _ = self.model.transcribe(audio_data, beam_size=5)
                for segment in segments:
                    if self.wake_word in segment.text.lower():
                        logger.info(f"🎯 WAKE WORD '{self.wake_word.upper()}' DETECTED!")
                        return True
            return False

        except Exception as e:
            logger.error(f"Listen error: {e}")
            return False

    def detect(self, timeout=None):
        """
        Continuously listens until wake word is detected or timeout occurs
        """
        logger.info(f"🎤 Starting wake word detection (timeout: {timeout}s)")
        self.start()
        start_time = time.time()

        try:
            while self.is_running:
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"⏱️ Wake word detection timeout ({timeout}s)")
                    self.stop()
                    return False

                if self.listen():
                    logger.info("✓ Wake word detected!")
                    self.stop()
                    return True

                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Wake word detection interrupted by user")
            self.stop()
            return False

        except Exception as e:
            logger.error(f"Detection error: {e}")
            self.stop()
            return False
        
        return False

    def stop(self):
        """Stop faster_whisper"""
        try:
            self.is_running = False
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None

            if self.pa:
                self.pa.terminate()

            logger.info("✓ faster_whisper stopped")

        except Exception as e:
            logger.error(f"Error stopping faster_whisper: {e}")