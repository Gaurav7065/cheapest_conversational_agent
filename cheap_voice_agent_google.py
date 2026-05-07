
import asyncio
import io
import os
import wave
import threading
import queue
import numpy as np
import sounddevice as sd
import pyttsx3
import webrtcvad
import noisereduce as nr
import speech_recognition as sr
from openai import OpenAI
from collections import deque

# ==============================
# 🔧 CONFIG
# ==============================

GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable not set. "
        "Set it before running: setx GROQ_API_KEY 'your_key'"
    )

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# VAD settings
VAD_AGGRESSIVENESS = 2
SPEECH_PADDING_MS = 400
SPEECH_PADDING_FRAMES = int(SPEECH_PADDING_MS / FRAME_DURATION_MS)
MIN_SPEECH_FRAMES = 8

# Noise reduction
NOISE_PROP_DECREASE = 1.0

# Mic cooldown after AI finishes speaking (prevents echo)
MIC_COOLDOWN_AFTER_SPEECH = 1  # seconds

# Interrupt detection
INTERRUPT_VOLUME_THRESHOLD = 500

# ==============================
# 🗣️ SYSTEM PROMPT
# ==============================

SYSTEM_PROMPT = """You are a helpful conversational AI assistant.

Your responses should be:
- Short and concise (1-2 sentences max)
- Natural and friendly
- Helpful and informative

Keep conversations engaging and ask follow-up questions when appropriate."""

# ==============================
# 🔧 GLOBALS
# ==============================

groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

raw_audio_queue = queue.Queue()
speech_queue = asyncio.Queue()
text_queue = asyncio.Queue()
response_queue = asyncio.Queue()

interrupt_event = asyncio.Event()
stop_event = threading.Event()

is_speaking = False
conversation_history = []

# ==============================
# 🔇 NOISE REDUCTION
# ==============================

def reduce_noise(audio_data: np.ndarray) -> np.ndarray:
    """Apply noise reduction to raw mic audio."""
    try:
        float_audio = audio_data.astype(np.float32) / 32768.0
        reduced = nr.reduce_noise(
            y=float_audio,
            sr=SAMPLE_RATE,
            stationary=False,
            prop_decrease=NOISE_PROP_DECREASE
        )
        return (reduced * 32768.0).astype(np.int16)
    except Exception:
        return audio_data.astype(np.int16)

# ==============================
# 🎙️ VAD MIC CAPTURE (thread)
# ==============================

class VADMicCapture(threading.Thread):
    def __init__(self, loop):
        super().__init__(daemon=True)
        self.loop = loop
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.triggered = False
        self.voiced_frames = []
        self.ring_buffer = deque(maxlen=SPEECH_PADDING_FRAMES)
        self.frame_buffer = np.array([], dtype=np.int16)

    def process_frame(self, frame_bytes: bytes) -> bool:
        try:
            return self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception:
            return False

    def run(self):
        print("🎙️  Mic capture thread started")

        def callback(indata, frames, time_info, status):
            if not is_speaking:
                cleaned = reduce_noise(indata.flatten())
                raw_audio_queue.put(cleaned.reshape(-1, 1))

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=FRAME_SIZE,
            callback=callback
        ):
            while not stop_event.is_set():
                try:
                    chunk = raw_audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self.frame_buffer = np.append(
                    self.frame_buffer,
                    chunk.flatten()
                )

                while len(self.frame_buffer) >= FRAME_SIZE:
                    frame = self.frame_buffer[:FRAME_SIZE]
                    self.frame_buffer = self.frame_buffer[FRAME_SIZE:]
                    frame_bytes = frame.tobytes()
                    is_speech = self.process_frame(frame_bytes)

                    if not self.triggered:
                        self.ring_buffer.append((frame, is_speech))
                        num_voiced = sum(1 for _, s in self.ring_buffer if s)

                        if num_voiced > 0.6 * self.ring_buffer.maxlen:
                            self.triggered = True
                            self.voiced_frames = [f for f, _ in self.ring_buffer]
                            self.ring_buffer.clear()
                    else:
                        self.voiced_frames.append(frame)
                        self.ring_buffer.append((frame, is_speech))
                        num_unvoiced = sum(1 for _, s in self.ring_buffer if not s)

                        if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                            self.triggered = False

                            if len(self.voiced_frames) >= MIN_SPEECH_FRAMES:
                                audio_data = np.concatenate(self.voiced_frames)
                                asyncio.run_coroutine_threadsafe(
                                    speech_queue.put(audio_data),
                                    self.loop
                                )
                                print(f"🎤 Speech captured ({len(self.voiced_frames) * FRAME_DURATION_MS}ms)")

                            self.voiced_frames = []
                            self.ring_buffer.clear()

# ==============================
# 🛑 INTERRUPT DETECTOR (thread)
# ==============================

class InterruptDetector(threading.Thread):
    def __init__(self, loop):
        super().__init__(daemon=True)
        self.loop = loop

    def run(self):
        def callback(indata, frames, time_info, status):
            if is_speaking:
                volume = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
                if volume > INTERRUPT_VOLUME_THRESHOLD:
                    print(f"\n🛑 Interrupt! Volume={volume:.0f}")
                    asyncio.run_coroutine_threadsafe(
                        self._set_interrupt(),
                        self.loop
                    )

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=FRAME_SIZE,
            callback=callback
        ):
            stop_event.wait()

    async def _set_interrupt(self):
        interrupt_event.set()

# ==============================
# 🔊 TTS ENGINE (pyttsx3)
# ==============================

class TTSEngine:
    def __init__(self):
        self.lock = threading.Lock()

    def speak(self, text: str):
        with self.lock:
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')

                voice_id = voices[0].id
                for v in voices:
                    if 'zira' in v.name.lower():
                        voice_id = v.id
                        break

                engine.setProperty('voice', voice_id)
                engine.setProperty('rate', 145)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"❌ TTS error: {e}")

tts_engine = TTSEngine()

# ==============================
# 🗣️ ASR — Google Speech Recognition
# ==============================

async def transcribe_audio(audio_data: np.ndarray) -> str:
    loop = asyncio.get_running_loop()
    audio_data = await loop.run_in_executor(None, reduce_noise, audio_data)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.astype(np.int16).tobytes())

    wav_buffer.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buffer) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text.strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"❌ Google Speech Recognition error: {e}")
            return ""

async def asr_worker():
    print("🗣️  ASR worker started")

    while True:
        audio_data = await speech_queue.get()

        if is_speaking:
            print("🔇 Discarding echo...")
            continue

        print("📤 Transcribing...")
        try:
            text = await transcribe_audio(audio_data)
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            continue

        if not text:
            print("⚠️  Empty transcription, skipping")
            continue

        print(f"\n🧑 You: {text}")

        if is_speaking:
            interrupt_event.set()
            await asyncio.sleep(0.3)

            while not response_queue.empty():
                try:
                    response_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        await text_queue.put(text)

# ==============================
# 🤖 LLM — Groq
# ==============================

def _split_into_sentences(text: str) -> list[str]:
    sentences = []
    buffer = text.strip()
    while buffer:
        for punct in [".", "?", "!", "\n"]:
            if punct in buffer:
                index = buffer.find(punct) + 1
                sentence = buffer[:index].strip()
                if sentence:
                    sentences.append(sentence)
                buffer = buffer[index:].strip()
                break
        else:
            sentences.append(buffer.strip())
            break
    return sentences

def _create_completion_sync(messages):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=300,
        temperature=0.85,
    )
    return response.choices[0].message.content.strip()

async def stream_sentences(user_text: str):
    conversation_history.append({
        "role": "user",
        "content": user_text
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history[-20:]
    ]

    loop = asyncio.get_running_loop()
    full_response = await loop.run_in_executor(
        None,
        _create_completion_sync,
        messages
    )

    if full_response:
        for sentence in _split_into_sentences(full_response):
            yield sentence

        conversation_history.append({
            "role": "assistant",
            "content": full_response.strip()
        })

async def llm_worker():
    print("🤖 LLM worker started")

    while True:
        user_text = await text_queue.get()
        interrupt_event.clear()
        print(f"💭 Thinking...")

        try:
            async for sentence in stream_sentences(user_text):
                if interrupt_event.is_set():
                    break
                print(f"📝 Queued: {sentence}")
                await response_queue.put(sentence)
        except Exception as e:
            print(f"❌ LLM error: {e}")
            await response_queue.put("I'm sorry, something went wrong. Could you please repeat that?")

# ==============================
# 🔊 TTS WORKER
# ==============================

async def tts_worker():
    global is_speaking
    print("🔊 TTS worker started")

    loop = asyncio.get_running_loop()

    while True:
        try:
            text = await asyncio.wait_for(response_queue.get(), timeout=0.3)
        except asyncio.TimeoutError:
            continue

        # skip if interrupted
        if interrupt_event.is_set():
            while not response_queue.empty():
                try:
                    response_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            is_speaking = False
            continue

        print(f"🤖 AI: {text}")

        is_speaking = True
        try:
            await loop.run_in_executor(None, tts_engine.speak, text)
        except Exception as e:
            print(f"❌ TTS error: {e}")

        if response_queue.empty() and not interrupt_event.is_set():
            await asyncio.sleep(MIC_COOLDOWN_AFTER_SPEECH)
            is_speaking = False

            while not speech_queue.empty():
                try:
                    speech_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            print("🎙️  Listening...")
        elif interrupt_event.is_set():
            is_speaking = False

        if not interrupt_event.is_set():
            await asyncio.sleep(0.08)

# ==============================
# 🚀 MAIN
# ==============================

async def main():
    loop = asyncio.get_running_loop()

    print("=" * 50)
    print("🚀  World's Cheapest Voice Agent")
    print("=" * 50)
    print(f"🎙️   Input : {sd.query_devices(kind='input')['name']}")
    print(f"🔊  Output: {sd.query_devices(kind='output')['name']}")
    print("=" * 50)
    print("💬  Speak naturally. Interrupt anytime.")
    print("⌨️   Press Ctrl+C to quit.")
    print("=" * 50 + "\n")

    vad_thread = VADMicCapture(loop)
    vad_thread.start()

    interrupt_thread = InterruptDetector(loop)
    interrupt_thread.start()

    greeting = "Hi there! How can I help you today?"
    print(f"🤖 AI: {greeting}")
    await loop.run_in_executor(None, tts_engine.speak, greeting)

    await asyncio.sleep(MIC_COOLDOWN_AFTER_SPEECH)
    is_speaking = False
    print("🎙️  Listening...")

    await asyncio.gather(
        asr_worker(),
        llm_worker(),
        tts_worker(),
    )

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        stop_event.set()