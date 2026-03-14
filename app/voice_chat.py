import os
import time
import wave
from typing import Optional

import numpy as np
import sounddevice as sd
import streamlit as st
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from TTS.api import TTS

from scripts.streaming import stream_graph_updates

load_dotenv()

# Keep a reliable default for local machines.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC")

wav_files = os.path.abspath(os.path.join(os.path.dirname(__file__), "wav_files"))
os.makedirs(wav_files, exist_ok=True)

LANGUAGE_HINTS = {
    "Auto Detect": None,
    "Hindi (hi)": "hi",
    "Telugu (te)": "te",
    "Tamil (ta)": "ta",
    "Kannada (kn)": "kn",
    "Malayalam (ml)": "ml",
    "Bengali (bn)": "bn",
    "Marathi (mr)": "mr",
    "Gujarati (gu)": "gu",
    "Punjabi (pa)": "pa",
    "Urdu (ur)": "ur",
}


@st.cache_resource(show_spinner=False)
def _load_stt_model(model_name: str):
    return whisper.load_model(model_name)


@st.cache_resource(show_spinner=False)
def _load_tts_model(model_name: str):
    return TTS(model_name, gpu=False)


def list_input_devices():
    """Return available input devices as (label, device_index)."""
    devices = sd.query_devices()
    options = [("System Default", None)]
    for idx, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            name = device.get("name", "Unknown input device")
            options.append((f"{idx}: {name}", idx))
    return options


def record_audio(
    filename=f"{wav_files}/user_order.wav",
    samplerate=16000,
    max_duration=18,
    base_speech_threshold=650,
    silence_ms_to_stop=1400,
    input_device=None,
):
    st.write("Speak now... recording stops after a short silence.")
    chunks = []
    speech_started = False
    silent_ms_after_speech = 0

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        callback=callback,
        dtype=np.int16,
        device=input_device,
    ):
        # Calibrate ambient noise for ~500ms so threshold adapts to room/mic noise.
        calibration_end = time.time() + 0.5
        noise_peaks = []
        speech_threshold = base_speech_threshold

        for _ in range(int(max_duration * 10)):
            sd.sleep(100)
            if not chunks:
                continue

            current_chunk = chunks[-1]
            peak = int(np.max(np.abs(current_chunk)))

            if time.time() < calibration_end:
                noise_peaks.append(peak)
                continue

            if noise_peaks:
                adaptive_floor = int(np.percentile(noise_peaks, 90))
                speech_threshold = max(base_speech_threshold, adaptive_floor + 220)
                noise_peaks = []

            if peak >= speech_threshold:
                speech_started = True
                silent_ms_after_speech = 0
            elif speech_started:
                silent_ms_after_speech += 100
                if silent_ms_after_speech >= silence_ms_to_stop:
                    break

    if not chunks:
        raise RuntimeError("No microphone audio captured.")

    audio_data = np.concatenate(chunks, axis=0)
    with wave.open(filename, "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(samplerate)
        wave_file.writeframes(audio_data.tobytes())

    return filename


def transcribe_audio(
    filename=f"{wav_files}/user_order.wav",
    selected_language="Auto Detect",
    translate_to_english=False,
):
    """Transcribe speech with fallback passes for stronger multilingual reliability."""
    if filename is None or not os.path.exists(filename):
        st.error("No valid audio file found. Please record again.")
        return ""

    try:
        stt_model = _load_stt_model(WHISPER_MODEL)
        language_hint = LANGUAGE_HINTS.get(selected_language, None)
        primary_task = "translate" if translate_to_english else "transcribe"

        def mean_logprob(resp: dict) -> float:
            segments = resp.get("segments") or []
            vals = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
            return float(np.mean(vals)) if vals else -999.0

        # Automatic language proposal from Whisper language-ID head.
        auto_candidates = []
        if language_hint is None:
            audio = whisper.load_audio(filename)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(stt_model.device)
            _, lang_probs = stt_model.detect_language(mel)
            auto_candidates = [code for code, _ in sorted(lang_probs.items(), key=lambda x: x[1], reverse=True)[:3]]

        lang_candidates = [language_hint] if language_hint else auto_candidates + [None]
        task_candidates = [primary_task] if translate_to_english else ["transcribe", "translate"]

        best_result: Optional[dict] = None
        best_score = -999.0

        for lang in lang_candidates:
            for task in task_candidates:
                result = stt_model.transcribe(
                    filename,
                    task=task,
                    language=lang,
                    temperature=0,
                    fp16=False,
                    condition_on_previous_text=False,
                    beam_size=6,
                    best_of=6,
                    no_speech_threshold=0.35,
                    logprob_threshold=-1.2,
                    compression_ratio_threshold=2.4,
                )
                text = (result.get("text") or "").strip()
                if len(text) < 3:
                    continue

                score = mean_logprob(result)
                if score > best_score:
                    best_score = score
                    best_result = result

        if not best_result:
            return ""

        detected_lang = best_result.get("language")
        if detected_lang:
            st.caption(f"Detected speech language: {detected_lang}")

        return (best_result.get("text") or "").strip()
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""


def get_llm_response(user_text):
    """Always forward transcribed text; action restrictions are enforced in agent prompt/tooling."""
    return stream_graph_updates(user_text)


def text_to_speech(response_text, max_words=50):
    """Convert response text to speech and play in chunks."""
    if not response_text.strip():
        st.error("No response text provided for TTS.")
        return None

    words = response_text.split()
    text_chunks = [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    output_files = []
    try:
        tts_model = _load_tts_model(TTS_MODEL_NAME)
        for idx, chunk in enumerate(text_chunks):
            output_file = os.path.join(wav_files, f"order_response_{idx}.wav")
            tts_model.tts_to_file(text=chunk, file_path=output_file)
            output_files.append(output_file)

        for file in output_files:
            audio = AudioSegment.from_wav(file)
            play(audio)

        return output_files
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None


def ai_voice_assistance():
    return record_audio, transcribe_audio, get_llm_response, text_to_speech
