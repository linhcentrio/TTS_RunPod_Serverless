#!/usr/bin/env python3
"""
RunPod Serverless Handler cho TTS Services - Non-GPU
- Hỗ trợ: OpenAI TTS, EdgeTTS, Gemini TTS
- Hỗ trợ đầy đủ tham số như các CLI gốc (OpenAI/Gemini) và tính năng list với EdgeTTS
- Lưu file lên MinIO cố định và trả presigned/public URL
"""

import runpod
import os
import tempfile
import uuid
import time
import sys
import gc
import json
import traceback
import asyncio
from pathlib import Path
import requests
from urllib.parse import quote
import logging
from datetime import datetime, timedelta

from minio import Minio
from minio.error import S3Error

# Mô-đun wrapper theo dự án (OpenAI/Gemini)
from openai_tts import OpenAITTSCLI  # CLI gốc OpenAI hỗ trợ streaming with_streaming_response [14]
from gemini_tts import GeminiTTSCLI   # CLI gốc Gemini hỗ trợ single/multi voices/styles [3]

# Import EdgeTTS an toàn để tránh shadowing gói 'edge-tts' (khuyến nghị đổi tên file local thành ms_edge_tts.py)
try:
    from ms_edge_tts import EdgeTTS, quick_tts, VOICES as EDGE_VOICES  # Ưu tiên module đã đổi tên [1]
except Exception:
    import edge_tts as edge_api  # Fallback sang gói pip chính thức (có Communicate) [4][11]
    EDGE_VOICES = {}  # Không có bảng rút gọn khi fallback [4][11]

    class EdgeTTS:
        def __init__(self, voice: str = "vi-VN-HoaiMyNeural"):
            self.voice = voice
        async def text_to_speech(self, text: str, output_file: str | None = None) -> bytes:
            comm = edge_api.Communicate(text, self.voice)
            audio = b""
            async for chunk in comm.stream():
                if chunk["type"] == "audio" and chunk.get("data"):
                    audio += chunk["data"]
            if not audio:
                raise RuntimeError("EdgeTTS produced empty audio bytes")
            if output_file:
                with open(output_file, "wb") as f:
                    f.write(audio)
            return audio

    async def quick_tts(text: str, voice: str = "vi-VN-HoaiMyNeural", output_file: str | None = None) -> bytes:
        tts = EdgeTTS(voice)
        return await tts.text_to_speech(text, output_file)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tts_handler")

# MinIO cố định (có thể override qua ENV)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "media.aiclip.ai")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "VtZ6MUPfyTOH3qSiohA2")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "video")  # mặc định "video" để đồng nhất log trước đó
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Presets instructions OpenAI (copy từ CLI gốc)
OPENAI_INSTRUCTION_PRESETS = {
    "default": "Speak naturally and clearly",
    "cheerful": "Speak in a cheerful and positive tone",
    "professional": "Speak in a professional and authoritative manner",
    "calm": "Speak in a calm and soothing voice",
    "excited": "Speak with enthusiasm and energy",
    "storytelling": "Narrate like telling an engaging story",
    "news": "Read like a news anchor with clarity and authority",
    "podcast": "Speak like a friendly podcast host",
    "meditation": "Speak in a gentle, meditative voice",
    "whisper": "Speak in a soft whisper",
    "dramatic": "Speak dramatically with varying emotions",
    "fast": "Speak at a faster pace with excitement",
    "slow": "Speak slowly and deliberately",
    "educational": "Speak like a teacher explaining concepts clearly",
}  # [2]

# Gemini voice/style/model danh sách như CLI gốc (không cần API key để trả list)
GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
]  # [3]
GEMINI_STYLES = {
    "default": "",
    "tv-host": "Speak like a professional TV host with energy and clarity",
    "storyteller": "Narrate like a captivating storyteller with dramatic expression",
    "news": "Read like a news anchor with authority and clarity",
    "podcast": "Speak like an engaging podcast host with warmth",
    "audiobook": "Narrate like a professional audiobook reader with steady pace",
    "cheerful": "Say cheerfully with positive energy",
    "bedtime": "Read like a gentle bedtime story with soft voice",
    "dramatic": "Express dramatically with varying emotions and pauses",
    "teacher": "Speak like a wise teacher with clear explanation",
    "poetry": "Read like poetry recital with artistic expression",
    "spooky": "Speak in a spooky whisper with mysterious tone",
    "fast": "Speak at a fast pace with excitement",
    "slow": "Speak slowly and clearly with emphasis",
    "motivational": "Speak like a motivational speaker with inspiration",
    "meditation": "Speak as a meditation guide with calm peaceful voice",
}  # [3]
GEMINI_MODELS = [
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash-preview-tts",
]  # [3]

# OpenAI options (không cần API key để trả list)
OPENAI_MODELS = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]  # [2]
OPENAI_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]  # [2]
OPENAI_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]  # [2]

# TTS metadata
TTS_SERVICES = {
    "openai": {"name": "OpenAI TTS", "requires_api_key": True},
    "edge": {"name": "Microsoft Edge TTS", "requires_api_key": False},
    "gemini": {"name": "Google Gemini TTS", "requires_api_key": True},
}  # [2][1][3]

# MinIO init
minio_client = None
try:
    minio_client = Minio(
        MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE
    )
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        logger.info(f"Created MinIO bucket: {MINIO_BUCKET}")
    else:
        logger.info(f"MinIO bucket exists: {MINIO_BUCKET}")
    logger.info("MinIO client initialized")
except Exception as e:
    logger.error(f"MinIO init failed: {e}")

# Client cache
tts_clients = {}

def get_tts_client(service: str, api_key: str | None):
    key = f"{service}:{api_key[:8] if api_key else 'nokey'}"
    if key in tts_clients:
        return tts_clients[key]
    if service == "openai":
        if not api_key:
            raise ValueError("OpenAI API key required")
        tts_clients[key] = OpenAITTSCLI(api_key=api_key)
    elif service == "edge":
        tts_clients[key] = EdgeTTS()
    elif service == "gemini":
        if not api_key:
            raise ValueError("Gemini API key required")
        tts_clients[key] = GeminiTTSCLI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported service: {service}")
    return tts_clients[key]

def ensure_bucket_and_upload(local_path: str, object_name: str) -> tuple[str, str]:
    if not minio_client:
        raise RuntimeError("MinIO client not initialized")
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
    minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
    presigned = minio_client.presigned_get_object(MINIO_BUCKET, object_name, expires=timedelta(days=7))
    scheme = "https" if MINIO_SECURE else "http"
    public = f"{scheme}://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
    return presigned, public

def resolve_text_input(data: dict) -> tuple[str | None, str | None]:
    """
    Hỗ trợ tương đương tham số CLI:
    - 'text' (positional)
    - 'input' (đường dẫn file/URL, serverless hỗ trợ URL http/https; nếu là chuỗi thường thì dùng trực tiếp)
    """
    if "input" in data and data["input"]:
        src = data["input"]
        if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://")):
            try:
                resp = requests.get(src, timeout=30)
                resp.raise_for_status()
                txt = resp.text.strip()
                if not txt:
                    return None, "Input URL returned empty content"
                return txt, None
            except Exception as e:
                return None, f"Failed to fetch input URL: {e}"
        elif isinstance(src, str):
            txt = src.strip()
            if not txt:
                return None, "Input provided but empty"
            return txt, None
        else:
            return None, "Input must be a string or URL"
    if "text" in data and data["text"]:
        txt = str(data["text"]).strip()
        if not txt:
            return None, "Text is empty"
        return txt, None
    return None, "Missing 'text' or 'input'"

# -------- OpenAI paths (đủ tham số CLI gốc) --------

async def openai_generate_one(client: OpenAITTSCLI, params: dict, out_path: str) -> tuple[bool, str]:
    try:
        text, err = resolve_text_input(params)
        if err:
            return False, err
        # preset_instructions -> instructions nếu chưa set
        instructions = params.get("instructions")
        if not instructions and params.get("preset_instructions"):
            preset = params["preset_instructions"]
            if preset in OPENAI_INSTRUCTION_PRESETS:
                instructions = OPENAI_INSTRUCTION_PRESETS[preset]
        ok = await client.generate_speech(
            text=text,
            output_path=out_path,
            model=params.get("model", "tts-1"),
            voice=params.get("voice", "alloy"),
            speed=params.get("speed", 1.0),
            instructions=instructions,
            response_format=params.get("response_format", "mp3"),
        )
        if not ok:
            return False, "OpenAI generate_speech returned False"
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return False, "OpenAI file not created or empty"
        return True, ""
    except Exception as e:
        return False, f"OpenAI error: {e}"

async def openai_preview(client: OpenAITTSCLI, params: dict, out_path: str) -> tuple[bool, str]:
    try:
        text, err = resolve_text_input(params)
        if err:
            return False, err
        instructions = params.get("instructions")
        if not instructions and params.get("preset_instructions"):
            preset = params["preset_instructions"]
            if preset in OPENAI_INSTRUCTION_PRESETS:
                instructions = OPENAI_INSTRUCTION_PRESETS[preset]
        audio = await client.preview_speech(
            text=text,
            model=params.get("model", "tts-1"),
            voice=params.get("voice", "alloy"),
            instructions=instructions,
        )
        if not audio:
            return False, "Preview returned empty bytes"
        with open(out_path, "wb") as f:
            f.write(audio)
        return True, ""
    except Exception as e:
        return False, f"OpenAI preview error: {e}"

async def openai_batch(client: OpenAITTSCLI, params: dict, temp_dir: str, path_prefix: str) -> tuple[list[dict], list[dict]]:
    texts = []
    if isinstance(params.get("batch"), list):
        texts = params["batch"]
    elif isinstance(params.get("batch"), dict) and "texts" in params["batch"]:
        texts = params["batch"]["texts"]
    else:
        return [], [{"index": -1, "error": "batch must be list or {'texts': [...]}"}]
    results_ok, results_fail = [], []
    ofmt = params.get("response_format", "mp3")
    for i, t in enumerate(texts):
        one = params.copy()
        one["text"] = t
        fname = params.get("output") or f"openai_{i+1}_{uuid.uuid4().hex[:6]}.{ofmt}"
        local = os.path.join(temp_dir, fname)
        ok, err = await openai_generate_one(client, one, local)
        if not ok:
            results_fail.append({"index": i, "error": err})
            continue
        try:
            obj = f"{path_prefix.rstrip('/')}/{fname}"
            ps, pu = ensure_bucket_and_upload(local, obj)
            results_ok.append({
                "index": i,
                "presigned_url": ps,
                "public_url": pu,
                "object_name": obj,
                "size_bytes": os.path.getsize(local),
            })
        except Exception as e:
            results_fail.append({"index": i, "error": f"Upload failed: {e}"})
    return results_ok, results_fail

# -------- Edge/Gemini --------

async def generate_tts_edge(client: EdgeTTS, params: dict, out_path: str) -> tuple[bool, str]:
    try:
        text, err = resolve_text_input(params)
        if err:
            return False, err
        await quick_tts(text=text, voice=params.get("voice", "vi-VN-HoaiMyNeural"), output_file=out_path)
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return False, "EdgeTTS file not created or empty"
        return True, ""
    except Exception as e:
        return False, f"EdgeTTS error: {e}"

def generate_tts_gemini(client: GeminiTTSCLI, params: dict, out_path: str) -> tuple[bool, str]:
    try:
        text, err = resolve_text_input(params)
        if err:
            return False, err
        mode = params.get("mode", "single")
        if mode == "single":
            result = client.generate_single_speaker(
                text=text,
                model=params.get("model", "gemini-2.5-flash-preview-tts"),
                voice=params.get("voice", "Kore"),
                style=params.get("style"),
                output_file=out_path,
            )
        else:
            reqs = ["speaker1_name", "voice1", "speaker2_name", "voice2"]
            miss = [r for r in reqs if not params.get(r)]
            if miss:
                return False, f"Missing parameters for multi: {', '.join(miss)}"
            result = client.generate_multi_speaker(
                text=text,
                speaker1_name=params["speaker1_name"],
                voice1=params["voice1"],
                speaker2_name=params["speaker2_name"],
                voice2=params["voice2"],
                model=params.get("model", "gemini-2.5-flash-preview-tts"),
                output_file=out_path,
            )
        if not result or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return False, "Gemini generation failed or empty"
        return True, ""
    except Exception as e:
        return False, f"Gemini error: {e}"

# -------- Validation --------

def validate_input(job_input: dict) -> tuple[bool, str]:
    if "service" not in job_input:
        return False, "Missing 'service'"
    srv = job_input["service"].lower()
    if srv not in TTS_SERVICES:
        return False, f"Invalid service '{srv}'"
    if srv == "openai":
        if "model" in job_input and job_input["model"] not in OPENAI_MODELS:
            return False, "Invalid OpenAI model"
        if "voice" in job_input and job_input["voice"] not in OPENAI_VOICES:
            return False, "Invalid OpenAI voice"
        if "speed" in job_input:
            try:
                sp = float(job_input["speed"])
                if not (0.25 <= sp <= 4.0):
                    return False, "OpenAI speed must be 0.25-4.0"
            except Exception:
                return False, "OpenAI speed must be a number"
        if "preset_instructions" in job_input and job_input["preset_instructions"] not in OPENAI_INSTRUCTION_PRESETS:
            return False, "Invalid preset_instructions"
        if "response_format" in job_input and job_input["response_format"] not in OPENAI_FORMATS:
            return False, "Invalid response_format"
        if "batch" in job_input and not (isinstance(job_input["batch"], list) or (isinstance(job_input["batch"], dict) and "texts" in job_input["batch"])):
            return False, "batch must be list or {'texts': [...]}"

    if srv == "gemini":
        if "model" in job_input and job_input["model"] not in GEMINI_MODELS:
            return False, "Invalid Gemini model"
        if "mode" in job_input and job_input["mode"] not in ["single", "multi"]:
            return False, "Gemini mode must be 'single' or 'multi'"
        if job_input.get("mode") == "single" and "voice" in job_input and job_input["voice"] not in GEMINI_VOICES:
            return False, "Invalid Gemini voice"
        if job_input.get("mode") == "multi":
            for p in ["speaker1_name", "voice1", "speaker2_name", "voice2"]:
                if not job_input.get(p):
                    return False, f"Missing parameter: {p}"
            if job_input["voice1"] not in GEMINI_VOICES or job_input["voice2"] not in GEMINI_VOICES:
                return False, "Invalid Gemini voice1/voice2"
        if "style" in job_input and job_input["style"] not in GEMINI_STYLES:
            return False, "Invalid Gemini style"
    return True, "OK"

# -------- Handler --------

def object_prefix(service: str, data: dict) -> str:
    # Hỗ trợ 'output_dir' (từ CLI) để map vào prefix MinIO nếu truyền vào
    user_dir = data.get("output_dir")
    date_dir = datetime.now().strftime('%Y/%m/%d')
    base = f"tts/{service}/{date_dir}"
    return f"{base}/{user_dir.strip('/')} " if user_dir else base

def infer_filename(default_name: str, data: dict, expected_ext: str) -> str:
    # Hỗ trợ 'output' (từ CLI) làm tên file nếu có; nếu không, sinh ngẫu nhiên
    out = data.get("output")
    if out:
        name = Path(out).name
        if '.' not in name:
            name = f"{name}.{expected_ext}"
        return name
    return default_name

async def handler(job: dict):
    job_id = job.get("id", f"job_{uuid.uuid4().hex[:8]}")
    t0 = time.time()
    try:
        data = job.get("input", {}) or {}
        ok, msg = validate_input(data)
        if not ok:
            return {"status": "failed", "job_id": job_id, "error": msg, "processing_time_seconds": round(time.time() - t0, 2)}

        service = data["service"].lower()
        api_key = data.get("api_key") or os.getenv(f"{service.upper()}_API_KEY")

        # Các lệnh list tương tự CLI
        if service == "openai" and data.get("list_models"):
            return {"status": "completed", "job_id": job_id, "models": OPENAI_MODELS}
        if service == "openai" and data.get("list_voices"):
            return {"status": "completed", "job_id": job_id, "voices": OPENAI_VOICES}
        if service == "openai" and data.get("list_instructions"):
            return {"status": "completed", "job_id": job_id, "instruction_presets": OPENAI_INSTRUCTION_PRESETS}
        if service == "edge" and data.get("list"):
            # Trả danh sách VOICES rút gọn nếu có, nếu không chỉ trả gợi ý voice chuẩn
            return {"status": "completed", "job_id": job_id, "voices": list(EDGE_VOICES.values()) or ["vi-VN-HoaiMyNeural"]}
        if service == "gemini" and data.get("list_voices"):
            return {"status": "completed", "job_id": job_id, "voices": GEMINI_VOICES}
        if service == "gemini" and data.get("list_styles"):
            return {"status": "completed", "job_id": job_id, "styles": GEMINI_STYLES}
        if service == "gemini" and data.get("list_models"):
            return {"status": "completed", "job_id": job_id, "models": GEMINI_MODELS}

        client = get_tts_client(service, api_key)

        with tempfile.TemporaryDirectory() as tmp:
            prefix = object_prefix(service, data)

            # OpenAI batch
            if service == "openai" and "batch" in data:
                success_items, fail_items = await openai_batch(client, data, tmp, prefix)
                return {
                    "status": "completed" if success_items else "failed",
                    "job_id": job_id,
                    "batch_success": success_items,
                    "batch_fail": fail_items,
                    "processing_time_seconds": round(time.time() - t0, 2),
                }

            # OpenAI preview
            if service == "openai" and data.get("preview"):
                preview_name = infer_filename(f"openai_preview_{uuid.uuid4().hex[:8]}.mp3", data, "mp3")
                local = os.path.join(tmp, preview_name)
                ok, err = await openai_preview(client, data, local)
                if not ok:
                    return {"status": "failed", "job_id": job_id, "error": "OpenAI preview failed", "error_detail": err}
                obj = f"{prefix}/{preview_name}"
                ps, pu = ensure_bucket_and_upload(local, obj)
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "service": "openai",
                    "mode": "preview",
                    "output_audio_url": ps,
                    "public_url": pu,
                    "object_name": obj,
                    "size_bytes": os.path.getsize(local),
                    "processing_time_seconds": round(time.time() - t0, 2),
                }

            # Single file theo service
            if service == "openai":
                ofmt = data.get("response_format", "mp3")
                file_name = infer_filename(f"openai_tts_{uuid.uuid4().hex[:8]}.{ofmt}", data, ofmt)
                local = os.path.join(tmp, file_name)
                ok, err = await openai_generate_one(client, data, local)
                if not ok:
                    return {
                        "status": "failed",
                        "job_id": job_id,
                        "service": "openai",
                        "error": "TTS generation failed",
                        "error_detail": err or "OpenAI generation returned False or file not created",
                        "generation_time_seconds": round(time.time() - t0, 2),
                        "processing_time_seconds": round(time.time() - t0, 2),
                    }
                obj = f"{prefix}/{file_name}"
                ps, pu = ensure_bucket_and_upload(local, obj)
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "service": "openai",
                    "output_audio_url": ps,
                    "public_url": pu,
                    "object_name": obj,
                    "size_bytes": os.path.getsize(local),
                    "processing_time_seconds": round(time.time() - t0, 2),
                }

            if service == "edge":
                file_name = infer_filename(f"edge_tts_{uuid.uuid4().hex[:8]}.mp3", data, "mp3")
                local = os.path.join(tmp, file_name)
                ok, err = await generate_tts_edge(client, data, local)
                if not ok:
                    return {"status": "failed", "job_id": job_id, "service": "edge", "error": "TTS generation failed", "error_detail": err}
                obj = f"{prefix}/{file_name}"
                ps, pu = ensure_bucket_and_upload(local, obj)
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "service": "edge",
                    "output_audio_url": ps,
                    "public_url": pu,
                    "object_name": obj,
                    "size_bytes": os.path.getsize(local),
                    "processing_time_seconds": round(time.time() - t0, 2),
                }

            if service == "gemini":
                file_name = infer_filename(f"gemini_tts_{uuid.uuid4().hex[:8]}.wav", data, "wav")
                local = os.path.join(tmp, file_name)
                ok, err = generate_tts_gemini(client, data, local)
                if not ok:
                    return {"status": "failed", "job_id": job_id, "service": "gemini", "error": "TTS generation failed", "error_detail": err}
                obj = f"{prefix}/{file_name}"
                ps, pu = ensure_bucket_and_upload(local, obj)
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "service": "gemini",
                    "output_audio_url": ps,
                    "public_url": pu,
                    "object_name": obj,
                    "size_bytes": os.path.getsize(local),
                    "processing_time_seconds": round(time.time() - t0, 2),
                }

    except Exception as e:
        return {
            "status": "failed",
            "job_id": job_id,
            "error": "Handler internal error",
            "error_detail": str(e),
            "traceback": traceback.format_exc()[-1500:],
            "processing_time_seconds": round(time.time() - t0, 2),
        }
    finally:
        gc.collect()
        if len(tts_clients) > 16:
            tts_clients.clear()

def health_check() -> tuple[bool, str]:
    issues = []
    try:
        if not minio_client:
            issues.append("MinIO not initialized")
        else:
            try:
                minio_client.bucket_exists(MINIO_BUCKET)
            except Exception as e:
                issues.append(f"MinIO connection failed: {e}")
    except Exception as e:
        issues.append(f"MinIO init error: {e}")
    try:
        from openai import AsyncOpenAI  # Kiểm tra import SDK [14]
    except Exception as e:
        issues.append(f"openai import failed: {e}")
    try:
        EdgeTTS()  # Kiểm tra khởi tạo EdgeTTS [4]
    except Exception as e:
        issues.append(f"EdgeTTS init failed: {e}")
    try:
        from google import genai  # Kiểm tra import Gemini [3]
    except Exception as e:
        issues.append(f"google-genai import failed: {e}")
    if issues:
        return False, "; ".join(issues)
    return True, "All systems operational"

if __name__ == "__main__":
    ok, msg = health_check()
    if not ok:
        logger.warning(f"Health check warning: {msg}")
    else:
        logger.info(f"Health OK: {msg}")
    runpod.serverless.start({"handler": handler})
