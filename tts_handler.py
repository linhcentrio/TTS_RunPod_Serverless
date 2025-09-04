#!/usr/bin/env python3
"""
RunPod Serverless Handler cho TTS Services - Non-GPU Version
Hỗ trợ OpenAI TTS, EdgeTTS, và Gemini TTS với MinIO storage
PHIÊN BẢN HOÀN CHỈNH & TỐI ƯU - Khắc phục tất cả lỗi
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import sys
import gc
import json
import traceback
import asyncio
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from urllib.parse import quote, urlparse
import logging
from datetime import datetime, timedelta

# Import TTS modules
from openai_tts import OpenAITTSCLI
from edge_tts import EdgeTTS, quick_tts
from gemini_tts import GeminiTTSCLI

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MinIO Configuration - Cố định
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client với comprehensive error handling
minio_client = None
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    
    # Ensure bucket exists
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        logger.info(f"✅ Created MinIO bucket: {MINIO_BUCKET}")
    else:
        logger.info(f"✅ MinIO bucket exists: {MINIO_BUCKET}")
        
    logger.info("✅ MinIO client initialized successfully")
    
except Exception as e:
    logger.error(f"❌ MinIO initialization failed: {e}")
    minio_client = None

# TTS Service Configurations
TTS_SERVICES = {
    "openai": {
        "name": "OpenAI TTS",
        "requires_api_key": True,
        "default_model": "tts-1",
        "default_voice": "alloy",
        "default_speed": 1.0,
        "supported_formats": ["mp3", "opus", "aac", "flac", "wav", "pcm"]
    },
    "edge": {
        "name": "Microsoft Edge TTS",
        "requires_api_key": False,
        "default_voice": "vi-VN-HoaiMyNeural",
        "supported_formats": ["mp3"]
    },
    "gemini": {
        "name": "Google Gemini TTS",
        "requires_api_key": True,
        "default_model": "gemini-2.5-flash-preview-tts",
        "default_voice": "Kore",
        "default_mode": "single",
        "supported_formats": ["wav"]
    }
}

# Global TTS clients cache
tts_clients = {}

def get_tts_client(service: str, api_key: str = None):
    """Get or create TTS client with enhanced caching"""
    try:
        cache_key = f"{service}_{api_key[:10] if api_key else 'nokey'}"
        
        if cache_key not in tts_clients:
            if service == "openai":
                if not api_key:
                    raise ValueError("OpenAI API key required")
                tts_clients[cache_key] = OpenAITTSCLI(api_key=api_key)
                
            elif service == "edge":
                tts_clients[cache_key] = EdgeTTS()
                
            elif service == "gemini":
                if not api_key:
                    raise ValueError("Gemini API key required")
                tts_clients[cache_key] = GeminiTTSCLI(api_key=api_key)
                
            else:
                raise ValueError(f"Unsupported service: {service}")
                
            logger.info(f"✅ TTS client created: {service}")
        
        return tts_clients[cache_key]
        
    except Exception as e:
        logger.error(f"❌ Failed to get TTS client for {service}: {e}")
        raise e

def upload_to_minio(local_path: str, object_name: str) -> tuple[str, str]:
    """Upload file to MinIO storage với comprehensive error handling"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")
            
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
            
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading to MinIO: {object_name} ({file_size_mb:.2f}MB)")
        
        # Double-check bucket exists
        try:
            if not minio_client.bucket_exists(MINIO_BUCKET):
                minio_client.make_bucket(MINIO_BUCKET)
                logger.info(f"✅ Created missing bucket: {MINIO_BUCKET}")
        except S3Error as e:
            logger.warning(f"⚠️ Bucket check failed: {e}")
        
        # Upload file
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        # Generate URLs
        presigned_url = minio_client.presigned_get_object(
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            expires=timedelta(days=7)
        )
        
        public_url = f"{'https' if MINIO_SECURE else 'http'}://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        
        logger.info(f"✅ Upload completed successfully")
        return presigned_url, public_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

async def generate_tts_openai(client: OpenAITTSCLI, params: dict, output_path: str) -> tuple[bool, str]:
    """Generate TTS using OpenAI với detailed error handling"""
    try:
        logger.info("🎙️ Generating TTS with OpenAI...")
        
        success = await client.generate_speech(
            text=params["text"],
            output_path=output_path,
            model=params.get("model", "tts-1"),
            voice=params.get("voice", "alloy"),
            speed=params.get("speed", 1.0),
            instructions=params.get("instructions"),
            response_format=params.get("response_format", "mp3")
        )
        
        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                return True, ""
            else:
                return False, "Generated file is empty"
        else:
            return False, "OpenAI generation returned False or file not created"
            
    except Exception as e:
        error_msg = f"OpenAI TTS error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

async def generate_tts_edge(client: EdgeTTS, params: dict, output_path: str) -> tuple[bool, str]:
    """Generate TTS using EdgeTTS với detailed error handling"""
    try:
        logger.info("🎙️ Generating TTS with EdgeTTS...")
        
        audio_data = await quick_tts(
            text=params["text"],
            voice=params.get("voice", "vi-VN-HoaiMyNeural"),
            output_file=output_path
        )
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                return True, ""
            else:
                return False, "Generated EdgeTTS file is empty"
        else:
            return False, "EdgeTTS did not create output file"
            
    except Exception as e:
        error_msg = f"EdgeTTS error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def generate_tts_gemini(client: GeminiTTSCLI, params: dict, output_path: str) -> tuple[bool, str]:
    """Generate TTS using Gemini với detailed error handling"""
    try:
        logger.info("🎙️ Generating TTS with Gemini...")
        
        mode = params.get("mode", "single")
        
        if mode == "single":
            result = client.generate_single_speaker(
                text=params["text"],
                model=params.get("model", "gemini-2.5-flash-preview-tts"),
                voice=params.get("voice", "Kore"),
                style=params.get("style"),
                output_file=output_path
            )
        elif mode == "multi":
            # Validate required multi-speaker parameters
            required_params = ["speaker1_name", "voice1", "speaker2_name", "voice2"]
            missing_params = [p for p in required_params if not params.get(p)]
            if missing_params:
                return False, f"Missing multi-speaker parameters: {', '.join(missing_params)}"
                
            result = client.generate_multi_speaker(
                text=params["text"],
                speaker1_name=params.get("speaker1_name"),
                voice1=params.get("voice1"),
                speaker2_name=params.get("speaker2_name"),
                voice2=params.get("voice2"),
                model=params.get("model", "gemini-2.5-flash-preview-tts"),
                output_file=output_path
            )
        else:
            return False, f"Invalid Gemini mode: {mode}. Must be 'single' or 'multi'"
        
        if result and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                return True, ""
            else:
                return False, "Generated Gemini file is empty"
        else:
            return False, "Gemini generation failed or returned None"
            
    except Exception as e:
        error_msg = f"Gemini TTS error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def validate_input_parameters(job_input: dict) -> tuple[bool, str]:
    """Comprehensive input parameter validation"""
    try:
        # Required parameters
        required_params = ["service", "text"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"
        
        service = job_input["service"].lower()
        
        # Validate service
        if service not in TTS_SERVICES:
            return False, f"Invalid service '{service}'. Must be one of: {', '.join(TTS_SERVICES.keys())}"
        
        service_config = TTS_SERVICES[service]
        
        # Validate API key if required
        if service_config["requires_api_key"]:
            api_key = job_input.get("api_key") or os.getenv(f"{service.upper()}_API_KEY")
            if not api_key:
                return False, f"{service_config['name']} requires API key"
        
        # Validate text length
        text = job_input["text"].strip()
        if not text:
            return False, "Text cannot be empty"
        if len(text) > 50000:  # Increased limit
            return False, "Text too long (max 50,000 characters)"
        
        # Service-specific validations
        if service == "openai":
            model = job_input.get("model", "tts-1")
            valid_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
            if model not in valid_models:
                return False, f"Invalid OpenAI model '{model}'. Must be one of: {', '.join(valid_models)}"
                
            voice = job_input.get("voice", "alloy")
            valid_voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
            if voice not in valid_voices:
                return False, f"Invalid OpenAI voice '{voice}'. Must be one of: {', '.join(valid_voices)}"
                
            speed = job_input.get("speed", 1.0)
            if not isinstance(speed, (int, float)) or not (0.25 <= speed <= 4.0):
                return False, "OpenAI speed must be a number between 0.25 and 4.0"
                
            response_format = job_input.get("response_format", "mp3")
            valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
            if response_format not in valid_formats:
                return False, f"Invalid response format '{response_format}'. Must be one of: {', '.join(valid_formats)}"
                
        elif service == "gemini":
            mode = job_input.get("mode", "single")
            if mode not in ["single", "multi"]:
                return False, "Gemini mode must be 'single' or 'multi'"
                
            # Validate voices for Gemini
            gemini_voices = [
                "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
                "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
                "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
                "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
                "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
            ]
            
            if mode == "single":
                voice = job_input.get("voice", "Kore")
                if voice not in gemini_voices:
                    return False, f"Invalid Gemini voice '{voice}'. Must be one of: {', '.join(gemini_voices[:5])}..."
            elif mode == "multi":
                required_multi = ["speaker1_name", "voice1", "speaker2_name", "voice2"]
                for param in required_multi:
                    if param not in job_input or not job_input[param]:
                        return False, f"Multi-speaker mode requires parameter: {param}"
                
                for voice_param in ["voice1", "voice2"]:
                    voice = job_input[voice_param]
                    if voice not in gemini_voices:
                        return False, f"Invalid Gemini {voice_param} '{voice}'"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def clear_memory():
    """Enhanced memory cleanup"""
    try:
        gc.collect()
        # Clear client cache periodically to prevent memory buildup
        if len(tts_clients) > 10:
            tts_clients.clear()
            logger.info("🧹 Cleared TTS client cache")
    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup warning: {e}")

async def handler(job):
    """
    Main RunPod handler cho TTS Services - PHIÊN BẢN HOÀN CHỈNH
    
    Input JSON format:
    {
        "input": {
            "service": "openai|edge|gemini",
            "text": "Text to convert to speech",
            "api_key": "API key if required",
            
            // OpenAI specific
            "model": "tts-1|tts-1-hd|gpt-4o-mini-tts",
            "voice": "alloy|ash|ballad|coral|echo|fable|nova|onyx|sage|shimmer",
            "speed": 1.0,
            "instructions": "Voice instructions for gpt-4o-mini-tts",
            "response_format": "mp3|wav|opus|aac|flac|pcm",
            
            // EdgeTTS specific  
            "voice": "vi-VN-HoaiMyNeural|en-US-AriaNeural|...",
            
            // Gemini specific
            "model": "gemini-2.5-flash-preview-tts|gemini-2.5-pro-preview-tts",
            "mode": "single|multi",
            "voice": "Kore|Puck|Charon|...",
            "style": "cheerful|professional|calm|...",
            // Multi-speaker specific
            "speaker1_name": "John",
            "voice1": "Puck", 
            "speaker2_name": "Mary",
            "voice2": "Kore"
        }
    }
    """
    job_id = job.get("id", f"job_{uuid.uuid4().hex[:8]}")
    start_time = time.time()
    
    logger.info(f"🚀 Starting job {job_id}")
    
    try:
        job_input = job.get("input", {})
        
        # Validate input parameters
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            logger.error(f"❌ Validation failed: {validation_message}")
            return {
                "error": f"Validation failed: {validation_message}",
                "status": "failed",
                "job_id": job_id,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        
        # Extract parameters
        service = job_input["service"].lower()
        text = job_input["text"].strip()
        api_key = job_input.get("api_key") or os.getenv(f"{service.upper()}_API_KEY")
        
        logger.info(f"🎙️ Service: {service}")
        logger.info(f"📝 Text length: {len(text)} characters")
        logger.info(f"🔑 API key provided: {'Yes' if api_key else 'No'}")
        
        # Get TTS client
        try:
            client = get_tts_client(service, api_key)
        except Exception as e:
            error_msg = f"Failed to initialize {service} client: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "error": error_msg,
                "status": "failed",
                "job_id": job_id,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Determine output format
            if service == "openai":
                output_format = job_input.get("response_format", "mp3")
            elif service == "edge":
                output_format = "mp3"
            elif service == "gemini":
                output_format = "wav"
            
            # Generate output file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{service}_tts_{timestamp}_{uuid.uuid4().hex[:8]}.{output_format}"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Generate TTS
            logger.info("🎵 Starting TTS generation...")
            generation_start = time.time()
            
            success = False
            error_detail = ""
            
            if service == "openai":
                success, error_detail = await generate_tts_openai(client, job_input, output_path)
            elif service == "edge":
                success, error_detail = await generate_tts_edge(client, job_input, output_path)
            elif service == "gemini":
                success, error_detail = generate_tts_gemini(client, job_input, output_path)
            
            generation_time = time.time() - generation_start
            
            if not success:
                logger.error(f"❌ TTS generation failed: {error_detail}")
                return {
                    "error": "TTS generation failed",
                    "error_detail": error_detail,
                    "service": service,
                    "status": "failed",
                    "job_id": job_id,
                    "generation_time_seconds": round(generation_time, 2),
                    "processing_time_seconds": round(time.time() - start_time, 2)
                }
            
            # Validate output file
            if not os.path.exists(output_path):
                error_msg = "Output file was not created"
                logger.error(f"❌ {error_msg}")
                return {
                    "error": error_msg,
                    "error_detail": "File system error - output not found",
                    "status": "failed",
                    "job_id": job_id,
                    "generation_time_seconds": round(generation_time, 2)
                }
            
            # Get file info
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ TTS generated: {file_size_mb:.2f}MB in {generation_time:.2f}s")
            
            # Upload to MinIO
            logger.info("📤 Uploading to MinIO storage...")
            upload_start = time.time()
            
            try:
                minio_object_name = f"tts/{service}/{datetime.now().strftime('%Y/%m/%d')}/{output_filename}"
                presigned_url, public_url = upload_to_minio(output_path, minio_object_name)
                upload_time = time.time() - upload_start
                logger.info(f"✅ Upload completed in {upload_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Failed to upload to storage: {str(e)}"
                logger.error(f"❌ {error_msg}")
                return {
                    "error": error_msg,
                    "error_detail": f"MinIO upload failed: {str(e)}",
                    "status": "failed",
                    "job_id": job_id,
                    "generation_time_seconds": round(generation_time, 2)
                }
            
            # Calculate final statistics
            total_time = time.time() - start_time
            
            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total time: {total_time:.2f}s (generation: {generation_time:.2f}s, upload: {upload_time:.2f}s)")
            
            # Prepare comprehensive response
            response = {
                "status": "completed",
                "job_id": job_id,
                "output_audio_url": presigned_url,
                "public_url": public_url,
                "object_name": minio_object_name,
                "processing_time_seconds": round(total_time, 2),
                "generation_time_seconds": round(generation_time, 2),
                "upload_time_seconds": round(upload_time, 2),
                "audio_info": {
                    "format": output_format,
                    "file_size_mb": round(file_size_mb, 2),
                    "text_length": len(text),
                    "estimated_duration_seconds": round(len(text) / 15, 1)  # Rough estimate
                },
                "service_info": {
                    "service": service,
                    "service_name": TTS_SERVICES[service]["name"]
                },
                "parameters": {
                    "text_preview": text[:200] + "..." if len(text) > 200 else text
                }
            }
            
            # Add service-specific parameters
            if service == "openai":
                response["parameters"].update({
                    "model": job_input.get("model", "tts-1"),
                    "voice": job_input.get("voice", "alloy"),
                    "speed": job_input.get("speed", 1.0),
                    "instructions": job_input.get("instructions"),
                    "response_format": output_format
                })
            elif service == "edge":
                response["parameters"].update({
                    "voice": job_input.get("voice", "vi-VN-HoaiMyNeural")
                })
            elif service == "gemini":
                response["parameters"].update({
                    "model": job_input.get("model", "gemini-2.5-flash-preview-tts"),
                    "mode": job_input.get("mode", "single"),
                    "voice": job_input.get("voice", "Kore"),
                    "style": job_input.get("style")
                })
                
                if job_input.get("mode") == "multi":
                    response["parameters"].update({
                        "speaker1_name": job_input.get("speaker1_name"),
                        "voice1": job_input.get("voice1"),
                        "speaker2_name": job_input.get("speaker2_name"),
                        "voice2": job_input.get("voice2")
                    })
            
            return response
            
    except Exception as e:
        error_msg = str(e)
        total_time = time.time() - start_time
        
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": "Handler internal error",
            "error_detail": error_msg,
            "traceback": traceback.format_exc()[-1000:],  # Last 1000 chars
            "status": "failed",
            "job_id": job_id,
            "processing_time_seconds": round(total_time, 2)
        }
    
    finally:
        clear_memory()

def health_check() -> tuple[bool, str]:
    """Comprehensive health check function"""
    try:
        issues = []
        
        # Check MinIO
        if not minio_client:
            issues.append("MinIO client not available")
        else:
            try:
                bucket_exists = minio_client.bucket_exists(MINIO_BUCKET)
                if not bucket_exists:
                    issues.append(f"MinIO bucket '{MINIO_BUCKET}' does not exist")
            except Exception as e:
                issues.append(f"MinIO connection failed: {str(e)}")
        
        # Check TTS modules
        try:
            EdgeTTS()  # Test EdgeTTS (no API key needed)
        except Exception as e:
            issues.append(f"EdgeTTS not available: {str(e)}")
        
        # Check OpenAI module (without API key)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            issues.append(f"OpenAI module not available: {str(e)}")
        
        # Check Gemini module (without API key)
        try:
            from google import genai
        except ImportError as e:
            issues.append(f"Gemini module not available: {str(e)}")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "All systems operational"
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    logger.info("🚀 Starting TTS Serverless Worker (Non-GPU) - HOÀN CHỈNH & TỐI ƯU")
    logger.info(f"🎙️ Supported services: {', '.join(TTS_SERVICES.keys())}")
    logger.info(f"📦 MinIO endpoint: {MINIO_ENDPOINT}")
    logger.info(f"🗄️ MinIO bucket: {MINIO_BUCKET}")
    
    try:
        # Health check on startup
        health_ok, health_msg = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            logger.error("❌ Service may not function properly")
            # Don't exit, just warn
        else:
            logger.info(f"✅ Health check passed: {health_msg}")
        
        logger.info("🎵 Ready to process TTS requests...")
        logger.info("📱 MinIO storage configured and ready!")
        logger.info("🔧 All optimizations and error handling included!")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
