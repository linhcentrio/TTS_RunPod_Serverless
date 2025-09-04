#!/usr/bin/env python3
"""
RunPod Serverless Handler cho TTS Services - Non-GPU Version
Hỗ trợ OpenAI TTS, EdgeTTS, và Gemini TTS với MinIO storage
Tham khảo từ wan_handler.py structure
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

# MinIO Configuration - Cố định như yêu cầu
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client với error handling
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    logger.info("✅ MinIO client initialized")
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
    """Get or create TTS client with caching"""
    try:
        cache_key = f"{service}_{api_key}" if api_key else service
        
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

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO storage với error handling"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")
            
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
            
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading to MinIO: {object_name} ({file_size_mb:.1f}MB)")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        # Generate presigned URL (7 days expiry)
        presigned_url = minio_client.presigned_get_object(
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            expires=timedelta(days=7)
        )
        
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        
        logger.info(f"✅ Upload completed: {file_url}")
        return presigned_url, file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

async def generate_tts_openai(client: OpenAITTSCLI, params: dict, output_path: str) -> bool:
    """Generate TTS using OpenAI"""
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
        
        return success
        
    except Exception as e:
        logger.error(f"❌ OpenAI TTS generation failed: {e}")
        return False

async def generate_tts_edge(client: EdgeTTS, params: dict, output_path: str) -> bool:
    """Generate TTS using EdgeTTS"""
    try:
        logger.info("🎙️ Generating TTS with EdgeTTS...")
        
        audio_data = await quick_tts(
            text=params["text"],
            voice=params.get("voice", "vi-VN-HoaiMyNeural"),
            output_file=output_path
        )
        
        return os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"❌ EdgeTTS generation failed: {e}")
        return False

def generate_tts_gemini(client: GeminiTTSCLI, params: dict, output_path: str) -> bool:
    """Generate TTS using Gemini (sync)"""
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
            raise ValueError(f"Invalid Gemini mode: {mode}")
            
        return result is not None
        
    except Exception as e:
        logger.error(f"❌ Gemini TTS generation failed: {e}")
        return False

def validate_input_parameters(job_input: dict) -> tuple[bool, str]:
    """Enhanced input parameter validation"""
    try:
        # Required parameters
        required_params = ["service", "text"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"
        
        service = job_input["service"].lower()
        
        # Validate service
        if service not in TTS_SERVICES:
            return False, f"Invalid service. Must be one of: {', '.join(TTS_SERVICES.keys())}"
        
        service_config = TTS_SERVICES[service]
        
        # Validate API key if required
        if service_config["requires_api_key"]:
            api_key = job_input.get("api_key") or os.getenv(f"{service.upper()}_API_KEY")
            if not api_key:
                return False, f"{service_config['name']} requires API key"
        
        # Validate text length
        text = job_input["text"]
        if len(text) > 10000:  # 10K character limit
            return False, "Text too long (max 10,000 characters)"
        
        # Service-specific validations
        if service == "openai":
            model = job_input.get("model", "tts-1")
            if model not in ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]:
                return False, "Invalid OpenAI model"
                
            voice = job_input.get("voice", "alloy")
            valid_voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
            if voice not in valid_voices:
                return False, f"Invalid OpenAI voice. Must be one of: {', '.join(valid_voices)}"
                
            speed = job_input.get("speed", 1.0)
            if not (0.25 <= speed <= 4.0):
                return False, "OpenAI speed must be between 0.25 and 4.0"
                
        elif service == "gemini":
            mode = job_input.get("mode", "single")
            if mode not in ["single", "multi"]:
                return False, "Gemini mode must be 'single' or 'multi'"
                
            if mode == "multi":
                required_multi = ["speaker1_name", "voice1", "speaker2_name", "voice2"]
                for param in required_multi:
                    if param not in job_input:
                        return False, f"Multi-speaker mode requires: {param}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def clear_memory():
    """Memory cleanup"""
    gc.collect()

async def handler(job):
    """
    Main RunPod handler cho TTS Services
    
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
            "response_format": "mp3|wav|opus|aac|flac",
            
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
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Validate input parameters
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            return {
                "error": validation_message,
                "status": "failed",
                "job_id": job_id
            }
        
        # Extract parameters
        service = job_input["service"].lower()
        text = job_input["text"]
        api_key = job_input.get("api_key") or os.getenv(f"{service.upper()}_API_KEY")
        
        logger.info(f"🚀 Job {job_id}: {TTS_SERVICES[service]['name']} TTS Started")
        logger.info(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        logger.info(f"🎙️ Service: {service}")
        
        # Get TTS client
        try:
            client = get_tts_client(service, api_key)
        except Exception as e:
            return {
                "error": f"Failed to initialize {service} client: {str(e)}",
                "status": "failed",
                "job_id": job_id
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
            output_filename = f"{service}_tts_{uuid.uuid4().hex[:8]}.{output_format}"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Generate TTS
            logger.info("🎵 Generating TTS audio...")
            generation_start = time.time()
            
            success = False
            if service == "openai":
                success = await generate_tts_openai(client, job_input, output_path)
            elif service == "edge":
                success = await generate_tts_edge(client, job_input, output_path)
            elif service == "gemini":
                success = generate_tts_gemini(client, job_input, output_path)
            
            generation_time = time.time() - generation_start
            
            if not success or not os.path.exists(output_path):
                return {
                    "error": "TTS generation failed",
                    "status": "failed",
                    "job_id": job_id,
                    "generation_time_seconds": round(generation_time, 2)
                }
            
            # Get file info
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ TTS generated: {file_size_mb:.2f}MB in {generation_time:.2f}s")
            
            # Upload to MinIO
            logger.info("📤 Uploading to MinIO storage...")
            try:
                minio_object_name = f"tts/{service}/{datetime.now().strftime('%Y%m%d')}/{output_filename}"
                presigned_url, public_url = upload_to_minio(output_path, minio_object_name)
            except Exception as e:
                return {
                    "error": f"Failed to upload to storage: {str(e)}",
                    "status": "failed",
                    "job_id": job_id
                }
            
            # Calculate final statistics
            total_time = time.time() - start_time
            
            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total time: {total_time:.1f}s (generation: {generation_time:.1f}s)")
            logger.info(f"📊 Output: {file_size_mb:.2f}MB")
            
            # Prepare response
            response = {
                "output_audio_url": presigned_url,
                "public_url": public_url,
                "object_name": minio_object_name,
                "processing_time_seconds": round(total_time, 2),
                "generation_time_seconds": round(generation_time, 2),
                "audio_info": {
                    "format": output_format,
                    "file_size_mb": round(file_size_mb, 2),
                    "text_length": len(text)
                },
                "service_info": {
                    "service": service,
                    "service_name": TTS_SERVICES[service]["name"]
                },
                "parameters": {
                    "text": text[:200] + "..." if len(text) > 200 else text
                },
                "status": "completed",
                "job_id": job_id
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
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": error_msg,
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "job_id": job_id
        }
    
    finally:
        clear_memory()

def health_check():
    """Health check function"""
    try:
        # Check MinIO
        if not minio_client:
            return False, "MinIO not available"
        
        # Test MinIO connection
        try:
            minio_client.bucket_exists(MINIO_BUCKET)
        except Exception as e:
            return False, f"MinIO connection failed: {str(e)}"
        
        # Check TTS modules
        try:
            # Test EdgeTTS (không cần API key)
            EdgeTTS()
        except Exception as e:
            return False, f"EdgeTTS not available: {str(e)}"
        
        return True, "All systems operational"
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    logger.info("🚀 Starting TTS Serverless Worker (Non-GPU)...")
    logger.info(f"🎙️ Supported services: {', '.join(TTS_SERVICES.keys())}")
    
    try:
        # Health check on startup
        health_ok, health_msg = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            sys.exit(1)
        
        logger.info(f"✅ Health check passed: {health_msg}")
        logger.info("🎵 Ready to process TTS requests...")
        logger.info("📱 MinIO storage configured and ready!")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
