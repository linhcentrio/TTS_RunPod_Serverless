# test_tts_handler.py - Test script cho TTS handler
import requests
import json

def test_edge_tts():
    """Test EdgeTTS"""
    payload = {
        "input": {
            "service": "edge",
            "text": "Xin chào, đây là test EdgeTTS trên RunPod serverless",
            "voice": "vi-VN-HoaiMyNeural"
        }
    }
    
    print("🧪 Testing EdgeTTS...")
    response = requests.post("http://localhost:8000/", json=payload)
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

def test_openai_tts():
    """Test OpenAI TTS"""
    payload = {
        "input": {
            "service": "openai",
            "text": "Hello, this is OpenAI TTS test on RunPod serverless",
            "api_key": "your-openai-api-key",  # Thay bằng API key thật
            "model": "tts-1",
            "voice": "alloy",
            "speed": 1.0,
            "response_format": "mp3"
        }
    }
    
    print("🧪 Testing OpenAI TTS...")
    response = requests.post("http://localhost:8000/", json=payload)
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

def test_gemini_single():
    """Test Gemini Single Speaker"""
    payload = {
        "input": {
            "service": "gemini",
            "text": "Hello, this is Gemini TTS single speaker test",
            "api_key": "your-gemini-api-key",  # Thay bằng API key thật
            "mode": "single",
            "voice": "Kore",
            "style": "cheerful"
        }
    }
    
    print("🧪 Testing Gemini Single Speaker...")
    response = requests.post("http://localhost:8000/", json=payload)
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

def test_gemini_multi():
    """Test Gemini Multi Speaker"""
    payload = {
        "input": {
            "service": "gemini",
            "text": "John: Hello Mary! How are you today? Mary: Hi John! I'm doing great, thank you for asking.",
            "api_key": "your-gemini-api-key",  # Thay bằng API key thật
            "mode": "multi",
            "speaker1_name": "John",
            "voice1": "Puck",
            "speaker2_name": "Mary", 
            "voice2": "Kore"
        }
    }
    
    print("🧪 Testing Gemini Multi Speaker...")
    response = requests.post("http://localhost:8000/", json=payload)
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("🧪 Testing TTS Serverless Handler...")
    
    # Test EdgeTTS (không cần API key)
    test_edge_tts()
    
    # Uncomment để test với API keys
    # test_openai_tts()
    # test_gemini_single()
    # test_gemini_multi()
