#!/usr/bin/env python3
"""
Gemini TTS CLI Module
Chuyển đổi văn bản thành giọng nói bằng Gemini AI
"""

import os
import wave
import argparse
import sys
from google import genai
from google.genai import types
import tempfile
import shutil
from datetime import datetime
from pathlib import Path


class GeminiTTSCLI:
    def __init__(self, api_key=None):
        """Khởi tạo ứng dụng TTS với Gemini API"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("❌ Cần cung cấp GEMINI_API_KEY trong biến môi trường hoặc tham số --api-key")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Danh sách 30 voice options từ tài liệu
        self.voice_options = [
            "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
            "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
            "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
            "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
            "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
        ]
        
        # Model options
        self.models = {
            "gemini-2.5-pro-preview-tts": "Gemini 2.5 Pro Preview TTS",
            "gemini-2.5-flash-preview-tts": "Gemini 2.5 Flash Preview TTS"
        }
        
        # Style Prompt Presets
        self.style_presets = {
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
            "meditation": "Speak as a meditation guide with calm peaceful voice"
        }

    def save_wave_file(self, filename, pcm_data, channels=1, rate=24000, sample_width=2):
        """Lưu dữ liệu PCM thành file WAV"""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)

    def create_output_filename(self, prefix="gemini_tts", extension="wav"):
        """Tạo tên file output với timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"

    def generate_single_speaker(self, text, model="gemini-2.5-flash-preview-tts", 
                               voice="Kore", style=None, output_file=None):
        """Tạo audio TTS cho single speaker"""
        try:
            # Áp dụng style nếu có
            final_text = text
            if style and style in self.style_presets:
                style_prompt = self.style_presets[style]
                if style_prompt:
                    final_text = f"{style_prompt}: {text}"
                    print(f"🎭 Áp dụng style: {style}")

            print(f"🤖 Sử dụng model: {self.models.get(model, model)}")
            print(f"🎵 Sử dụng voice: {voice}")
            print("⏳ Đang tạo audio...")

            response = self.client.models.generate_content(
                model=model,
                contents=final_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            ),
                        ),
                    ),
                ),
            )

            # Lấy dữ liệu audio
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Tạo tên file output
            if not output_file:
                output_file = self.create_output_filename("single_speaker")
            
            # Lưu file
            self.save_wave_file(output_file, audio_data)
            print(f"✅ Audio đã được lưu: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
            return None

    def generate_multi_speaker(self, text, speaker1_name, voice1, speaker2_name, voice2,
                              model="gemini-2.5-flash-preview-tts", output_file=None):
        """Tạo audio TTS cho multi-speaker"""
        try:
            # Validate speakers trong text
            if not (speaker1_name in text and speaker2_name in text):
                print(f"❌ Text phải chứa tên của cả hai speaker: {speaker1_name} và {speaker2_name}")
                return None

            print(f"🤖 Sử dụng model: {self.models.get(model, model)}")
            print(f"👥 Speaker 1: {speaker1_name} ({voice1})")
            print(f"👥 Speaker 2: {speaker2_name} ({voice2})")
            print("⏳ Đang tạo audio multi-speaker...")

            response = self.client.models.generate_content(
                model=model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=[
                                types.SpeakerVoiceConfig(
                                    speaker=speaker1_name,
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=voice1,
                                        ),
                                    ),
                                ),
                                types.SpeakerVoiceConfig(
                                    speaker=speaker2_name,
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=voice2,
                                        ),
                                    ),
                                ),
                            ]
                        ),
                    ),
                ),
            )

            # Lấy dữ liệu audio
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Tạo tên file output
            if not output_file:
                output_file = self.create_output_filename("multi_speaker")
            
            # Lưu file
            self.save_wave_file(output_file, audio_data)
            print(f"✅ Audio multi-speaker đã được lưu: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
            return None

    def list_voices(self):
        """Hiển thị danh sách các voice có sẵn"""
        print("🎵 Danh sách voices có sẵn:")
        for i, voice in enumerate(self.voice_options, 1):
            print(f"  {i:2d}. {voice}")

    def list_styles(self):
        """Hiển thị danh sách style presets"""
        print("🎭 Danh sách style presets:")
        for key, description in self.style_presets.items():
            if description:
                print(f"  {key:15s} - {description}")
            else:
                print(f"  {key:15s} - Mặc định")

    def list_models(self):
        """Hiển thị danh sách models có sẵn"""
        print("🤖 Danh sách models có sẵn:")
        for model_id, name in self.models.items():
            print(f"  {model_id}")
            print(f"    -> {name}")


def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Gemini TTS CLI - Chuyển đổi văn bản thành giọng nói",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Single speaker cơ bản
  python gemini_tts.py single "Xin chào, đây là Gemini TTS"
  
  # Single speaker với style và voice
  python gemini_tts.py single "Tin tức hôm nay" --voice Kore --style news
  
  # Multi-speaker
  python gemini_tts.py multi "John: Chào bạn!\\nMary: Chào John!" --speaker1 John --voice1 Puck --speaker2 Mary --voice2 Kore
  
  # Đọc từ file
  python gemini_tts.py single --input script.txt --output output.wav
  
  # Hiển thị danh sách
  python gemini_tts.py --list-voices
  python gemini_tts.py --list-styles
        """
    )

    # Global arguments
    parser.add_argument('--api-key', help='Gemini API key (hoặc dùng biến môi trường GEMINI_API_KEY)')
    parser.add_argument('--list-voices', action='store_true', help='Hiển thị danh sách voices')
    parser.add_argument('--list-styles', action='store_true', help='Hiển thị danh sách style presets')
    parser.add_argument('--list-models', action='store_true', help='Hiển thị danh sách models')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Chế độ hoạt động')
    
    # Single speaker command
    single_parser = subparsers.add_parser('single', help='Tạo audio single speaker')
    single_parser.add_argument('text', nargs='?', help='Văn bản cần chuyển đổi')
    single_parser.add_argument('--input', '-i', help='File văn bản input')
    single_parser.add_argument('--output', '-o', help='File audio output')
    single_parser.add_argument('--voice', '-v', default='Kore', help='Tên voice (mặc định: Kore)')
    single_parser.add_argument('--style', '-s', help='Style preset')
    single_parser.add_argument('--model', '-m', default='gemini-2.5-flash-preview-tts', 
                              help='Model ID (mặc định: gemini-2.5-flash-preview-tts)')
    
    # Multi-speaker command
    multi_parser = subparsers.add_parser('multi', help='Tạo audio multi-speaker')
    multi_parser.add_argument('text', nargs='?', help='Văn bản hội thoại')
    multi_parser.add_argument('--input', '-i', help='File văn bản input')
    multi_parser.add_argument('--output', '-o', help='File audio output')
    multi_parser.add_argument('--speaker1', required=True, help='Tên speaker 1')
    multi_parser.add_argument('--voice1', required=True, help='Voice cho speaker 1')
    multi_parser.add_argument('--speaker2', required=True, help='Tên speaker 2')
    multi_parser.add_argument('--voice2', required=True, help='Voice cho speaker 2')
    multi_parser.add_argument('--model', '-m', default='gemini-2.5-flash-preview-tts',
                              help='Model ID (mặc định: gemini-2.5-flash-preview-tts)')

    args = parser.parse_args()

    try:
        # Khởi tạo TTS client
        tts = GeminiTTSCLI(api_key=args.api_key)
        
        # Handle list commands
        if args.list_voices:
            tts.list_voices()
            return
        
        if args.list_styles:
            tts.list_styles()
            return
            
        if args.list_models:
            tts.list_models()
            return

        # Validate command
        if not args.command:
            parser.print_help()
            return

        # Get text input
        text = None
        if hasattr(args, 'input') and args.input:
            if not Path(args.input).exists():
                print(f"❌ File không tồn tại: {args.input}")
                return
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        elif hasattr(args, 'text') and args.text:
            text = args.text
        else:
            print("❌ Cần cung cấp văn bản qua tham số hoặc file --input")
            return

        if not text:
            print("❌ Văn bản trống")
            return

        # Execute command
        if args.command == 'single':
            # Validate voice
            if args.voice not in tts.voice_options:
                print(f"❌ Voice không hợp lệ: {args.voice}")
                print("Sử dụng --list-voices để xem danh sách")
                return
            
            # Validate style
            if args.style and args.style not in tts.style_presets:
                print(f"❌ Style không hợp lệ: {args.style}")
                print("Sử dụng --list-styles để xem danh sách")
                return
            
            result = tts.generate_single_speaker(
                text=text,
                model=args.model,
                voice=args.voice,
                style=args.style,
                output_file=args.output
            )
            
        elif args.command == 'multi':
            # Validate voices
            if args.voice1 not in tts.voice_options:
                print(f"❌ Voice1 không hợp lệ: {args.voice1}")
                return
            if args.voice2 not in tts.voice_options:
                print(f"❌ Voice2 không hợp lệ: {args.voice2}")
                return
            
            result = tts.generate_multi_speaker(
                text=text,
                speaker1_name=args.speaker1,
                voice1=args.voice1,
                speaker2_name=args.speaker2,
                voice2=args.voice2,
                model=args.model,
                output_file=args.output
            )

    except ValueError as e:
        print(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Đã hủy bỏ")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
