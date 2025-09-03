#!/usr/bin/env python3
"""
OpenAI TTS CLI Module
Chuyển đổi văn bản thành giọng nói bằng OpenAI TTS API
"""

import os
import argparse
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


class OpenAITTSCLI:
    def __init__(self, api_key: Optional[str] = None):
        """Khởi tạo OpenAI TTS CLI"""
        if not HAS_OPENAI:
            raise ImportError("❌ Cần cài đặt: pip install openai")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("❌ Cần cung cấp OPENAI_API_KEY trong biến môi trường hoặc tham số --api-key")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Available models
        self.models = {
            "tts-1": {
                "name": "TTS-1 (Nhanh)",
                "description": "Tốc độ cao, chất lượng tiêu chuẩn",
                "supports_instructions": False
            },
            "tts-1-hd": {
                "name": "TTS-1 HD (Chất lượng cao)",
                "description": "Chất lượng cao, tốc độ chậm hơn",
                "supports_instructions": False
            },
            "gpt-4o-mini-tts": {
                "name": "GPT-4o Mini TTS (Thông minh)",
                "description": "Model mới nhất, hỗ trợ hướng dẫn chi tiết",
                "supports_instructions": True
            }
        }
        
        # Available voices with descriptions
        self.voices = {
            "alloy": "Giọng trung tính, cân bằng",
            "ash": "Giọng nam trưởng thành, hơi trầm, phù hợp phim tài liệu",
            "ballad": "Giọng nữ mềm mại, ấm áp, phù hợp nội dung tư vấn",
            "coral": "Giọng nữ trẻ, rõ ràng, tự tin, phù hợp giáo dục",
            "echo": "Giọng nam trẻ, năng động, phù hợp quảng cáo",
            "fable": "Giọng nam uy tín, phù hợp thông báo chính thức",
            "nova": "Giọng nữ chuyên nghiệp, phù hợp tin tức",
            "onyx": "Giọng nam trầm, sang trọng, phù hợp thuyết trình",
            "sage": "Giọng nữ từng trải, ấm áp, phù hợp podcast",
            "shimmer": "Giọng nữ tươi sáng, năng động, phù hợp giải trí"
        }
        
        # Instruction presets for gpt-4o-mini-tts
        self.instruction_presets = {
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
            "educational": "Speak like a teacher explaining concepts clearly"
        }

    async def generate_speech(self, text: str, output_path: str, 
                            model: str = "tts-1", voice: str = "alloy", 
                            speed: float = 1.0, instructions: Optional[str] = None,
                            response_format: str = "mp3") -> bool:
        """Tạo speech từ text"""
        try:
            print(f"🤖 Model: {self.models[model]['name']}")
            print(f"🎵 Voice: {voice} - {self.voices.get(voice, 'Không có mô tả')}")
            print(f"⚡ Speed: {speed}x")
            if instructions and self.models[model]['supports_instructions']:
                print(f"📝 Instructions: {instructions}")
            print("⏳ Đang tạo audio...")

            # Prepare parameters
            params = {
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": response_format
            }
            
            # Add speed for non-gpt-4o-mini-tts models
            if not self.models[model]['supports_instructions']:
                params["speed"] = speed
            
            # Add instructions for gpt-4o-mini-tts
            if instructions and self.models[model]['supports_instructions']:
                params["instructions"] = instructions

            # Generate speech
            async with self.client.audio.speech.with_streaming_response.create(**params) as response:
                if response_format == "pcm":
                    # Handle PCM format
                    temp_pcm = output_path + ".pcm"
                    with open(temp_pcm, 'wb') as f:
                        async for chunk in response.iter_bytes():
                            f.write(chunk)
                    
                    # Convert PCM to MP3 using ffmpeg
                    import subprocess
                    result = subprocess.run([
                        "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
                        "-i", temp_pcm, "-acodec", "libmp3lame", "-b:a", "192k", 
                        output_path.replace('.pcm', '.mp3')
                    ], capture_output=True)
                    
                    os.remove(temp_pcm)
                    if result.returncode != 0:
                        print(f"❌ Lỗi chuyển đổi PCM: {result.stderr.decode()}")
                        return False
                    
                    output_path = output_path.replace('.pcm', '.mp3')
                else:
                    # Handle other formats (mp3, wav, etc.)
                    with open(output_path, 'wb') as f:
                        async for chunk in response.iter_bytes():
                            f.write(chunk)

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Tạo thành công: {output_path} ({file_size:.2f} MB)")
            return True

        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
            return False

    async def preview_speech(self, text: str, model: str = "tts-1", 
                           voice: str = "alloy", instructions: Optional[str] = None) -> Optional[bytes]:
        """Tạo preview ngắn để test voice"""
        try:
            # Limit preview text
            preview_text = text[:200] + "..." if len(text) > 200 else text
            
            params = {
                "model": model,
                "input": preview_text,
                "voice": voice,
                "response_format": "mp3"
            }
            
            if instructions and self.models[model]['supports_instructions']:
                params["instructions"] = instructions
            else:
                params["speed"] = 1.0

            async with self.client.audio.speech.with_streaming_response.create(**params) as response:
                audio_data = b""
                async for chunk in response.iter_bytes():
                    audio_data += chunk
                return audio_data

        except Exception as e:
            print(f"❌ Lỗi preview: {str(e)}")
            return None

    async def batch_generate(self, texts: List[str], output_dir: str, 
                           model: str = "tts-1", voice: str = "alloy",
                           speed: float = 1.0, instructions: Optional[str] = None,
                           prefix: str = "tts") -> List[str]:
        """Tạo nhiều file audio cùng lúc"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        successful_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🚀 Bắt đầu xử lý {len(texts)} đoạn văn bản...")
        
        for i, text in enumerate(texts, 1):
            output_file = output_dir / f"{prefix}_{timestamp}_{i:03d}.mp3"
            print(f"\n📝 Xử lý {i}/{len(texts)}: {text[:50]}...")
            
            success = await self.generate_speech(
                text=text,
                output_path=str(output_file),
                model=model,
                voice=voice,
                speed=speed,
                instructions=instructions
            )
            
            if success:
                successful_files.append(str(output_file))
            else:
                print(f"❌ Thất bại: {i}/{len(texts)}")
        
        print(f"\n✅ Hoàn thành: {len(successful_files)}/{len(texts)} file thành công")
        return successful_files

    def create_output_filename(self, prefix: str = "openai_tts", extension: str = "mp3") -> str:
        """Tạo tên file output với timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"

    def list_models(self):
        """Hiển thị danh sách models"""
        print("🤖 Danh sách OpenAI TTS Models:")
        for model_id, info in self.models.items():
            print(f"  {model_id}")
            print(f"    -> {info['name']}")
            print(f"    -> {info['description']}")
            print(f"    -> Hỗ trợ instructions: {'Có' if info['supports_instructions'] else 'Không'}")
            print()

    def list_voices(self):
        """Hiển thị danh sách voices"""
        print("🎵 Danh sách OpenAI TTS Voices:")
        for voice, description in self.voices.items():
            print(f"  {voice:10s} - {description}")

    def list_instructions(self):
        """Hiển thị danh sách instruction presets"""
        print("📝 Danh sách Instruction Presets (chỉ cho gpt-4o-mini-tts):")
        for preset, instruction in self.instruction_presets.items():
            print(f"  {preset:15s} - {instruction}")

    async def interactive_mode(self):
        """Chế độ tương tác"""
        print("🎙️ Chế độ tương tác OpenAI TTS (gõ 'quit' để thoát)")
        
        # Default settings
        current_model = "tts-1"
        current_voice = "alloy"
        current_speed = 1.0
        current_instructions = None
        
        while True:
            print(f"\n⚙️ Cài đặt hiện tại: Model={current_model}, Voice={current_voice}, Speed={current_speed}x")
            if current_instructions:
                print(f"📝 Instructions: {current_instructions}")
            
            command = input("\n💬 Nhập lệnh (text/set/help/quit): ").strip()
            
            if command.lower() == 'quit':
                break
            elif command.lower() == 'help':
                print("""
Lệnh có sẵn:
  text <văn bản>     - Tạo TTS từ văn bản
  set model <model>  - Đổi model
  set voice <voice>  - Đổi voice  
  set speed <speed>  - Đổi tốc độ
  set instructions <text> - Đặt hướng dẫn (chỉ gpt-4o-mini-tts)
  models            - Hiển thị danh sách models
  voices            - Hiển thị danh sách voices
  instructions      - Hiển thị presets instructions
  quit              - Thoát
                """)
            elif command.startswith('set '):
                parts = command.split(' ', 2)
                if len(parts) >= 3:
                    setting, value = parts[1], parts[2]
                    if setting == 'model' and value in self.models:
                        current_model = value
                        print(f"✅ Đã đổi model: {value}")
                    elif setting == 'voice' and value in self.voices:
                        current_voice = value
                        print(f"✅ Đã đổi voice: {value}")
                    elif setting == 'speed':
                        try:
                            speed = float(value)
                            if 0.25 <= speed <= 4.0:
                                current_speed = speed
                                print(f"✅ Đã đổi speed: {speed}x")
                            else:
                                print("❌ Speed phải từ 0.25 đến 4.0")
                        except:
                            print("❌ Speed không hợp lệ")
                    elif setting == 'instructions':
                        if self.models[current_model]['supports_instructions']:
                            current_instructions = value
                            print(f"✅ Đã đặt instructions: {value}")
                        else:
                            print(f"❌ Model {current_model} không hỗ trợ instructions")
                    else:
                        print("❌ Cài đặt không hợp lệ")
                else:
                    print("❌ Lệnh set cần đủ tham số")
            elif command == 'models':
                self.list_models()
            elif command == 'voices':
                self.list_voices()
            elif command == 'instructions':
                self.list_instructions()
            elif command.startswith('text '):
                text = command[5:]
                if text:
                    output_file = self.create_output_filename()
                    success = await self.generate_speech(
                        text=text,
                        output_path=output_file,
                        model=current_model,
                        voice=current_voice,
                        speed=current_speed,
                        instructions=current_instructions
                    )
                    if success:
                        print(f"🎵 File đã lưu: {output_file}")
                else:
                    print("❌ Cần nhập văn bản")
            else:
                print("❌ Lệnh không hợp lệ. Gõ 'help' để xem hướng dẫn")


async def main():
    parser = argparse.ArgumentParser(
        description="🎙️ OpenAI TTS CLI - Chuyển đổi văn bản thành giọng nói",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # TTS cơ bản
  python openai_tts.py "Xin chào thế giới"
  
  # TTS với voice và model cụ thể
  python openai_tts.py "Tin tức hôm nay" --voice nova --model tts-1-hd
  
  # TTS với instructions (chỉ gpt-4o-mini-tts)
  python openai_tts.py "Chào mừng" --model gpt-4o-mini-tts --instructions "Speak cheerfully"
  
  # Đọc từ file
  python openai_tts.py --input script.txt --output audio.mp3 --voice coral
  
  # Batch processing
  python openai_tts.py --batch texts.json --output-dir ./output
  
  # Preview voice
  python openai_tts.py "Test voice" --preview --voice shimmer
  
  # Chế độ tương tác
  python openai_tts.py --interactive
        """
    )

    # Global arguments
    parser.add_argument('text', nargs='?', help='Văn bản cần chuyển đổi')
    parser.add_argument('--api-key', help='OpenAI API key (hoặc dùng biến môi trường OPENAI_API_KEY)')
    parser.add_argument('--input', '-i', help='File văn bản input')
    parser.add_argument('--output', '-o', help='File audio output')
    parser.add_argument('--batch', help='File JSON chứa danh sách văn bản để xử lý hàng loạt')
    parser.add_argument('--output-dir', help='Thư mục output cho batch processing', default='./output')
    
    # Model and voice settings
    parser.add_argument('--model', '-m', default='tts-1', 
                       choices=['tts-1', 'tts-1-hd', 'gpt-4o-mini-tts'],
                       help='Model TTS (mặc định: tts-1)')
    parser.add_argument('--voice', '-v', default='alloy',
                       choices=['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer'],
                       help='Voice (mặc định: alloy)')
    parser.add_argument('--speed', '-s', type=float, default=1.0, 
                       help='Tốc độ đọc 0.25-4.0 (mặc định: 1.0, chỉ cho tts-1/tts-1-hd)')
    parser.add_argument('--instructions', help='Hướng dẫn giọng điệu (chỉ cho gpt-4o-mini-tts)')
    parser.add_argument('--preset-instructions', choices=list(OpenAITTSCLI({}).instruction_presets.keys()),
                       help='Preset instructions có sẵn')
    parser.add_argument('--response-format', default='mp3', choices=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
                       help='Định dạng audio output (mặc định: mp3)')
    
    # Utility arguments
    parser.add_argument('--preview', action='store_true', help='Chỉ tạo preview ngắn để test')
    parser.add_argument('--interactive', action='store_true', help='Chế độ tương tác')
    parser.add_argument('--list-models', action='store_true', help='Hiển thị danh sách models')
    parser.add_argument('--list-voices', action='store_true', help='Hiển thị danh sách voices')
    parser.add_argument('--list-instructions', action='store_true', help='Hiển thị danh sách instruction presets')

    args = parser.parse_args()

    try:
        # Initialize TTS client - pass empty dict if no api_key to avoid error in __init__
        if args.api_key or os.getenv('OPENAI_API_KEY'):
            tts = OpenAITTSCLI(api_key=args.api_key)
        else:
            # Just for listing info, don't need API key
            if args.list_models or args.list_voices or args.list_instructions:
                tts = type('MockTTS', (), {})()
                tts.models = OpenAITTSCLI.__dict__['__init__'].__defaults__[0] if hasattr(OpenAITTSCLI.__dict__['__init__'], '__defaults__') else OpenAITTSCLI({'dummy': 'key'}).models
                tts.voices = OpenAITTSCLI({'dummy': 'key'}).voices
                tts.instruction_presets = OpenAITTSCLI({'dummy': 'key'}).instruction_presets
                tts.list_models = lambda: OpenAITTSCLI({'dummy': 'key'}).list_models()
                tts.list_voices = lambda: OpenAITTSCLI({'dummy': 'key'}).list_voices()
                tts.list_instructions = lambda: OpenAITTSCLI({'dummy': 'key'}).list_instructions()
            else:
                tts = OpenAITTSCLI(api_key=args.api_key)
        
        # Handle list commands
        if args.list_models:
            tts.list_models()
            return
        
        if args.list_voices:
            tts.list_voices()
            return
            
        if args.list_instructions:
            tts.list_instructions()
            return

        # Interactive mode
        if args.interactive:
            await tts.interactive_mode()
            return

        # Validate speed
        if not (0.25 <= args.speed <= 4.0):
            print("❌ Speed phải từ 0.25 đến 4.0")
            return

        # Handle instructions
        instructions = None
        if args.preset_instructions:
            instructions = tts.instruction_presets[args.preset_instructions]
        elif args.instructions:
            instructions = args.instructions

        # Validate instructions usage
        if instructions and not tts.models[args.model]['supports_instructions']:
            print(f"❌ Model {args.model} không hỗ trợ instructions. Chỉ gpt-4o-mini-tts mới hỗ trợ.")
            return

        # Get text input
        text = None
        if args.batch:
            # Batch processing
            if not Path(args.batch).exists():
                print(f"❌ File batch không tồn tại: {args.batch}")
                return
            
            try:
                with open(args.batch, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = data
                    elif isinstance(data, dict) and 'texts' in data:
                        texts = data['texts']
                    else:
                        print("❌ File JSON phải chứa array hoặc object với key 'texts'")
                        return
                
                successful_files = await tts.batch_generate(
                    texts=texts,
                    output_dir=args.output_dir,
                    model=args.model,
                    voice=args.voice,
                    speed=args.speed,
                    instructions=instructions
                )
                
                print(f"\n🎉 Batch processing hoàn tất!")
                print(f"📁 Thư mục output: {args.output_dir}")
                return
                
            except json.JSONDecodeError:
                print("❌ File JSON không hợp lệ")
                return
            except Exception as e:
                print(f"❌ Lỗi xử lý batch: {str(e)}")
                return

        elif args.input:
            if not Path(args.input).exists():
                print(f"❌ File không tồn tại: {args.input}")
                return
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        elif args.text:
            text = args.text
        else:
            print("❌ Cần cung cấp văn bản qua tham số, --input, hoặc --batch")
            return

        if not text:
            print("❌ Văn bản trống")
            return

        # Preview mode
        if args.preview:
            print("🔊 Tạo preview...")
            audio_data = await tts.preview_speech(
                text=text,
                model=args.model,
                voice=args.voice,
                instructions=instructions
            )
            if audio_data:
                preview_file = f"preview_{datetime.now().strftime('%H%M%S')}.mp3"
                with open(preview_file, 'wb') as f:
                    f.write(audio_data)
                print(f"✅ Preview đã lưu: {preview_file}")
                
                # Optional: play preview if available
                try:
                    import subprocess
                    import platform
                    system = platform.system()
                    if system == "Darwin":  # macOS
                        subprocess.run(["afplay", preview_file])
                    elif system == "Windows":
                        subprocess.run(["start", preview_file], shell=True)
                    elif system == "Linux":
                        subprocess.run(["xdg-open", preview_file])
                except:
                    pass
            return

        # Generate speech
        output_file = args.output or tts.create_output_filename()
        
        # Add extension if not present
        if not Path(output_file).suffix:
            output_file = f"{output_file}.{args.response_format}"

        success = await tts.generate_speech(
            text=text,
            output_path=output_file,
            model=args.model,
            voice=args.voice,
            speed=args.speed,
            instructions=instructions,
            response_format=args.response_format
        )

        if success:
            print(f"🎵 File đã lưu: {output_file}")
        else:
            print("❌ Tạo TTS thất bại")
            sys.exit(1)

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
    asyncio.run(main())
