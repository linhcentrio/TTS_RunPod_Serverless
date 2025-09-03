#!/usr/bin/env python3
"""
Module TTS đơn giản sử dụng EdgeTTS
Chỉ có tính năng TTS cơ bản, không có SRT hay audio processing phức tạp
"""

import edge_tts
import asyncio
import argparse
import sys
import os
from typing import Optional, Dict, List
import io

# Cấu hình giọng đọc
VOICES = {
    # Tiếng Việt
    'vi-hoaimy': 'vi-VN-HoaiMyNeural',
    'vi-namminh': 'vi-VN-NamMinhNeural',
    
    # Tiếng Anh
    'en-aria': 'en-US-AriaNeural',
    'en-guy': 'en-US-GuyNeural',
    'en-jenny': 'en-US-JennyNeural',
    'en-davis': 'en-US-DavisNeural',
    
    # Tiếng Nhật
    'ja-nanami': 'ja-JP-NanamiNeural',
    'ja-keita': 'ja-JP-KeitaNeural',
    
    # Tiếng Trung
    'zh-xiaoxiao': 'zh-CN-XiaoxiaoNeural',
    'zh-yunxi': 'zh-CN-YunxiNeural',
    
    # Tiếng Hàn
    'ko-sun': 'ko-KR-SunHiNeural',
    'ko-injoon': 'ko-KR-InJoonNeural',
    
    # Tiếng Pháp
    'fr-denise': 'fr-FR-DeniseNeural',
    'fr-henri': 'fr-FR-HenriNeural',
    
    # Tiếng Tây Ban Nha
    'es-elia': 'es-ES-EliaNeural',
    'es-alvaro': 'es-ES-AlvaroNeural',
}

# Voice mặc định
DEFAULT_VOICE = 'vi-VN-HoaiMyNeural'

class EdgeTTS:
    """Lớp TTS đơn giản sử dụng EdgeTTS"""
    
    def __init__(self, voice: str = DEFAULT_VOICE):
        """
        Khởi tạo TTS engine
        
        Args:
            voice: Voice ID hoặc tên rút gọn
        """
        self.voice = self._resolve_voice(voice)
    
    def _resolve_voice(self, voice: str) -> str:
        """Chuyển đổi tên rút gọn thành voice ID"""
        # Nếu là voice ID đầy đủ
        if '-' in voice and voice.count('-') >= 2:
            return voice
        
        # Nếu là tên rút gọn
        voice_lower = voice.lower()
        if voice_lower in VOICES:
            return VOICES[voice_lower]
        
        # Tìm trong danh sách
        for short_name, full_name in VOICES.items():
            if voice_lower in short_name or voice_lower in full_name.lower():
                return full_name
        
        # Mặc định
        print(f"⚠️  Không tìm thấy giọng '{voice}', sử dụng mặc định: {DEFAULT_VOICE}")
        return DEFAULT_VOICE
    
    async def text_to_speech(self, text: str, output_file: Optional[str] = None) -> bytes:
        """
        Chuyển văn bản thành giọng nói
        
        Args:
            text: Văn bản cần đọc
            output_file: Đường dẫn file output (tùy chọn)
        
        Returns:
            Audio data dạng bytes
        """
        if not text.strip():
            raise ValueError("Văn bản không được để trống")
        
        try:
            # Tạo TTS communication
            communicate = edge_tts.Communicate(text, self.voice)
            
            # Thu thập audio data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if not audio_data:
                raise ValueError("Không thể tạo audio từ văn bản")
            
            # Lưu file nếu có output_file
            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                print(f"✅ Đã lưu audio: {output_file}")
            
            return audio_data
            
        except Exception as e:
            raise RuntimeError(f"Lỗi TTS: {e}")
    
    def set_voice(self, voice: str):
        """Thay đổi giọng đọc"""
        self.voice = self._resolve_voice(voice)
        print(f"🎙️  Đã chuyển sang giọng: {self.voice}")
    
    def get_voice(self) -> str:
        """Lấy giọng hiện tại"""
        return self.voice

# Hàm tiện ích
def list_voices():
    """Liệt kê tất cả giọng đọc có sẵn"""
    print("🎙️  Danh sách giọng đọc có sẵn:\n")
    
    current_lang = ""
    for short_name, full_name in VOICES.items():
        lang = full_name.split('-')[0]
        
        if lang != current_lang:
            current_lang = lang
            lang_names = {
                'vi': '🇻🇳 Tiếng Việt',
                'en': '🇺🇸 Tiếng Anh', 
                'ja': '🇯🇵 Tiếng Nhật',
                'zh': '🇨🇳 Tiếng Trung',
                'ko': '🇰🇷 Tiếng Hàn',
                'fr': '🇫🇷 Tiếng Pháp',
                'es': '🇪🇸 Tiếng Tây Ban Nha'
            }
            print(f"\n{lang_names.get(lang, lang.upper())}:")
        
        print(f"  {short_name:12} -> {full_name}")

async def quick_tts(text: str, voice: str = DEFAULT_VOICE, output_file: Optional[str] = None) -> bytes:
    """
    Hàm tiện ích để tạo TTS nhanh
    
    Args:
        text: Văn bản cần đọc
        voice: Giọng đọc
        output_file: File output (tùy chọn)
    
    Returns:
        Audio data
    """
    tts = EdgeTTS(voice)
    return await tts.text_to_speech(text, output_file)

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description="🎙️  EdgeTTS - Text to Speech đơn giản",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  %(prog)s --list                                      # Liệt kê giọng đọc
  %(prog)s "Xin chào" -o hello.mp3                     # TTS cơ bản
  %(prog)s "Hello world" -v en-aria -o hello_en.mp3    # TTS tiếng Anh
  %(prog)s -f input.txt -v vi-hoaimy -o output.mp3     # TTS từ file
        """
    )
    
    # Nhóm lệnh chính
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('text', nargs='?', help='Văn bản cần đọc')
    group.add_argument('-f', '--file', help='Đọc văn bản từ file')
    group.add_argument('--list', action='store_true', help='Liệt kê giọng đọc có sẵn')
    
    # Tùy chọn
    parser.add_argument('-v', '--voice', default=DEFAULT_VOICE,
                       help=f'Giọng đọc (mặc định: {DEFAULT_VOICE})')
    parser.add_argument('-o', '--output', help='File audio output')
    parser.add_argument('--quiet', action='store_true', help='Không in thông báo')
    
    args = parser.parse_args()
    
    # Xử lý lệnh list
    if args.list:
        list_voices()
        return
    
    # Lấy văn bản
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File không tồn tại: {args.file}")
            sys.exit(1)
        
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"❌ Lỗi đọc file: {e}")
            sys.exit(1)
    else:
        text = args.text
    
    if not text:
        print("❌ Văn bản không được để trống")
        sys.exit(1)
    
    # Thông báo
    if not args.quiet:
        print(f"🎙️  Giọng đọc: {args.voice}")
        print(f"📝 Văn bản: {text[:50]}{'...' if len(text) > 50 else ''}")
        if args.output:
            print(f"📁 Output: {args.output}")
        print("🚀 Đang tạo audio...")
    
    # Tạo TTS
    async def run_tts():
        try:
            audio_data = await quick_tts(text, args.voice, args.output)
            
            if not args.quiet:
                size_mb = len(audio_data) / (1024 * 1024)
                print(f"✅ Hoàn tất! Kích thước: {size_mb:.2f}MB")
                
                if not args.output:
                    print("💡 Sử dụng -o để lưu file audio")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return False
    
    # Chạy async
    success = asyncio.run(run_tts())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
