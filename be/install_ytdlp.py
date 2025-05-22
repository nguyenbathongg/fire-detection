import subprocess

def install_ytdlp_from_github():
    """
    Cài đặt yt-dlp phiên bản mới nhất từ GitHub
    """
    try:
        print("🔄 Đang gỡ bỏ yt-dlp cũ (nếu có)...")
        subprocess.run(['pip', 'uninstall', 'yt-dlp', '-y'], capture_output=True)

        print("🔄 Đang cài đặt yt-dlp mới nhất từ GitHub...")
        result = subprocess.run([
            'pip', 'install',
            'git+https://github.com/yt-dlp/yt-dlp.git'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Cài đặt yt-dlp mới thành công!")
            return True
        else:
            print(f"❌ Cài đặt thất bại: {result.stderr}")
    except Exception as e:
        print(f"❌ Lỗi khi cài đặt yt-dlp: {e}")
    return False

if __name__ == "__main__":
    install_ytdlp_from_github()
