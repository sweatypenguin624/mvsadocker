import subprocess
import os

class Downloader:
    def __init__(self, root_folder_id=None, remote=None, rclone_path="tools/rclone-v1.75.0-linux-amd64/rclone"):
        self.rclone_path = rclone_path
        if remote:
            self.remote = remote.rstrip(':')
        elif root_folder_id:
            self.remote = f"Gdrive-yogesh,root_folder_id={root_folder_id}"
        else:
            self.remote = "Gdrive-yogesh,root_folder_id=1fHpvoRMCz0xFK-pWZm_CyW1tTGAQkWmB"
        
    def download_file(self, gdrive_path: str, local_path: str, timeout: float = None) -> bool:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"File already exists locally, skipping download: {local_path}")
            return True
            
        tmp_path = local_path + ".tmp"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"Downloading {gdrive_path} to {local_path} ...")
        
        cmd = [
            self.rclone_path, "--config", "config/rclone.conf",
            "copyto", f"{self.remote}:{gdrive_path}", tmp_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Download timed out after {timeout}s: {gdrive_path}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

        if result.returncode != 0:
            print(f"Failed to download {gdrive_path}: {result.stderr}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
            
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            print(f"File downloaded but is missing or empty: {gdrive_path}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
            
        os.rename(tmp_path, local_path)
        print(f"Successfully downloaded {local_path}")
        return True
