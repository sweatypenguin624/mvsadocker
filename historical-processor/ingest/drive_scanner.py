import subprocess
import json

class DriveScanner:
    def __init__(self, root_folder_id=None, remote=None, rclone_path="tools/rclone-v1.75.0-linux-amd64/rclone"):
        self.rclone_path = rclone_path
        if remote:
            self.remote = remote if remote.endswith(':') else remote + ':'
        elif root_folder_id:
            self.remote = f"Gdrive-yogesh,root_folder_id={root_folder_id}:"
        else:
            self.remote = "Gdrive-yogesh,root_folder_id=1fHpvoRMCz0xFK-pWZm_CyW1tTGAQkWmB:"

    def scan(self, manifest):
        print(f"Scanning {self.remote} with rclone...")
        cmd = [self.rclone_path, "--config", "config/rclone.conf", "lsjson", "-R", "--files-only", "--fast-list", "--tpslimit", "4", "--tpslimit-burst", "4", self.remote]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error scanning Drive:")
            print(result.stderr)
            return

        try:
            files = json.loads(result.stdout)
        except Exception as e:
            print(f"Failed to parse rclone output: {e}")
            return
            
        print(f"Discovered {len(files)} total files.")
        
        added = 0
        for f in files:
            path = f.get('Path', '')
            name = f.get('Name', '')
            file_id = f.get('ID', path)
            
            if not name.endswith('.dav'):
                continue
            if "backup" in path.lower():
                continue
                
            parts = path.split('/')
            if len(parts) >= 2:
                date_folder = parts[-2]
                date = date_folder
            else:
                date = "unknown"
                
            start_time = name.replace('.dav', '')
            
            manifest.add_or_update(
                file_id=file_id,
                gdrive_path=path,
                filename=name,
                date=date,
                start_time=start_time
            )
            added += 1
                
        print(f"Added/Updated {added} .dav files in manifest.")
