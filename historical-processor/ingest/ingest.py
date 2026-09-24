import argparse
import sys
import os
from pathlib import Path

INGEST_DIR = Path(__file__).resolve().parent
HIST_DIR = INGEST_DIR.parent
for p in [str(INGEST_DIR), str(HIST_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ingest.manifest import IngestManifest
    from ingest.drive_scanner import DriveScanner
    from ingest.downloader import Downloader
except ImportError:
    from manifest import IngestManifest
    from drive_scanner import DriveScanner
    from downloader import Downloader

def print_status(manifest):
    stats = manifest.get_stats()
    print("\n--- Historical Ingestion Status ---")
    print(f"Database:   {manifest.db_path}")
    print(f"Total:      {stats.get('TOTAL', 0)}")
    print(f"Completed:  {stats.get('COMPLETED', 0)}")
    print(f"Downloaded: {stats.get('DOWNLOADED', 0)}")
    print(f"Discovered: {stats.get('DISCOVERED', 0)}")
    print(f"Failed:     {stats.get('FAILED', 0)}")
    print(f"Pending:    {stats.get('DISCOVERED', 0) + stats.get('FAILED', 0)}")
    
def main():
    parser = argparse.ArgumentParser(description="Google Drive Ingestion for Historical Processor")
    parser.add_argument('--scan', action='store_true', help="Scan Google Drive and update manifest")
    parser.add_argument('--next-batch', action='store_true', help="Download the next batch of files")
    parser.add_argument('--status', action='store_true', help="Show current ingestion status")
    parser.add_argument('--batch-size', type=int, default=5, help="Number of files to download per batch")
    parser.add_argument('--folder-id', type=str, default=None, help="Google Drive folder ID")
    parser.add_argument('--db', type=str, default="historical-processor/data/manifest.db", help="Path to manifest SQLite DB")
    parser.add_argument('--reset-failed', action='store_true', help="Reset failed files to discovered for retry")
    args = parser.parse_args()
    
    manifest = IngestManifest(db_path=args.db)
    
    if args.reset_failed:
        for item in manifest.get_by_status("FAILED"):
            manifest.update_status(item['file_id'], "DISCOVERED")
        print("Reset all FAILED files to DISCOVERED.")

    if args.status:
        print_status(manifest)
        
    if args.scan:
        scanner = DriveScanner(root_folder_id=args.folder_id)
        scanner.scan(manifest)
        print_status(manifest)
        
    if args.next_batch:
        batch = manifest.get_pending_batch(batch_size=args.batch_size)
        if not batch:
            print("No pending files to download.")
            return
            
        print("Current batch:")
        for item in batch:
            print(f"{item['date']}/{item['filename']}")
            
        downloader = Downloader(root_folder_id=args.folder_id)
        base_dir = "historical-processor/data/videos/historical"
        
        for item in batch:
            file_id = item['file_id']
            gdrive_path = item['gdrive_path']
            date = item['date']
            filename = item['filename']
            
            manifest.update_status(file_id=file_id, status="DOWNLOADING")
            
            local_dir = os.path.join(base_dir, date)
            local_path = os.path.join(local_dir, filename)
            
            success = downloader.download_file(gdrive_path, local_path)
            
            if success:
                manifest.update_status(file_id=file_id, status="DOWNLOADED", local_path=local_path)
            else:
                manifest.update_status(file_id=file_id, status="FAILED", error="Download failed or file empty")
                
        print("\nBatch download complete.")
        print_status(manifest)

if __name__ == "__main__":
    main()
