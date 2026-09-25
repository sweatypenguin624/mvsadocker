import sqlite3
import os
import datetime
from typing import List, Dict, Any

class IngestManifest:
    def __init__(self, db_path="historical-processor/data/manifest.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    file_id TEXT PRIMARY KEY,
                    gdrive_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    date TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    local_path TEXT,
                    status TEXT DEFAULT 'DISCOVERED',
                    downloaded_at TEXT,
                    processed_at TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            ''')
            cols = {row[1] for row in cursor.execute("PRAGMA table_info(videos)")}
            if "camera" not in cols:
                cursor.execute("ALTER TABLE videos ADD COLUMN camera TEXT")
            conn.commit()

    def add_or_update(self, file_id: str, gdrive_path: str, filename: str, date: str, start_time: str,
                      camera: str = None):
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO videos (file_id, gdrive_path, filename, date, start_time, camera)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    gdrive_path = excluded.gdrive_path,
                    filename = excluded.filename,
                    camera = COALESCE(excluded.camera, videos.camera)
            ''', (file_id, gdrive_path, filename, date, start_time, camera))
            conn.commit()


    def get_pending_batch(self, batch_size: int = 5, max_retries: int = 3) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM videos
                WHERE status = 'DISCOVERED' OR (status = 'FAILED' AND retry_count < ?)
                ORDER BY (status = 'FAILED') ASC, date ASC, start_time ASC
                LIMIT ?
            ''', (max_retries, batch_size))
            return [dict(row) for row in cursor.fetchall()]

    def count_pending(self, max_retries: int = 3) -> int:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM videos WHERE status = 'DISCOVERED' "
                "OR (status = 'FAILED' AND retry_count < ?)", (max_retries,)).fetchone()[0]

    def get_all(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM videos ORDER BY camera, date, start_time")]

    def reset_retries(self) -> None:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("UPDATE videos SET retry_count = 0, status = 'DISCOVERED' WHERE status = 'FAILED'")
            conn.commit()


    def update_status(self, file_id: str, status: str, local_path: str = None, error: str = None):
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            now = datetime.datetime.now().isoformat()

            updates = ["status = ?"]
            params = [status]

            if local_path is not None:
                updates.append("local_path = ?")
                params.append(local_path)

            if error is not None:
                updates.append("error = ?")
                params.append(error)
                if status == 'FAILED':
                    updates.append("retry_count = retry_count + 1")

            if status == 'DOWNLOADED':
                updates.append("downloaded_at = ?")
                params.append(now)
            elif status == 'COMPLETED':
                updates.append("processed_at = ?")
                params.append(now)

            query = f"UPDATE videos SET {', '.join(updates)} WHERE file_id = ?"
            params.append(file_id)

            cursor.execute(query, params)
            conn.commit()


    def get_stats(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, COUNT(*) FROM videos GROUP BY status")
            stats = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*) FROM videos")
            total = cursor.fetchone()[0]

            stats['TOTAL'] = total
            return stats

    def get_by_status(self, status: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE status = ?", (status,))
            return [dict(row) for row in cursor.fetchall()]
