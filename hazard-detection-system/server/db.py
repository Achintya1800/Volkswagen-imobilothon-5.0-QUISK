# server/db.py
import sqlite3
import threading
from datetime import datetime

DB_FILE = "server/hazards.db"
LOCK = threading.Lock()

def init_db():
    with LOCK, sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS hazards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            hazard_type TEXT,
            confidence REAL,
            count INTEGER,
            first_seen TEXT,
            last_seen TEXT
        )
        """)
        conn.commit()

def list_hazards():
    with LOCK, sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, latitude, longitude, hazard_type, confidence, count, first_seen, last_seen FROM hazards")
        rows = cur.fetchall()
        return [dict(id=r[0], latitude=r[1], longitude=r[2], hazard_type=r[3],
                     confidence=r[4], count=r[5], first_seen=r[6], last_seen=r[7]) for r in rows]

def insert_hazard(lat, lon, hazard_type, confidence):
    now = datetime.utcnow().isoformat()
    with LOCK, sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO hazards (latitude, longitude, hazard_type, confidence, count, first_seen, last_seen)
                       VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (lat, lon, hazard_type, confidence, now, now))
        conn.commit()
        return cur.lastrowid

def update_hazard(hazard_id, new_lat, new_lon, new_confidence):
    now = datetime.utcnow().isoformat()
    with LOCK, sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        # increment count and average confidence (simple running average)
        cur.execute("SELECT confidence, count FROM hazards WHERE id=?", (hazard_id,))
        row = cur.fetchone()
        if row:
            old_conf, old_count = row
            combined_conf = (old_conf * old_count + new_confidence) / (old_count + 1)
            cur.execute("""UPDATE hazards SET latitude=?, longitude=?, confidence=?, count=count+1, last_seen=? WHERE id=?""",
                        (new_lat, new_lon, combined_conf, now, hazard_id))
            conn.commit()
