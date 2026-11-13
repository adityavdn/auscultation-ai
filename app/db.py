"""Doctor inbox storage (stdlib sqlite). On the free host this resets on restart: fine for a prototype."""
import json
import sqlite3
import time
from contextlib import closing


class CaseStore:
    def __init__(self, path):
        self.path = str(path)
        with closing(self._conn()) as c, c:
            c.execute("""CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, patient_name TEXT, urgency TEXT,
                summary TEXT, analysis TEXT, transcript TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.path, check_same_thread=False)

    def add(self, patient_name, urgency, summary, analysis, transcript):
        with closing(self._conn()) as c, c:
            cur = c.execute("INSERT INTO cases (created, patient_name, urgency, summary, analysis, transcript) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (time.time(), patient_name, urgency, json.dumps(summary), json.dumps(analysis),
                             json.dumps(transcript)))
            return self.get(cur.lastrowid, c)

    def get(self, case_id, conn=None):
        c = conn or self._conn()
        try:
            row = c.execute("SELECT id, created, patient_name, urgency, summary, analysis, transcript "
                            "FROM cases WHERE id = ?", (case_id,)).fetchone()
        finally:
            if conn is None:
                c.close()
        return self._row(row) if row else None

    def list(self, limit=100):
        with closing(self._conn()) as c:
            rows = c.execute("SELECT id, created, patient_name, urgency, summary, analysis, transcript "
                             "FROM cases ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(r):
        return {"id": r[0], "created": r[1], "patient_name": r[2], "urgency": r[3],
                "summary": json.loads(r[4]), "analysis": json.loads(r[5]), "transcript": json.loads(r[6])}
