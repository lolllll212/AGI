"""Conversations kept on disk."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Callable


class SessionStore:
    """Conversations on disk (SQLite), so they survive a restart and can outnumber what fits in RAM."""

    def __init__(self, path: str, ttl_s: float | None = 30 * 86400, clock: Callable[[], float] = time.time):
        self.path, self.ttl_s, self.clock = path, ttl_s, clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("create table if not exists sessions "
                         "(id text primary key, updated real not null, messages text not null)")
        self._db.commit()
        self.prune()

    def load(self, session: str) -> list[dict] | None:
        with self._lock:
            row = self._db.execute("select updated, messages from sessions where id = ?", (session,)).fetchone()
        if row is None or (self.ttl_s is not None and self.clock() - row[0] > self.ttl_s):
            return None
        try:
            return json.loads(row[1])
        except ValueError:
            return None

    def save(self, session: str, messages: list[dict]) -> None:
        with self._lock:
            self._db.execute("insert or replace into sessions values (?, ?, ?)",
                             (session, self.clock(), json.dumps(messages)))
            self._db.commit()

    def delete(self, session: str) -> bool:
        with self._lock:
            found = self._db.execute("delete from sessions where id = ?", (session,)).rowcount > 0
            self._db.commit()
        return found

    def prune(self) -> int:
        """Delete conversations idle for longer than the time-to-live."""
        if self.ttl_s is None:
            return 0
        with self._lock:
            removed = self._db.execute("delete from sessions where updated < ?",
                                       (self.clock() - self.ttl_s,)).rowcount
            self._db.commit()
        return removed

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("select count(*) from sessions").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()
