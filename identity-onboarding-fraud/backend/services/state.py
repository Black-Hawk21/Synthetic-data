"""Process-local application state: the "current" synthetic dataset the
Red/Blue Team pages are working with. Backed up to SQLite (section 18) so
history survives, but kept in-memory as a DataFrame for speed -- this is a
research/demo app, not a production multi-tenant service.
"""
from __future__ import annotations

import threading

import pandas as pd


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.dataset: pd.DataFrame | None = None
        self.last_attack_batches: dict[str, pd.DataFrame] = {}
        self.last_attack_meta: list[dict] = []
        self.feedback_history: list[dict] = []

    def set_dataset(self, df: pd.DataFrame) -> None:
        with self._lock:
            self.dataset = df

    def get_dataset(self) -> pd.DataFrame | None:
        with self._lock:
            return self.dataset

    def append_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        with self._lock:
            if self.dataset is None:
                self.dataset = df
            else:
                self.dataset = pd.concat([self.dataset, df], ignore_index=True)
            return self.dataset


state = AppState()
