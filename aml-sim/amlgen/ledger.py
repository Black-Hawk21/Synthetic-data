"""Append-only stores for transactions and episodes."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

TXN_COLUMNS = [
    "txn_id", "timestamp", "sender", "receiver", "amount", "channel",
    "sender_country", "receiver_country", "cross_border",
    "episode_id", "pattern", "is_laundering",
]


class Ledger:
    """Collects transaction chunks as columnar dicts, concatenated once at the end."""

    def __init__(self) -> None:
        self._chunks: List[Dict[str, np.ndarray]] = []
        self._n = 0

    def add_bulk(self, sender, receiver, timestamp, amount, channel,
                 sender_country, receiver_country, episode_id=None,
                 pattern=None, is_laundering=0) -> None:
        n = len(sender)
        if n == 0:
            return
        def col(value, dtype=object):
            if np.isscalar(value) or value is None:
                return np.full(n, value, dtype=dtype)
            return np.asarray(value)
        self._chunks.append({
            "timestamp": np.asarray(timestamp, dtype=np.int64),
            "sender": np.asarray(sender, dtype=np.int32),
            "receiver": np.asarray(receiver, dtype=np.int32),
            "amount": np.asarray(amount, dtype=float),
            "channel": col(channel),
            "sender_country": col(sender_country),
            "receiver_country": col(receiver_country),
            "episode_id": col(episode_id),
            "pattern": col(pattern),
            "is_laundering": col(is_laundering, dtype=np.int8).astype(np.int8),
        })
        self._n += n

    def __len__(self) -> int:
        return self._n

    def to_frame(self, accounts: pd.DataFrame) -> pd.DataFrame:
        """Concatenate chunks column-wise and resolve account indices to IDs."""
        if not self._chunks:
            raise RuntimeError("ledger is empty")
        cols = {k: np.concatenate([c[k] for c in self._chunks]) for k in self._chunks[0]}
        order = np.argsort(cols["timestamp"], kind="stable")
        ids = accounts["account_id"].to_numpy()
        df = pd.DataFrame({
            "txn_id": [f"T{i:09d}" for i in range(order.size)],
            "timestamp": pd.to_datetime(cols["timestamp"][order], unit="s"),
            "sender": pd.Categorical(ids[cols["sender"][order]]),
            "receiver": pd.Categorical(ids[cols["receiver"][order]]),
            "amount": np.round(cols["amount"][order], 2),
            "channel": pd.Categorical(cols["channel"][order]),
            "sender_country": pd.Categorical(cols["sender_country"][order]),
            "receiver_country": pd.Categorical(cols["receiver_country"][order]),
            "episode_id": cols["episode_id"][order],
            "pattern": pd.Categorical(cols["pattern"][order]),
            "is_laundering": cols["is_laundering"][order],
        })
        df["cross_border"] = (df["sender_country"].astype(str)
                              != df["receiver_country"].astype(str)).astype(np.int8)
        df["episode_id"] = pd.Series(df["episode_id"]).fillna("").astype(str).replace("None", "")
        return df[TXN_COLUMNS]


class EpisodeStore:
    """Ground truth at the episode level: the abstraction that makes labels useful."""

    def __init__(self) -> None:
        self.episodes: List[dict] = []
        self.members: List[dict] = []
        self._counter = 0

    def new_id(self, prefix: str = "E") -> str:
        self._counter += 1
        return f"{prefix}{self._counter:06d}"

    def record(self, episode_id: str, pattern: str, family: str, is_laundering: int,
               start_ts: int, end_ts: int, total_amount: float, n_txns: int,
               difficulty: float, members: Dict[str, Sequence[int]]) -> None:
        accounts = sorted({int(a) for group in members.values() for a in group})
        self.episodes.append({
            "episode_id": episode_id,
            "pattern": pattern,
            "family": family,
            "is_laundering": int(is_laundering),
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "duration_hours": round((end_ts - start_ts) / 3600.0, 3),
            "total_amount": round(float(total_amount), 2),
            "n_transactions": int(n_txns),
            "n_accounts": len(accounts),
            "difficulty": round(float(difficulty), 3),
        })
        for role, group in members.items():
            for acc in group:
                self.members.append({"episode_id": episode_id, "account_idx": int(acc), "role": role})

    def to_frames(self, accounts: pd.DataFrame):
        ep = pd.DataFrame(self.episodes)
        mem = pd.DataFrame(self.members)
        if not ep.empty:
            ep["start_time"] = pd.to_datetime(ep["start_ts"], unit="s")
            ep["end_time"] = pd.to_datetime(ep["end_ts"], unit="s")
            ep = ep.drop(columns=["start_ts", "end_ts"])
        if not mem.empty:
            ids = accounts["account_id"].to_numpy()
            mem["account_id"] = ids[mem["account_idx"].to_numpy()]
            mem = mem.drop(columns=["account_idx"])
            mem = mem.merge(ep[["episode_id", "pattern", "family", "is_laundering"]],
                            on="episode_id", how="left")
        return ep, mem
