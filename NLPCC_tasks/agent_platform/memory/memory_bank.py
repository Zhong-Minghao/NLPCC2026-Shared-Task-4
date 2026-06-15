"""
MemoryBank: stores and retrieves sector/fund-level principles.

Keys are either fund_id strings (e.g. "512880.SH") or sector labels
(e.g. "金融_证券"). When retrieving for a list of fund_ids, both the
fund-specific key and its sector key are returned.
"""

import json
import os
from typing import Dict, List, Optional

from loguru import logger

from agent_platform.memory.memory_types import FUND_SECTOR_MAP, SectorPrinciple

MAX_PRINCIPLES_PER_KEY = 3


class MemoryBank:
    def __init__(self, persistence_path: Optional[str] = None):
        """
        Args:
            persistence_path: If provided, load on init and save after updates.
                              If None, memory lives only in this session.
        """
        self.persistence_path = persistence_path
        # key → list of SectorPrinciple (max MAX_PRINCIPLES_PER_KEY each)
        self._store: Dict[str, List[SectorPrinciple]] = {}

        if persistence_path and os.path.exists(persistence_path):
            self.load_from_disk(persistence_path)
            logger.info(
                f"MemoryBank: loaded {sum(len(v) for v in self._store.values())} "
                f"principles from {persistence_path}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_relevant_memories(
        self, fund_ids: List[str]
    ) -> Dict[str, List[SectorPrinciple]]:
        """Return principles for the given funds (both fund-level and sector-level)."""
        result: Dict[str, List[SectorPrinciple]] = {}
        seen_keys = set()

        for fund_id in fund_ids:
            for key in self._keys_for_fund(fund_id):
                if key in self._store and key not in seen_keys:
                    result[key] = self._store[key]
                    seen_keys.add(key)

        return result

    def get_all_principles(self) -> Dict[str, List[dict]]:
        """Return the full store as plain dicts (for passing to LLM)."""
        return {
            key: [p.to_dict() for p in principles]
            for key, principles in self._store.items()
        }

    def update_principles(self, updated: Dict[str, List[dict]]) -> None:
        """
        Merge LLM-generated principles into the store.
        Replaces the principle list for each key, capped at MAX_PRINCIPLES_PER_KEY.
        """
        for key, raw_list in updated.items():
            if not isinstance(raw_list, list):
                continue
            principles = []
            for item in raw_list[:MAX_PRINCIPLES_PER_KEY]:
                try:
                    principles.append(SectorPrinciple.from_dict(item))
                except Exception:
                    continue
            if principles:
                self._store[key] = principles
                logger.debug(f"MemoryBank: updated {len(principles)} principles for '{key}'")

    def save_to_disk(self, path: Optional[str] = None) -> None:
        target = path or self.persistence_path
        if not target:
            return
        data = {
            key: [p.to_dict() for p in principles]
            for key, principles in self._store.items()
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"MemoryBank: saved to {target}")

    def load_from_disk(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._store = {}
        for key, raw_list in data.items():
            self._store[key] = [SectorPrinciple.from_dict(item) for item in raw_list]

    def is_empty(self) -> bool:
        return len(self._store) == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keys_for_fund(fund_id: str) -> List[str]:
        """Return the fund_id itself plus its sector label (if mapped)."""
        keys = [fund_id]
        sector = FUND_SECTOR_MAP.get(fund_id)
        if sector:
            keys.append(sector)
        return keys
