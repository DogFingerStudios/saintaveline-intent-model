from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class WaypointMatch:
    waypoint: str
    score: float
    top_k: List[Tuple[str, float]]
    needs_clarification: bool
    clarification_choices: List[str]


class WaypointMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._waypoints: List[str] = []
        self._waypoint_embeddings: Optional[np.ndarray] = None

    def set_waypoints(self, waypoints: List[str]) -> None:
        self._waypoints = list(waypoints)

        if len(self._waypoints) == 0:
            self._waypoint_embeddings = None
            return

        emb = self._model.encode(self._waypoints, normalize_embeddings=True)
        self._waypoint_embeddings = np.asarray(emb, dtype=np.float32)

    def match(
        self,
        user_text: str,
        top_k: int = 5,
        min_score: float = 0.35,
        min_margin: float = 0.05
    ) -> WaypointMatch:
        if self._waypoint_embeddings is None or len(self._waypoints) == 0:
            return WaypointMatch(
                waypoint="",
                score=0.0,
                top_k=[],
                needs_clarification=True,
                clarification_choices=[]
            )

        query_emb = self._model.encode([user_text], normalize_embeddings=True)
        query_emb = np.asarray(query_emb, dtype=np.float32)[0]

        # Cosine similarity because embeddings are normalized -> dot product == cosine
        scores = np.dot(self._waypoint_embeddings, query_emb)

        idx_sorted = np.argsort(-scores)
        best_idx = int(idx_sorted[0])
        best_score = float(scores[best_idx])

        k = min(top_k, len(self._waypoints))
        top = []
        for i in range(k):
            idx = int(idx_sorted[i])
            top.append((self._waypoints[idx], float(scores[idx])))

        # Tie-break / ambiguity rules:
        # 1) If best score is too low -> clarify
        needs_clarification = False
        choices: List[str] = []

        if best_score < min_score:
            needs_clarification = True
            choices = [name for (name, _) in top[:min(3, len(top))]]
        else:
            # 2) If top1 is not far enough ahead of top2 -> clarify
            if len(top) >= 2:
                second_score = float(top[1][1])
                if (best_score - second_score) < min_margin:
                    needs_clarification = True
                    choices = [name for (name, _) in top[:min(3, len(top))]]

        return WaypointMatch(
            waypoint=self._waypoints[best_idx] if not needs_clarification else "",
            score=best_score,
            top_k=top,
            needs_clarification=needs_clarification,
            clarification_choices=choices
        )
