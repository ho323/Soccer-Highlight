import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
from path_utils import default_highlight_output_dir

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in re.findall(r"[A-Za-z0-9가-힣]+", text)]


def _index_text(desc: Dict) -> str:
    return f"{desc.get('event_type', '')}: {desc.get('description', '')}"


class BaseSearchEngine:
    engine_name = "base"

    def __init__(self, output_dir: str = default_highlight_output_dir()):
        self.output_dir = output_dir
        self.descriptions_dir = os.path.join(output_dir, "descriptions")
        os.makedirs(self.descriptions_dir, exist_ok=True)
        self.descriptions: Optional[List[Dict]] = None

    def build_index(self, descriptions: List[Dict]):
        self.descriptions = descriptions

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        raise NotImplementedError

    def _attach_meta(self, idx: int, score: float, rank: int) -> Dict:
        result = dict(self.descriptions[idx])
        result["search_score"] = float(score)
        result["search_rank"] = rank
        result["search_engine"] = self.engine_name
        return result

    def save_index(self):
        raise NotImplementedError

    def load_index(self) -> bool:
        raise NotImplementedError


class SemanticSearchEngine(BaseSearchEngine):
    engine_name = "semantic"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, output_dir: str = default_highlight_output_dir()):
        super().__init__(output_dir=output_dir)
        self.model = None
        self.embeddings = None
        self._model_unavailable = False

    def _load_model(self):
        if self.model is not None or self._model_unavailable:
            return
        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(self.embedding_model, device="cpu")
        except Exception as exc:
            logger.warning("Semantic model unavailable. Falling back to BM25 path: %s", exc)
            self._model_unavailable = True

    def build_index(self, descriptions: List[Dict]):
        super().build_index(descriptions)
        self._load_model()
        if self._model_unavailable:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            return
        texts = [_index_text(d) for d in descriptions]
        if not texts:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            return

        embs = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.embeddings = embs / norms

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        if self.embeddings is None or self.descriptions is None:
            return []

        self._load_model()
        if self._model_unavailable or self.model is None:
            return []
        q = self.model.encode([query], convert_to_numpy=True)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        sims = np.dot(self.embeddings, q.T).flatten()
        order = np.argsort(sims)[::-1][:top_k]
        out = []
        for idx in order:
            score = float(sims[idx])
            if score < min_score:
                continue
            out.append(self._attach_meta(idx=idx, score=score, rank=len(out) + 1))
        return out

    def save_index(self):
        if self.descriptions is None or self.embeddings is None:
            return

        desc_path = os.path.join(self.descriptions_dir, "search_index_semantic_descriptions.json")
        emb_path = os.path.join(self.descriptions_dir, "search_index_semantic_embeddings.npy")
        save_data = []
        for d in self.descriptions:
            row = dict(d)
            if os.path.isabs(row.get("clip_path", "")):
                row["clip_path"] = os.path.relpath(row["clip_path"], self.output_dir)
            save_data.append(row)
        with open(desc_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        np.save(emb_path, self.embeddings)

    def load_index(self) -> bool:
        desc_path = os.path.join(self.descriptions_dir, "search_index_semantic_descriptions.json")
        emb_path = os.path.join(self.descriptions_dir, "search_index_semantic_embeddings.npy")
        if not os.path.exists(desc_path) or not os.path.exists(emb_path):
            return False

        with open(desc_path, "r", encoding="utf-8") as f:
            self.descriptions = json.load(f)

        for d in self.descriptions:
            if not os.path.isabs(d.get("clip_path", "")):
                d["clip_path"] = os.path.join(self.output_dir, d["clip_path"])
        self.embeddings = np.load(emb_path)
        return True


class BM25SearchEngine(BaseSearchEngine):
    engine_name = "bm25"

    def __init__(self, output_dir: str = default_highlight_output_dir()):
        super().__init__(output_dir=output_dir)
        self.doc_tokens: List[List[str]] = []
        self.doc_freq = defaultdict(int)
        self.doc_len: List[int] = []
        self.avg_doc_len = 0.0
        self.k1 = 1.2
        self.b = 0.75

    def build_index(self, descriptions: List[Dict]):
        super().build_index(descriptions)
        self.doc_tokens = []
        self.doc_freq = defaultdict(int)
        self.doc_len = []

        for desc in descriptions:
            tokens = _tokenize(_index_text(desc))
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            unique = set(tokens)
            for t in unique:
                self.doc_freq[t] += 1
        self.avg_doc_len = sum(self.doc_len) / max(1, len(self.doc_len))

    def _idf(self, token: str) -> float:
        n_docs = max(1, len(self.doc_tokens))
        df = self.doc_freq.get(token, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _score_doc(self, q_tokens: List[str], idx: int) -> float:
        if idx >= len(self.doc_tokens):
            return 0.0
        doc = self.doc_tokens[idx]
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for token in q_tokens:
            f = tf.get(token, 0)
            if f == 0:
                continue
            idf = self._idf(token)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_doc_len, 1e-8))
            score += idf * (f * (self.k1 + 1) / max(denom, 1e-8))
        return score

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        if self.descriptions is None:
            return []
        q_tokens = _tokenize(query)
        scores = [self._score_doc(q_tokens, i) for i in range(len(self.descriptions))]
        order = np.argsort(scores)[::-1][:top_k]
        out = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:
                continue
            if score < min_score:
                continue
            out.append(self._attach_meta(idx=idx, score=score, rank=len(out) + 1))
        return out

    def save_index(self):
        if self.descriptions is None:
            return
        path = os.path.join(self.descriptions_dir, "search_index_bm25_descriptions.json")
        save_data = []
        for d in self.descriptions:
            row = dict(d)
            if os.path.isabs(row.get("clip_path", "")):
                row["clip_path"] = os.path.relpath(row["clip_path"], self.output_dir)
            save_data.append(row)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def load_index(self) -> bool:
        path = os.path.join(self.descriptions_dir, "search_index_bm25_descriptions.json")
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            self.descriptions = json.load(f)
        for d in self.descriptions:
            if not os.path.isabs(d.get("clip_path", "")):
                d["clip_path"] = os.path.join(self.output_dir, d["clip_path"])
        self.build_index(self.descriptions)
        return True


class HybridSearchEngine(BaseSearchEngine):
    engine_name = "hybrid"

    def __init__(self, output_dir: str = default_highlight_output_dir(), alpha: float = 0.7):
        super().__init__(output_dir=output_dir)
        self.alpha = float(alpha)
        self.semantic = SemanticSearchEngine(output_dir=output_dir)
        self.bm25 = BM25SearchEngine(output_dir=output_dir)

    def build_index(self, descriptions: List[Dict]):
        super().build_index(descriptions)
        self.semantic.build_index(descriptions)
        self.bm25.build_index(descriptions)

    def _normalize(self, rows: List[Dict]) -> Dict[int, float]:
        if not rows:
            return {}
        vals = [float(r["search_score"]) for r in rows]
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-8:
            return {int(r["clip_index"]): 1.0 for r in rows}
        return {int(r["clip_index"]): (float(r["search_score"]) - lo) / (hi - lo) for r in rows}

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        if self.descriptions is None:
            return []
        semantic_rows = self.semantic.search(query, top_k=max(top_k * 3, 20), min_score=0.0)
        bm25_rows = self.bm25.search(query, top_k=max(top_k * 3, 20), min_score=0.0)

        sem = self._normalize(semantic_rows)
        b25 = self._normalize(bm25_rows)
        all_ids = set(sem) | set(b25)
        scored = []
        for idx in all_ids:
            score = self.alpha * sem.get(idx, 0.0) + (1.0 - self.alpha) * b25.get(idx, 0.0)
            if score >= min_score:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        out = []
        for idx, score in scored[:top_k]:
            out.append(self._attach_meta(idx=idx, score=score, rank=len(out) + 1))
        return out

    def save_index(self):
        self.semantic.save_index()
        self.bm25.save_index()
        meta_path = os.path.join(self.descriptions_dir, "search_index_hybrid_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"alpha": self.alpha}, f, ensure_ascii=False, indent=2)

    def load_index(self) -> bool:
        ok1 = self.semantic.load_index()
        ok2 = self.bm25.load_index()
        if not ok1 and not ok2:
            return False
        # Prefer semantic descriptions when available.
        self.descriptions = self.semantic.descriptions if ok1 else self.bm25.descriptions
        self.semantic.descriptions = self.descriptions
        self.bm25.descriptions = self.descriptions
        if self.descriptions is not None:
            self.bm25.build_index(self.descriptions)
        return True


class SceneSearchEngine:
    def __init__(
        self,
        output_dir: str = default_highlight_output_dir(),
        engine_type: str = "semantic",
        hybrid_alpha: float = 0.7,
    ):
        self.output_dir = output_dir
        self.engine_type = engine_type
        self.hybrid_alpha = hybrid_alpha
        self.engine = self._create_engine(engine_type, hybrid_alpha)

    @property
    def embeddings(self):
        # Keep compatibility with previous UI checks.
        return getattr(self.engine, "embeddings", None)

    def _create_engine(self, engine_type: str, hybrid_alpha: float):
        if engine_type == "bm25":
            return BM25SearchEngine(output_dir=self.output_dir)
        if engine_type == "hybrid":
            return HybridSearchEngine(output_dir=self.output_dir, alpha=hybrid_alpha)
        return SemanticSearchEngine(output_dir=self.output_dir)

    def set_engine(self, engine_type: str, hybrid_alpha: Optional[float] = None):
        alpha_changed = False
        if hybrid_alpha is not None and hybrid_alpha != self.hybrid_alpha:
            self.hybrid_alpha = hybrid_alpha
            alpha_changed = True
        if engine_type == self.engine_type and not (engine_type == "hybrid" and alpha_changed):
            return
        self.engine_type = engine_type
        self.engine = self._create_engine(engine_type, self.hybrid_alpha)

    def _fallback_bm25_engine(self):
        return BM25SearchEngine(output_dir=self.output_dir)

    def build_index(self, descriptions: List[Dict]):
        try:
            self.engine.build_index(descriptions)
        except Exception as exc:
            logger.warning("Primary search index build failed (%s). Falling back to BM25.", exc)
            self.engine_type = "bm25"
            self.engine = self._fallback_bm25_engine()
            self.engine.build_index(descriptions)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        try:
            rows = self.engine.search(query=query, top_k=top_k, min_score=min_score)
        except Exception as exc:
            logger.warning("Primary search failed (%s). Falling back to BM25.", exc)
            self.engine_type = "bm25"
            self.engine = self._fallback_bm25_engine()
            if self.engine.load_index():
                rows = self.engine.search(query=query, top_k=top_k, min_score=min_score)
            else:
                rows = []

        if not rows and self.engine_type in ("semantic", "hybrid"):
            fallback = self._fallback_bm25_engine()
            if fallback.load_index():
                rows = fallback.search(query=query, top_k=top_k, min_score=min_score)
        return rows

    def save_index(self):
        self.engine.save_index()

    def load_index(self) -> bool:
        ok = False
        try:
            ok = self.engine.load_index()
        except Exception as exc:
            logger.warning("Primary search index load failed (%s).", exc)
            ok = False

        if ok:
            return True
        if self.engine_type == "bm25":
            return False

        logger.warning("Falling back to BM25 index load.")
        self.engine_type = "bm25"
        self.engine = self._fallback_bm25_engine()
        return self.engine.load_index()

