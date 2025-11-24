import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from razdel import tokenize as rz_tokenize
import pymorphy3
from rank_bm25 import BM25Okapi

MORPH = pymorphy3.MorphAnalyzer()

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r'[^\w\s\-ё]', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def lemma_tokens(s: str) -> List[str]:
    toks = [t.text for t in rz_tokenize(s)]
    lemmas = []
    for t in toks:
        if len(t) < 2:
            continue
        lemmas.append(MORPH.parse(t)[0].normal_form)
    return lemmas

def make_ngrams(tokens: List[str], n_from: int = 2, n_to: int = 3) -> List[str]:
    out = []
    for n in range(n_from, n_to + 1):
        out += ['_'.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return out

@dataclass
class Chunk:
    id: Any
    text: str
    meta: Optional[Dict[str, Any]] = None

class LexIndex:
    def __init__(self, chunks: List[Chunk], ngram_range: Tuple[int, int] = (2, 3)):
        self.chunks = chunks
        self.norm_texts = [normalize_text(c.text) for c in chunks]
        self.lemmas = [lemma_tokens(t) for t in self.norm_texts]
        self.ngram_range = ngram_range
        self.ngrams = [make_ngrams(lems, ngram_range[0], ngram_range[1]) for lems in self.lemmas]
        safe_ngrams = [ngr if len(ngr) > 0 else ["<empty>"] for ngr in self.ngrams]
        self.bm25_uni = BM25Okapi(self.lemmas)
        self.bm25_ng  = BM25Okapi(safe_ngrams)

    @staticmethod
    def _exact_bonus(text_norm: str, raw_query: str) -> float:
        qn = normalize_text(raw_query)
        return 1.0 if qn and qn in text_norm else 0.0

    @staticmethod
    def _proximity_bonus(doc_lemmas: List[str], q_lemmas: List[str], window: int = 6) -> float:
        if not q_lemmas:
            return 0.0
        pos: Dict[str, List[int]] = {}
        for i, w in enumerate(doc_lemmas):
            pos.setdefault(w, []).append(i)
        present = [q for q in q_lemmas if q in pos]
        if len(set(present)) < 2:
            return 0.0
        best = 10**9
        anchors = pos[present[0]]
        for a in anchors:
            span_min, span_max = a, a
            for q in present[1:]:
                j = min(pos[q], key=lambda x: abs(x - a))
                span_min, span_max = min(span_min, j), max(span_max, j)
            best = min(best, span_max - span_min + 1)
        return max(0.0, min(1.0, (window + 1) / (best + 1)))

    @staticmethod
    def _minmax(xs: List[float]) -> List[float]:
        if not xs:
            return xs
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-12:
            return [0.0 for _ in xs]
        return [(x - lo) / (hi - lo) for x in xs]

    def score_all(self,
                  query: str,
                  alpha: float = 0.6,
                  beta: float = 0.3,
                  delta: float = 0.07,
                  gamma: float = 0.03,
                  proximity_window: int = 6) -> List[float]:
        """
        Returns normalized final scores for ALL chunks (same order as self.chunks).
        """
        q_norm = normalize_text(query)
        q_lemmas = lemma_tokens(q_norm)
        q_ngrams = make_ngrams(q_lemmas, self.ngram_range[0], self.ngram_range[1]) or ["<empty>"]

        s_uni = list(self.bm25_uni.get_scores(q_lemmas))
        s_ng  = list(self.bm25_ng.get_scores(q_ngrams))
        s_uni = self._minmax(s_uni)
        s_ng  = self._minmax(s_ng)

        exact = [self._exact_bonus(txt, query) for txt in self.norm_texts]
        prox  = [self._proximity_bonus(doc, q_lemmas, window=proximity_window) for doc in self.lemmas]

        final = [alpha*s_uni[i] + beta*s_ng[i] + delta*exact[i] + gamma*prox[i]
                 for i in range(len(self.chunks))]
        return final