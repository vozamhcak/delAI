# faq_search.py
from typing import List, Dict, Optional
import ast

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from config import (
    FAQ_CSV_PATH,
    EMBEDDING_MODEL_NAME,
    W_EMBED,
    W_LEX,
    HYBRID_THRESHOLD,
    TOP_K,
)
from lex_index import LexIndex, Chunk


def cosine_to_unit(scores: torch.Tensor) -> torch.Tensor:
    """
    Перевод косинусного сходства из [-1, 1] в [0, 1].
    """
    return (scores + 1.0) / 2.0


def load_faq_data(csv_path: str) -> pd.DataFrame:
    """
    Ожидается CSV с колонками:
    - question
    - answer
    - embedding (строка вида "[0.1, 0.2, ...]")
    """
    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
    )
    return df


def build_lex_index(df: pd.DataFrame) -> LexIndex:
    """
    Строим лексический индекс из списка вопросов.
    """
    chunks = [
        Chunk(
            id=i,
            text=df.iloc[i]["question"],
            meta={"answer": df.iloc[i]["answer"]},
        )
        for i in range(len(df))
    ]
    return LexIndex(chunks)


class HybridFAQ:
    """
    Класс, инкапсулирующий:
    - загрузку FAQ,
    - модель эмбеддингов,
    - лексический индекс,
    - гибридный поиск.
    """

    def __init__(
        self,
        csv_path: str,
        model_name: str,
        w_embed: float,
        w_lex: float,
        threshold: float,
    ):
        self.csv_path = csv_path
        self.w_embed = w_embed
        self.w_lex = w_lex
        self.threshold = threshold

        # Устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем данные FAQ
        self.df = load_faq_data(self.csv_path)

        # Загружаем модель эмбеддингов
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

        # Готовим тензор всех эмбеддингов
        self.embeddings = torch.tensor(
            np.stack(self.df["embedding"].values),
            device=self.device,
        )

        # Лексический индекс
        self.lex_index = build_lex_index(self.df)

    # ------------------------------

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Возвращает top_k результатов с полным набором скорингов.
        НИКАКИХ порогов здесь не применяем — только сортировка.
        """
        if not query.strip():
            return []

        # Эмбеддинг запроса
        query_emb = self.model.encode([query], convert_to_tensor=True)
        query_emb = query_emb.to(self.device)

        # Семантический скор
        cos_scores = util.cos_sim(query_emb, self.embeddings)[0]
        embed_scores = cosine_to_unit(cos_scores)

        # Лексический скор от LexIndex
        lex_scores_list = self.lex_index.score_all(query)
        lex_scores = torch.tensor(
            lex_scores_list,
            dtype=torch.float32,
            device=self.device,
        )

        # Гибридный скор
        hybrid_scores = self.w_embed * embed_scores + self.w_lex * lex_scores

        # Индексы top_k
        top_k = min(top_k, len(self.df))
        top_idx = torch.argsort(hybrid_scores, descending=True)[:top_k].tolist()

        results: List[Dict] = []
        for idx in top_idx:
            q = self.df.iloc[idx]["question"]
            a = self.df.iloc[idx]["answer"]
            results.append(
                {
                    "query": query,
                    "question": q,
                    "answer": a,
                    "lex_score": float(lex_scores[idx]),
                    "embed_score": float(embed_scores[idx]),
                    "hybrid_score": float(hybrid_scores[idx]),
                }
            )

        return results

    # ------------------------------

    def search_best(self, query: str) -> Optional[Dict]:
        """
        Возвращает лучший результат (top1) или None,
        если гибридный скор ниже порога.
        """
        results = self.search(query, top_k=1)
        if not results:
            return None

        best = results[0]
        if best["hybrid_score"] >= self.threshold:
            return best

        return None


# Глобальный экземпляр, который будет использовать бот
faq_engine = HybridFAQ(
    csv_path=FAQ_CSV_PATH,
    model_name=EMBEDDING_MODEL_NAME,
    w_embed=W_EMBED,
    w_lex=W_LEX,
    threshold=HYBRID_THRESHOLD,
)


def search_faq_best(query: str) -> Optional[Dict]:
    """
    Упрощённая функция для использования в боте.
    Возвращает dict:
        {
          "question": ...,
          "answer": ...,
          "lex_score": ...,
          "embed_score": ...,
          "hybrid_score": ...
        }
    или None, если нет подходящего ответа.
    """
    return faq_engine.search_best(query)
