
from typing import List, Dict
import ast

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from lex_index import LexIndex, Chunk
from config import FAQ_CSV_PATH, EMBEDDING_MODEL_NAME, W_EMBED, W_LEX


def cosine_to_unit(scores: torch.Tensor) -> torch.Tensor:
    return (scores + 1.0) / 2.0


def _load_faq_data(csv_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
    )
    return df


def _build_lex_index(df: pd.DataFrame) -> LexIndex:
    
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
    

    def __init__(self, csv_path: str, model_name: str, w_embed: float, w_lex: float):
        self.csv_path = csv_path
        self.w_embed = w_embed
        self.w_lex = w_lex

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df = _load_faq_data(self.csv_path)

        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

        self.embeddings = torch.tensor(
            np.stack(self.df["embedding"].values),
            device=self.device,
        )

        self.lex_index = _build_lex_index(self.df)


    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        
        query = (query or "").strip()
        if not query:
            return []

        query_emb = self.model.encode([query], convert_to_tensor=True).to(self.device)

        cos_scores = util.cos_sim(query_emb, self.embeddings)[0]
        embed_scores = cosine_to_unit(cos_scores)

        lex_scores_list = self.lex_index.score_all(query)
        lex_scores = torch.tensor(lex_scores_list, dtype=torch.float32, device=self.device)

        hybrid_scores = self.w_embed * embed_scores + self.w_lex * lex_scores

        top_k = min(top_k, len(self.df))
        top_idx = torch.argsort(hybrid_scores, descending=True)[:top_k].tolist()

        results = []
        for idx in top_idx:
            results.append(
                {
                    "query": query,
                    "question": self.df.iloc[idx]["question"],
                    "answer": self.df.iloc[idx]["answer"],
                    "lex_score": float(lex_scores[idx]),
                    "embed_score": float(embed_scores[idx]),
                    "hybrid_score": float(hybrid_scores[idx]),
                }
            )

        return results

_engine = HybridFAQ(
    csv_path=FAQ_CSV_PATH,
    model_name=EMBEDDING_MODEL_NAME,
    w_embed=W_EMBED,
    w_lex=W_LEX,
)


def get_hybrid_results(query: str, top_k: int = 3) -> List[Dict]:
    return _engine.search(query, top_k)
