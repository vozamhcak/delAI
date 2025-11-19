import os
import ast
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from lex_index import LexIndex, Chunk


def cosine_to_unit(scores: torch.Tensor) -> torch.Tensor:
    """Преобразует косинус [-1, 1] → [0, 1]."""
    return (scores + 1.0) / 2.0


def load_faq_data(csv_path: str) -> pd.DataFrame:
    """Загружает CSV с вопросами, ответами и эмбеддингами."""
    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
    return df


def build_lex_index(df: pd.DataFrame) -> LexIndex:
    """Создает лексический индекс по всем вопросам."""
    chunks = [
        Chunk(id=i, text=q, meta={"answer": df.iloc[i]["answer"]})
        for i, q in enumerate(df["question"].tolist())
    ]
    return LexIndex(chunks)


def search_faq(query: str,
               csv_path: str,
               model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
               w_embed: float = 0.6,
               w_lex: float = 0.4,
               top_k: int = 5):
    """Находит наиболее похожие вопросы и ответы."""

    print("Загрузка базы FAQ...")
    df = load_faq_data(csv_path)

    print("Создание лексического индекса...")
    lex_index = build_lex_index(df)

    print("Загрузка модели эмбеддингов...")
    model = SentenceTransformer(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_emb = model.encode([query], convert_to_tensor=True, normalize_embeddings=False).to(device)
    all_embs = torch.tensor(np.stack(df["embedding"].values), device=device)

    cos_scores = util.cos_sim(query_emb, all_embs)[0]
    cos_unit = cosine_to_unit(cos_scores)

    lex_scores = torch.tensor(lex_index.score_all(query), dtype=torch.float32, device=device)

    hybrid = w_embed * cos_unit + w_lex * lex_scores

    top_idx = torch.argsort(hybrid, descending=True)[:top_k].tolist()

    print(f"\nЗапрос: {query}")
    print("Результаты поиска:")

    for rank, idx in enumerate(top_idx, 1):
        q = df.iloc[idx]["question"]
        a = df.iloc[idx]["answer"]
        score = float(hybrid[idx])
        print(f"\n{rank}. Вопрос: {q}\nОтвет: {a}\nОценка: {score:.3f}")

    return [(df.iloc[i]["question"], df.iloc[i]["answer"], float(hybrid[i])) for i in top_idx]


if __name__ == "__main__":
    FAQ_CSV_PATH = "faq_data.csv"
    QUERY = "Как часто убирают снег?"

    try:
        results = search_faq(
            query=QUERY,
            csv_path=FAQ_CSV_PATH,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            w_embed=0.6,
            w_lex=0.4,
            top_k=5
        )

        print(f"\nНайдено {len(results)} результатов.")

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
