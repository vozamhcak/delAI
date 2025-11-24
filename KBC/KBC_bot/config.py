
import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8344814091:AAHdMa9A4AdIkXZBBaRlo6p73QwrFg-8NkU")

FAQ_CSV_PATH = os.getenv("FAQ_CSV_PATH", "faq_data.csv")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

W_EMBED = 0.5
W_LEX = 0.5

HYBRID_THRESHOLD = 0.75

TOP_K = 5
