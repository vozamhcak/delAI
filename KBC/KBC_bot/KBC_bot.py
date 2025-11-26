# KBC_bot.py
import asyncio
import json
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

from openai import OpenAI

from config import (
    TELEGRAM_TOKEN,
    SYSTEM_PROMPT,
    HYBRID_THRESHOLD,
    MAX_QA_PAIRS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_BASE,
    DEEPSEEK_MODEL,
)
from faq_search import get_hybrid_results


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Проверки конфигурации
if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
    raise RuntimeError(
        "TELEGRAM_TOKEN не задан. Установите переменную окружения TELEGRAM_BOT_TOKEN "
        "или пропишите токен прямо в config.py (для отладки)."
    )

if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY не задан. Установите переменную окружения DEEPSEEK_API_KEY "
        "с вашим DeepSeek API ключом."
    )

# Инициализация DeepSeek-клиента (OpenAI-совместимый)
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
)

# Telegram-бот
bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()


def _build_qa_pairs_for_prompt(results, threshold: float, max_pairs: int):
    """
    Формирует список qa_pairs для DeepSeek:
    - берём результаты в порядке убывания hybrid_score
    - оставляем только те, у кого hybrid_score > threshold
    - ограничиваем числом max_pairs
    Формат элемента:
    {
        "question": "...",
        "answer": "..."
    }
    """
    qa_pairs = []
    for r in results:
        if r.get("hybrid_score", 0.0) > threshold:
            qa_pairs.append(
                {
                    "question": r.get("question", ""),
                    "answer": r.get("answer", ""),
                }
            )
        if len(qa_pairs) >= max_pairs:
            break
    return qa_pairs


def _call_deepseek_sync(user_query: str, qa_pairs):
    """
    Синхронный вызов DeepSeek API.
    На вход:
      - user_query: текст запроса пользователя
      - qa_pairs: список {"question": ..., "answer": ...}
    Возвращает: текст ответа от модели.
    """
    # Собираем payload в формате, понятном для LLM
    payload = {
        "user_query": user_query,
        "qa_pairs": qa_pairs,
    }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            # Отправляем user_query и qa_pairs как JSON-текст
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        stream=False,
    )

    return response.choices[0].message.content.strip()


async def call_deepseek(user_query: str, qa_pairs):
    """
    Асинхронная обёртка над _call_deepseek_sync,
    чтобы не блокировать event loop.
    """
    return await asyncio.to_thread(_call_deepseek_sync, user_query, qa_pairs)


@dp.message(CommandStart())
async def start(message: types.Message):
    text = (
        "Здравствуйте! Я виртуальный диспетчер для жителей дома.\n\n"
        "Отправьте мне ваш вопрос — я постараюсь помочь, используя базу знаний и ИИ."
    )
    await message.answer(text)


@dp.message()
async def handle_query(message: types.Message):
    user_query = (message.text or "").strip()
    if not user_query:
        await message.answer("Пожалуйста, отправьте текстовый вопрос.")
        return

    await message.answer("Ищу ответ...")

    try:
        # 1. Гибридный поиск по FAQ: берём top-3 кандидата
        results = await asyncio.to_thread(get_hybrid_results, user_query, MAX_QA_PAIRS)

        # 2. Фильтруем по порогу hybrid_score > 0.6 и оставляем не более 3 пар
        qa_pairs = _build_qa_pairs_for_prompt(results, HYBRID_THRESHOLD, MAX_QA_PAIRS)

        logger.info(
            "Query: %s | Retrieved %d results | qa_pairs used: %d",
            user_query,
            len(results),
            len(qa_pairs),
        )

        # 3. Отправляем user_query и qa_pairs в DeepSeek
        answer_text = await call_deepseek(user_query, qa_pairs)

        # 4. Возвращаем пользователю ответ от DeepSeek
        await message.answer(answer_text)

    except Exception as e:
        logger.exception("Ошибка при обработке запроса")
        await message.answer(
            "Произошла техническая ошибка при обработке запроса. "
            "Попробуйте ещё раз позже или обратитесь к оператору."
        )


async def main():
    logger.info("KBC_bot запущен. Нажмите Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
