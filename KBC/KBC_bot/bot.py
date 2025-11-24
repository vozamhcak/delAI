# bot.py
import asyncio
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

from config import TELEGRAM_TOKEN
from faq_search import search_faq_best

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: types.Message):
    text = (
        "Здравствуйте! Это бот базы знаний по FAQ ЖК.\n\n"
        "Просто отправьте ваш вопрос, а я попробую найти подходящий ответ."
    )
    await message.answer(text)


@dp.message()
async def handle_query(message: types.Message):
    query = (message.text or "").strip()
    if not query:
        await message.answer("Пожалуйста, отправьте текстовый вопрос.")
        return

    await message.answer("Ищу ответ...")

    try:
        # ВАЖНО: поиск блокирующий (модель, диск и т.п.),
        # поэтому выносим его в отдельный поток.
        best = await asyncio.to_thread(search_faq_best, query)

        if best is None:
            await message.answer(
                "Пока я не знаю ответ на этот вопрос.\n"
                "Мы продолжаем пополнять базу знаний, попробуйте переформулировать запрос."
            )
            return

        response = (
            f"<b>Ваш вопрос:</b> {query}\n\n"
            f"<b>Похожий вопрос:</b> {best['question']}\n"
            f"<b>Ответ:</b> {best['answer']}\n\n"
            f"<b>Схожесть (гибридный скор):</b> {best['hybrid_score']:.3f}\n"
            f"<i>(лексический: {best['lex_score']:.3f}, "
            f"эмбеддинговый: {best['embed_score']:.3f})</i>"
        )
        await message.answer(response)

    except FileNotFoundError:
        logger.exception("FAQ CSV file not found")
        await message.answer(
            "Файл базы FAQ не найден. Проверьте путь к CSV в config.py."
        )
    except Exception as e:
        logger.exception("Unexpected error during search")
        await message.answer(f"Произошла ошибка при поиске ответа: {e}")


async def main():
    logger.info("KBC_bot started. Press Ctrl+C to stop.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
