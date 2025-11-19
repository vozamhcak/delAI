import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from faq_search import search_faq

TOKEN = "8344814091:AAHdMa9A4AdIkXZBBaRlo6p73QwrFg-8NkU"  
FAQ_CSV_PATH = "faq_data.csv"

logging.basicConfig(level=logging.INFO)


bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

@dp.message(CommandStart())
async def start(message: types.Message):
    text = (
        "Здравствуйте! Это бот для поиска по FAQ.\n\n"
        "Отправьте мне вопрос — я постараюсь найти наиболее подходящий ответ."
    )
    await message.answer(text)

@dp.message()
async def handle_query(message: types.Message):
    query = message.text.strip()
    if not query:
        await message.answer("Пожалуйста, отправьте текстовый вопрос.")
        return

    await message.answer("Ищу ответ, подождите...")

    try:
        results = search_faq(
            query=query,
            csv_path=FAQ_CSV_PATH,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            w_embed=0.6,
            w_lex=0.4,
            top_k=3
        )

        if not results:
            await message.answer("Не удалось найти подходящий ответ.")
            return

        best_q, best_a, best_score = results[0]
        response = (
            f"<b>Ваш вопрос:</b> {query}\n\n"
            f"<b>Похожий вопрос:</b> {best_q}\n"
            f"<b>Ответ:</b> {best_a}\n"
            f"<b>Схожесть:</b> {best_score:.3f}"
        )
        await message.answer(response)

    except FileNotFoundError:
        await message.answer("Файл базы FAQ не найден. Проверьте путь к CSV.")
    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")

async def main():
    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
