import asyncio
import logging
import os

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
API_URL = os.environ.get("API_URL", "http://localhost:8000")
TOP_K = 3

dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer(
        "Send me a text query (e.g. cat on the sofa) — "
        "I'll find matching photos in the collection."
    )


@dp.message(F.text)
async def on_search(message: Message):
    query = message.text.strip()
    async with httpx.AsyncClient(base_url=API_URL, timeout=30) as client:
        try:
            resp = await client.post("/search", json={"query": query, "top_k": TOP_K})
            resp.raise_for_status()
        except httpx.ConnectError:
            await message.answer(f"API is unreachable. Is uvicorn running on {API_URL}?")
            return
        except httpx.HTTPStatusError as e:
            await message.answer(f"API error: {e.response.status_code}")
            return

        results = resp.json()["results"]
        if not results:
            await message.answer("No matches found.")
            return

        for r in results:
            image_resp = await client.get(r["url"])
            if image_resp.status_code != 200:
                continue
            filename = r["path"].split("/")[-1]
            await message.answer_photo(
                BufferedInputFile(image_resp.content, filename=filename),
                caption=f"{filename} | score: {r['score']:.4f}",
            )


async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
