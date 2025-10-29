import os
import aiohttp
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram import F # Магические фильтры
from fastapi import FastAPI, Request

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

SERVER_URL = "http://127.0.0.1:8000/process_audio/"

load_dotenv(find_dotenv())

bot = Bot(token=os.getenv('TOKEN'))
dp = Dispatcher()


# определим тип сообщений, которые будем получать (остальные игнорируем)
ALLOWED_UPDATES = ["message", "edited_message"]
# "message" - Обычное новое сообщение (текст, фото, voice и т. д.)
# "edited_message" - Пользователь отредактировал уже отправленное сообщение

# Папка для временного хранения скачанных аудиофайлов
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)
# exist_ok=True - не выдаст ошибку, если папка уже создана

# Обработка команды /start
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Привет! Отправь голосовое сообщение для классификации.")

# Ловим все сообщения, где есть .voice или .audio
@dp.message(F.voice | F.audio)
async def handle_audio(message: Message):
    """
    Обработчик голосовых и аудиосообщений:
    1. Определяем какой файл пришёл (voice или audio)
    2. Скачиваем его во временную папку
    3. Отправляем на сервер FastAPI
    4. Получаем JSON с результатом
    5. Отправляем пользователю читаемый ответ
    6. Удаляем временный файл
    """
    # Определяем ID файла в Telegram (нужен для скачивания)
    file_id = message.voice.file_id if message.voice else message.audio.file_id

    # Скачиваем файл из Telegram во временную папку
    file_info = await bot.get_file(file_id)  # Получаем объект файла (путь к файлу, размер, дата и т.д.). ТГ хранит все файлы на своих серверах, а не у нас локально.
    file_path_local = os.path.join(TEMP_DIR, f"{file_id}.ogg")  # Путь для сохранения на нашем ПК.

    await bot.download_file(file_info.file_path, destination=file_path_local)  # Скачиваем файл с серверов Telegram и сохраняем на локальный диск

    await message.answer("🎧 Обрабатываю аудио...")  # Сообщение пользователю пока файл скачивается и готовится к отправке на сервер

    try:
        # Создаём асинхронную HTTP-сессию, которую можно использовать для отправки запросов
        async with aiohttp.ClientSession() as session:

            # Открываем локальный файл с аудио в бинарном режиме ("rb" — read binary)
            with open(file_path_local, "rb") as f:

                form = aiohttp.FormData()

                # Добавляем файл в форму:
                form.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path_local),
                    content_type="audio/ogg"
                )

                # Отправляем POST-запрос на сервер с данными формы
                async with session.post(SERVER_URL, data=form) as resp:

                    # Проверяем статус ответа сервера
                    if resp.status == 200:                          # Если сервер ответил успешно
                        data = await resp.json()                    # Получаем JSON с результатами и преобразуем его из JSON в обычный Python словарь (dict).
                        text = data.get("text", "")                 # берём значение ключа "text" из словаря. "" — это значение по умолчанию, если ключа "text" вдруг нет.
                        label = data.get("label", -1)               # Берём ключ "label" из JSON. Если его нет, используем -1 как значение по умолчанию.
                        description = data.get("description", "")   # Берём ключ "description" из JSON. Если ключ отсутствует, по умолчанию пустая строка.

                        # Формируем читаемый ответ для пользователя
                        reply = (
                            f"💬️Распознанный текст:\n{text}\n\n"
                            f"Класс: {label}\n"
                            f"{description}"
                        )

                    # Сервер вернул ошибку
                    else:
                        reply = f"Ошибка сервера: {resp.status}"

    # Ловим ошибки сети или другие исключения
    except Exception as e:
        reply = f"Ошибка при обработке: {e}"

    finally:
        # Удаляем временный файл, чтобы не захламлять диск
        if os.path.exists(file_path_local):
            os.remove(file_path_local)

    # Отправляем результат пользователю в Telegram
    await message.answer(reply)

@dp.message()
async def handle_text(message: Message):
    """Ответ на обычные текстовые сообщения"""
    await message.answer("🎙 Отправь голосовое сообщение для классификации!")


# Создадим функцию, чтобы бот начал слушать Telegram:
async def main():
    print("Бот запускается...")
    await bot.delete_webhook(drop_pending_updates=True)  # очищаем старый webhook и обновления.
    # у Telegram может остаться “подвешенный” webhook — и тогда polling не заработает.

    await dp.start_polling(bot, allowed_updates=ALLOWED_UPDATES)

if __name__ == "__main__":
    import uvicorn
    asyncio.run(main())

