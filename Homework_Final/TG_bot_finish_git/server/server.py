# '''
# Это серверное приложение на FastAPI, которое принимает аудиофайлы, распознаёт речь и классифицирует полученный текст
#
# Этапы работы кода:
#     1.Создаётся сервер с помощью FastAPI
#     2. Создаётся папка uploads/ для хранения загруженных файлов.
#     3. Определяется маршрут /process_audio/, который принимает POST-запрос с аудиофайлом.
#     4. Когда приходит запрос:
#         - Файл сохраняется в uploads/.
#         - Вызывается функция audio() из модуля asr_stub → возвращает текст из аудио.
#         - Вызывается функция classify_text() из classifier_stub - классифицирует текст
#         - Возвращается JSON-ответ, содержащий распознанный текст и результат классификации.
# '''


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from .asr_stub import audio
from .classifier_stub import classify_text

# Создание приложения (Создаём экземпляр FastAPI)
app = FastAPI(title="Classification Server")

# Создание папки для загрузок
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
        # uploads - имя папки, которая будет создана (если ранее не было)
        # exist_ok=True - не выдаст ошибку, если папка уже создана


# Создадим декоратор FastApi. 
@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    """
    Принимает аудиофайл от телеграм-бота, выполняет:
    1. Сохранение файла
    2. Распознавание речи
    3. Классификацию текста
    4. Возврат результата
    """

    try:
        # 1️. Сохраняем аудио
        safe_name = Path(file.filename).name

        # Определим путь для сохранения файла
        file_path = UPLOAD_DIR / safe_name

        # сохраним загруженный аудиофайл на диск
        with open(file_path, "wb") as buffer:
            # Копируем содержимое из загруженного файла (который пришёл от клиента)
            shutil.copyfileobj(file.file, buffer)

        # Преобразование аудио в текст в файле asr_stube.py.
        text = audio(str(file_path))

        # Текст из аудио. Выполняется классификация в файле classifier_stub
        result = classify_text(text)

        # 4️. Ответ в формате JSON
        return JSONResponse(content={
            "text": text,
            "label": result["label"],
            "description": result["description"]
        })

    except Exception as e:
        # если где-то произойдёт ошибка, программа не «упадёт», а выполнит то, что написано в except.

        return JSONResponse(content={"error": str(e)}, status_code=500)
