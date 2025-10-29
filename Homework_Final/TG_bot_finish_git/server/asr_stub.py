import os
import torch
import librosa
import soundfile as sf
import whisper
from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from whisper import load_model

def audio(file_path: str) -> str:
    """
    Распознаёт речь из аудиофайла и возвращает текст.
    :param file_path: путь к файлу (ogg, mp3, wav, m4a и т.д.)
    :return: распознанный текст (str)
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Загрузка аудио
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        print(f"🎵 Длина: {len(y)/sr:.2f} с, частота дискретизации: {sr} Гц")

        # Фильтр для шумоподавления
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
            return b, a

        b, a = butter_bandpass(80, 7000, sr, order=5)
        y_bp = filtfilt(b, a, y)
        print("Сгенерировано: полосовой фильтр 80–7000 Гц")


        # Распознавание речи (Whisper)

        # Настройка модели 
        model_size = "medium"  # tiny/base/small/medium/large
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_size, device=device)

        # Сохраняем обработанное аудио во временный WAV-файл
        sf.write("tmp_input.wav", y_bp, sr)

        # Распознавание речи
        result = model.transcribe(
            "tmp_input.wav",
            task="transcribe",
            language="en",
            fp16=torch.cuda.is_available()
        )

        text = result["text"].strip()
        print("\n🎤 Распознанный текст:\n", text)

        return text

    except Exception as e:
        print(f"❌ Ошибка в ASR: {e}")
        return ""