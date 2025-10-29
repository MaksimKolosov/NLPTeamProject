from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# путь к папке с обученной моделью
MODEL_PATH = r"C:\Users\Konsta\PycharmProjects\pythonProject\TG_bot\results_dataset"  

# Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Определяем устройство: GPU если есть, иначе CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # переводим модель в режим оценки

# Функция для классификации текста
def classify_text(text: str) -> dict:
    # Токенизация текста
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Прогон через модель
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        label_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, label_id].item()

       # Присвоим класс новости
    id2label = {
        0: "❌ FAKE❌",
        1: "✅ПРАВДИВАЯ✅"
    }
    label_name = id2label[label_id]


    # Формируем описание для бота
    description = f"Новость классифицирована как: {label_name}"

    return {"label": label_id, "description": description}

