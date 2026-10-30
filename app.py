from pathlib import Path
import re

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# --------------------------------------------------
# 1. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using Fine-Tuned FLAN-T5",
    version="1.0"
)


# --------------------------------------------------
# 2. Model and tokenizer paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "saved_summary_model"


# --------------------------------------------------
# 3. Load the fine-tuned model and tokenizer
# --------------------------------------------------

model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH))

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))


# --------------------------------------------------
# 4. Select the available device
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

model.to(device)
model.eval()


# --------------------------------------------------
# 5. HTML templates
# --------------------------------------------------

templates = Jinja2Templates(directory=str(BASE_DIR))


# --------------------------------------------------
# 6. Input schema
# --------------------------------------------------

class DialogueInput(BaseModel):
    dialogue: str


# --------------------------------------------------
# 7. Text cleaning
# --------------------------------------------------

def clean_data(text):
    if text is None:
        return ""

    text = str(text)

    # Replace line breaks with spaces
    text = re.sub(r"[\r\n]+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Preserve capitalization
    return text.strip()


# --------------------------------------------------
# 8. Generate a summary
# --------------------------------------------------

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    if not dialogue:
        return "Please enter a dialogue to summarize."

    # Tokenize the input dialogue
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate the summary without calculating gradients
    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            min_new_tokens=30,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
    )

    # Convert token IDs into readable text
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# --------------------------------------------------
# 9. Summarization API endpoint
# --------------------------------------------------

@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(dialogue_input.dialogue)

    return {"summary": summary}


# --------------------------------------------------
# 10. Homepage
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )