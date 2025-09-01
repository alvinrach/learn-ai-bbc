from fastapi import FastAPI
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer
import nltk
nltk.download('stopwords')
import contractions
import re
from typing import List

class InputData(BaseModel):
    text: List[str]

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

loaded_model = torch.jit.load("./experiments/deployment/3b_model_scripted.pt", map_location=device)
loaded_model.eval()

@app.post('/predict')
async def predict(inp: InputData):
    def txtprocess(txt):
        txt = str(txt).lower()
        txt = contractions.fix(txt)

        txt = re.sub(r'[^a-zA-Z]', ' ', txt)
        txt = re.sub(' +', ' ', txt)

        txt = ' '.join(txt.split())

        return txt

    stop_words = set(nltk.corpus.stopwords.words('english'))

    def remove_stopwords(txt):
        no_stopword_txt = [w for w in txt.split() if not w in stop_words]
        return ' '.join(no_stopword_txt)

    x_test = [txtprocess(i) for i in inp.text]
    x_test = [remove_stopwords(i) for i in x_test]

    paddedtesttext = tokenizer(x_test, padding=True, truncation=True, return_tensors='pt')['input_ids']

    with torch.no_grad():
        paddedtesttext = paddedtesttext.to(device)
        logits = loaded_model(paddedtesttext)
        pred = logits.argmax(dim=1).tolist()
    
    return pred
