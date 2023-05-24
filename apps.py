import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer,RobertaTokenizer,RobertaModel,AutoTokenizer,BertModel
import torch

labelnames=['appreciation', 'surprise','negative','confusion','sadness','fear','happy',
               'desire','affection','distress','confidence','gratitude','embarrassment','neutral']


class bertmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        #self.model=RobertaModel.from_pretrained("roberta-base",return_dict=False)
        self.classifier = torch.nn.Linear(768, 14)
        self.l2 = torch.nn.Dropout(0.2)
    def forward(self, input_ids, attention_mask, token_type_ids):
        _, features = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.classifier(features)
        o1 = self.l2(output)
        return o1

def Model(text):
    test_encoding = (tokenizer.encode_plus(text, max_length=30, truncation=True, return_token_type_ids=True,
                                           padding='max_length', return_tensors='pt')).to("cuda")
    with torch.no_grad():
        pred=[]
        labelcon=[]
        predict={}
        test_prediction = model(**test_encoding)
        copy = test_prediction.cpu().flatten().detach().numpy()
        for label, prediction in zip(labelnames, copy):
            if prediction.any() < 0.5:
                continue
            prediction=round(prediction,2)
            pred.append(prediction)
            labelcon.append(label)
    return pred,labelcon

MODEL_PATH = 'goemotionsbert.pth'
model=bertmodel()

model.load_state_dict(torch.load(MODEL_PATH))
model=model.to("cuda")
#tokenizer=RobertaTokenizer.from_pretrained("roberta-base")
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.values()

    message=str(message)
    prediction,labels = Model(message)
    res = render_template('result.html', prediction=prediction,labels=labels)
    return res

if __name__ == '__main__':
    app.run(debug=True)