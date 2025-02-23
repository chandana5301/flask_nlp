# Multi label Text classification
This project is about Multi label classification using go emotions dataset from hugging face. The Labels are reduced to 13 labels +Neutral class The new labels are 'appreciation', 'surprise', 'negative', 'confusion', 'sadness', 'fear', 'happy', 'desire', 'affection', 'distress', 'confidence', 'gratitude', 'embarrassment', 'neutral'

The model is trained on bert and roberta, save the model as pt or pth file Use the trained model to test and upload it in the same folder as this repository.

Install: pip install flask pip install -r requirements.txt

After uploading your model file change the name of the file in "apps.py" ---> MODEL_PATH=" " Run it using the command "python apps.py"
