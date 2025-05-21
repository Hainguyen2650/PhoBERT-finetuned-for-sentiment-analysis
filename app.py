import json
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import gradio as gr
import plotly.express as px
import pandas as pd
from rich import print as rprint
import threading
import time
import os

class CustomClassifier(nn.Module):
    def __init__(self, pretrained_name: str, num_labels: int, dropout_prob: float = 0.3):
        super().__init__()
        self.base = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        x = self.dropout(cls)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

# Load model and tokenizer
with open("MainModel/custom_config.json", "r") as f:
    cfg = json.load(f)

model = CustomClassifier(pretrained_name=cfg["pretrained_name"], num_labels=cfg["num_labels"])
state_dict = load_file("MainModel/model.safetensors")
model.load_state_dict(state_dict)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("MainModel")


id2label = {0: 'anger', 1: 'disgust', 2: 'enjoyment', 3: 'fear', 4: 'other', 5: 'sadness', 6: 'surprise'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐",
    "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

last_active = time.time()
TIMEOUT = 600  # 10 minutes

def update_activity():
    global last_active
    last_active = time.time()

def monitor_inactivity():
    while True:
        if time.time() - last_active > TIMEOUT:
            rprint("[bold red]No activity detected. Shutting down the app.[/bold red]")
            os._exit(0)
        time.sleep(30)

threading.Thread(target=monitor_inactivity, daemon=True).start()

def predict_with_proba(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = id2label[pred_id]
        return pred_label, probs

def gradio_predict(text):
    update_activity()
    label, probs = predict_with_proba(text)
    emoji = emotions_emoji_dict.get(label, "")
    proba_df = pd.DataFrame({
        "Emotion": [id2label[i] for i in range(len(probs))],
        "Probability": probs
    })

    fig = px.bar(proba_df, x="Emotion", y="Probability", color="Emotion", title="Xác suất cảm xúc",
                 labels={"Probability": "Xác suất"}, height=400)

    return f"{label} {emoji}", fig  #, proba_df

# Custom theme
custom_css = """
#component-0 > div {
    font-size: 26px;
    font-weight: bold;
    color: #333;
}
.gr-button-primary {
    background-color: green !important;
    color: white !important;
    font-size: 18px !important;
    padding: 10px 20px !important;
}
textarea, .output_class, .input_class {
    font-size: 18px !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='font-size: 40px; color: #2c3e50;'>🧠 PHÂN TÍCH CẢM XÚC VĂN BẢN</h1>")
    gr.Markdown("### Nhập một đoạn văn và hệ thống sẽ phân loại cảm xúc của nó.")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=5, label="Nhập văn bản", placeholder="Ví dụ: Hôm nay trời thật đẹp ☀️")
            submit_btn = gr.Button("Dự đoán cảm xúc", elem_classes="gr-button-primary")
        with gr.Column():
            label_output = gr.Label(label="Kết quả")
            #proba_table = gr.Dataframe(headers=["Cảm xúc", "Xác suất"], visible=False)
            proba_plot = gr.Plot(label="Biểu đồ xác suất")

    submit_btn.click(fn=gradio_predict, inputs=input_text, outputs=[label_output, proba_plot]) #, proba_table

if __name__ == "__main__":
    rprint("[bold green]🚀 Gradio app running![/bold green] [blue underline]http://localhost:7860[/blue underline]")
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
