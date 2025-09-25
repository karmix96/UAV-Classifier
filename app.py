from fastai.vision.all import load_learner
from huggingface_hub import hf_hub_download
import gradio as gr, os

# ---- Model location on the Hub ----
REPO_ID  = "karmix96/uav_classification_model"       
FILENAME = "uav_classification_model.pkl" 

# Download (cached) and load
local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=token)

# WARNING: only load pickles you trust
learn = load_learner(local_path)
labels = learn.dls.vocab

def predict(img):
    pred, idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a UAV photo"),
    outputs=gr.Label(num_top_classes=min(5, len(labels)), label="Prediction"),
    title="UAV Classifier (fastai)",
    description=f"Classes: {', '.join(labels)}",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
