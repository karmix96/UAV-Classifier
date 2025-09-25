from fastai.vision.all import load_learner
from huggingface_hub import hf_hub_download
import gradio as gr, os

# (No custom helpers used in your UAV training pipeline)

REPO_ID  = "karmix96/uav_classification_model"          # <-- put your actual model repo
FILENAME = "uav_classification_model.pkl"   # <-- exact filename in that repo

# define token BEFORE using it
token = os.getenv("HF_TOKEN")  # only needed if the model repo is private

# download model from Hub
local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=token)

# load into fastai
learn = load_learner(local_path)   # WARNING: only load pickles you trust
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
    examples=[
        "examples/multirotor1.jpg",
        "examples/fixedwing1.jpg",
    ],
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
