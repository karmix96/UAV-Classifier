# UAV Classifier (fastai + Gradio)

Binary UAV image classifier (Multirotor vs Fixed-wing). Training in fastai; served with Gradio.

## Demo
- **Hugging Face Space:** <https://huggingface.co/spaces/karmix96/uav_classifier_app>
- **Kaggle Notebook:** <https://www.kaggle.com/code/michailkaralis/uav-classification>


## Training
- Notebook: [notebooks/train_uav_classifier.py](notebooks/train_uav_classifier.py)
- Model export: `models/uav_classification_model.pkl` *(not committed; uploaded under GitHub Releases)*

## Results

![Demo](assets/demo1.png)
![Training results](assets/demo2.png)

## Ethics & Limitations

Small web-scraped dataset; not for safety-critical use.

## Run locally
```bash
pip install -r app/requirements.txt
python app/app.py
