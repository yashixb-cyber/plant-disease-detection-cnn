"""
Plant Disease Detection - web app (run on your own laptop)

Usage:
    pip install -r requirements.txt
    python app/app.py                         # uses models/mobilenetv2_field.keras if present,
                                              # otherwise models/mobilenetv2.keras
    python app/app.py --model models/basic_cnn.keras
    python app/app.py --share                 # also create a public link

Before running, download from Google Drive (MyDrive/plant-disease-project/models/):
    mobilenetv2_field.keras (from Notebook 4, recommended), mobilenetv2.keras, basic_cnn.keras,
    class_names.json  ->  put them in the  models/  folder.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pretty_name(folder_name):
    """'Tomato___Early_blight' -> 'Tomato - Early blight'"""
    crop, disease = folder_name.split('___')
    crop = crop.replace('_', ' ').replace(',', '').strip()
    disease = disease.replace('_', ' ').strip()
    return f'{crop} - {disease[0].upper() + disease[1:]}'


def preprocess(pil_img):
    """Same as training: fix phone-photo rotation, RGB, 224x224, pixel values 0-255
    (the model normalises internally)."""
    img = ImageOps.exif_transpose(pil_img).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None, ...]


def main():
    parser = argparse.ArgumentParser(description='Plant disease detection web app')
    field_model = os.path.join(ROOT, 'models', 'mobilenetv2_field.keras')
    default_model = field_model if os.path.exists(field_model) else os.path.join(ROOT, 'models', 'mobilenetv2.keras')
    parser.add_argument('--model', default=default_model)
    parser.add_argument('--classes', default=os.path.join(ROOT, 'models', 'class_names.json'))
    parser.add_argument('--share', action='store_true', help='create a public gradio.live link')
    args = parser.parse_args()

    for path in [args.model, args.classes]:
        if not os.path.exists(path):
            raise SystemExit(f'File not found: {path}\n'
                             'Download the trained model files from Google Drive into the models/ folder.')

    import gradio as gr
    from tensorflow import keras

    model = keras.models.load_model(args.model)
    with open(args.classes) as f:
        class_names = json.load(f)
    labels = [pretty_name(c) for c in class_names]

    def predict(pil_img):
        if pil_img is None:
            return {}, 'Please upload a leaf image.'
        probs = model.predict(preprocess(pil_img), verbose=0)[0]
        top = int(probs.argmax())
        status = '✅ Healthy' if 'healthy' in class_names[top] else '⚠️ Disease detected'
        msg = f'### {status}\n**Prediction:** {labels[top]}  \n**Confidence:** {probs[top] * 100:.1f}%'
        if probs[top] < 0.5:
            msg += '\n\n_Low confidence: try a clearer photo of a single leaf on a plain background._'
        return {labels[i]: float(p) for i, p in enumerate(probs)}, msg

    examples_dir = os.path.join(ROOT, 'app', 'examples')
    examples = ([[os.path.join(examples_dir, f)] for f in sorted(os.listdir(examples_dir))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if os.path.isdir(examples_dir) else None)

    gr.Interface(
        fn=predict,
        inputs=gr.Image(type='pil', label='Upload a leaf photo'),
        outputs=[gr.Label(num_top_classes=3, label='Top-3 predictions'), gr.Markdown()],
        title='🌿 Plant Disease Detection (CNN)',
        description=f'Model: {os.path.basename(args.model)} · 38 classes · 14 crops. '
                    'Study project, not a replacement for an agricultural expert.',
        examples=examples or None,
    ).launch(share=args.share)


if __name__ == '__main__':
    main()
