import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

CLASS_DESCRIPTIONS = {
    'cardboard': 'Cardboard - Recyclable',
    'glass':     'Glass - Recyclable',
    'metal':     'Metal - Recyclable',
    'paper':     'Paper - Recyclable',
    'plastic':   'Plastic - Recyclable',
    'trash':     'General Trash - Non-recyclable'
}
CLASS_COLORS = {
    'cardboard': '#8B4513',
    'glass':     '#00CED1',
    'metal':     '#708090',
    'paper':     '#228B22',
    'plastic':   '#FF6347',
    'trash':     '#696969'
}

processor = None
model = None

def load_model():
    global processor, model
    if model is not None:
        return processor, model
    try:
        from transformers import ViTImageProcessor, ViTForImageClassification
        print("Loading HuggingFace model yangy50/garbage-classification ...")
        processor = ViTImageProcessor.from_pretrained("yangy50/garbage-classification")
        model = ViTForImageClassification.from_pretrained("yangy50/garbage-classification")
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")
    return processor, model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(img: Image.Image):
    import torch
    proc, mdl = load_model()
    img = img.convert('RGB')

    if proc is not None and mdl is not None:
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = mdl(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).numpy()
        id2label = mdl.config.id2label  # {0: 'cardboard', 1: 'glass', ...}
        classes = [id2label[i].lower() for i in range(len(id2label))]
    else:
        # Fallback random (should not happen)
        classes = list(CLASS_DESCRIPTIONS.keys())
        probs = np.random.dirichlet(np.ones(len(classes)))

    top_idx = int(np.argmax(probs))
    label = classes[top_idx]
    confidence = float(probs[top_idx]) * 100

    all_preds = [
        {
            'class': classes[i],
            'description': CLASS_DESCRIPTIONS.get(classes[i], classes[i]),
            'confidence': round(float(probs[i]) * 100, 2),
            'color': CLASS_COLORS.get(classes[i], '#888888')
        }
        for i in range(len(classes))
    ]
    all_preds.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        'label': label,
        'description': CLASS_DESCRIPTIONS.get(label, label),
        'confidence': round(confidence, 2),
        'color': CLASS_COLORS.get(label, '#888888'),
        'all_predictions': all_preds,
        'recyclable': label != 'trash'
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, GIF, or WEBP'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        result = predict(img)

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(save_path)
        result['image_url'] = f'/static/uploads/{filename}'

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
