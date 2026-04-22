import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)

UPLOAD_FOLDER  = 'uploads'
OUTPUT_FOLDER  = 'output'
WEIGHTS_PATH   = 'generator.weights.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (64, 64)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER']       = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']  = 16 * 1024 * 1024

DRUG_CLASSES = ['amoxicillin', 'atorvastatin', 'metformin']

DRUG_INFO = {
    'amoxicillin': {
        'category': 'Antibiotic',
        'use': 'Bacterial infection treatment',
        'description': 'A broad-spectrum penicillin antibiotic used to treat a range of bacterial infections.',
        'color': '#06e0a3'
    },
    'atorvastatin': {
        'category': 'Statin',
        'use': 'Cholesterol management',
        'description': 'A statin medication used to prevent cardiovascular disease and treat abnormal lipid levels.',
        'color': '#08a4b3'
    },
    'metformin': {
        'category': 'Antidiabetic',
        'use': 'Type 2 diabetes management',
        'description': 'A first-line medication for the treatment of type 2 diabetes, particularly in overweight patients.',
        'color': '#7c3aed'
    }
}

# ── Lightweight Generator (matches training notebook) ────────────────────────

def build_generator():
    inp = layers.Input(shape=(64, 64, 3))
    e1 = layers.LeakyReLU()(layers.Conv2D(32,  4, strides=2, padding='same', use_bias=False)(inp))
    e2 = layers.LeakyReLU()(layers.BatchNormalization()(layers.Conv2D(64,  4, strides=2, padding='same', use_bias=False)(e1)))
    e3 = layers.LeakyReLU()(layers.BatchNormalization()(layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(e2)))

    d1 = layers.ReLU()(layers.BatchNormalization()(layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(e3)))
    d1 = layers.Concatenate()([d1, e2])
    d2 = layers.ReLU()(layers.BatchNormalization()(layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(d1)))
    d2 = layers.Concatenate()([d2, e1])
    out = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(d2)
    return Model(inp, out)

# ── Load generator ────────────────────────────────────────────────────────────

generator = build_generator()

if os.path.exists(WEIGHTS_PATH):
    generator.load_weights(WEIGHTS_PATH)
    print(f'[INFO] Loaded trained weights from {WEIGHTS_PATH}')
else:
    print('[WARN] No trained weights found. Run Untitled.ipynb first for best results.')
    print('[INFO] Running with random weights — output will be noisy but not blank.')

# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    """Load image and normalize to [-1, 1] to match tanh generator output."""
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 127.5 - 1.0   # [-1, 1]
    return tf.expand_dims(arr.astype(np.float32), axis=0)

def postprocess_image(tensor):
    """Convert tanh output [-1, 1] back to uint8 [0, 255]."""
    img = (tensor + 1.0) * 127.5
    return img.clip(0, 255).astype(np.uint8)

def generate_compound(input_path, drug_class):
    input_tensor = preprocess_image(input_path)
    generated    = generator(input_tensor, training=False)[0].numpy()
    generated    = postprocess_image(generated)

    out_dir = os.path.join(OUTPUT_FOLDER, f'output_{drug_class}')
    os.makedirs(out_dir, exist_ok=True)

    filename = f'{drug_class}_{uuid.uuid4().hex[:8]}.png'
    out_path = os.path.join(out_dir, filename)
    plt.imsave(out_path, generated)
    return out_path, filename, drug_class

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('drugassistant.html', drug_classes=DRUG_CLASSES, drug_info=DRUG_INFO)

@app.route('/generate', methods=['POST'])
def generate():
    if 'compound_image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file       = request.files['compound_image']
    drug_class = request.form.get('drug_class', '').lower()

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    if drug_class not in DRUG_CLASSES:
        return jsonify({'error': f'Invalid drug class. Choose from: {", ".join(DRUG_CLASSES)}'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Upload a PNG or JPG image.'}), 400

    filename    = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    try:
        out_path, out_filename, cls = generate_compound(upload_path, drug_class)
    except Exception as e:
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

    return jsonify({
        'success':      True,
        'drug_class':   cls,
        'output_image': f'/output_image/{cls}/{out_filename}',
        'drug_info':    DRUG_INFO.get(cls, {})
    })

@app.route('/output_image/<drug_class>/<filename>')
def serve_output(drug_class, filename):
    directory = os.path.join(OUTPUT_FOLDER, f'output_{drug_class}')
    return send_from_directory(directory, filename)

@app.route('/gallery/<drug_class>')
def gallery(drug_class):
    if drug_class not in DRUG_CLASSES:
        return jsonify({'error': 'Invalid drug class'}), 400
    out_dir = os.path.join(OUTPUT_FOLDER, f'output_{drug_class}')
    images  = []
    if os.path.exists(out_dir):
        images = [
            f'/output_image/{drug_class}/{f}'
            for f in sorted(os.listdir(out_dir))
            if f.lower().endswith('.png')
        ]
    return jsonify({'drug_class': drug_class, 'images': images[:30]})

if __name__ == '__main__':
    app.run(debug=True)
