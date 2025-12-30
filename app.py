from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = 'model.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print("Model not found! Please run train_model.py first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess image
        img = Image.open(file.stream).convert('RGB') # Ensure RGB
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        
        # Based on training: 0 = Organic, 1 = Recycle
        # We use a threshold of 0.5
        result = 'Recycle' if prediction[0][0] > 0.5 else 'Organic'
        confidence = float(prediction[0][0]) if result == 'Recycle' else 1.0 - float(prediction[0][0])

        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)