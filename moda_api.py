from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).convert("L").resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        # Simulazione finta
        prediction_index = np.random.randint(0, len(class_names))
        label = class_names[prediction_index]

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🧠 Janym API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
