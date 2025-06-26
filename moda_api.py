{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dc6f782b-d6ef-4f5a-a6af-3fab5ebdfd32",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5000\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "from tensorflow.keras.models import load_model\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "import io\n",
    "\n",
    "app = Flask(__name__)\n",
    "model = load_model(\"moda_model.h5\")\n",
    "\n",
    "class_names = [\"T-shirt/top\", \"Trouser\", \"Pullover\", \"Dress\", \"Coat\",\n",
    "               \"Sandal\", \"Shirt\", \"Sneaker\", \"Bag\", \"Ankle boot\"]\n",
    "\n",
    "@app.route(\"/predici\", methods=[\"POST\"])\n",
    "def predici():\n",
    "    if \"file\" not in request.files:\n",
    "        return jsonify({\"errore\": \"Nessun file trovato\"}), 400\n",
    "\n",
    "    file = request.files[\"file\"]\n",
    "    img = Image.open(io.BytesIO(file.read())).convert(\"L\").resize((28, 28))\n",
    "    img_array = np.array(img) / 255.0\n",
    "    img_array = np.expand_dims(img_array, 0)\n",
    "\n",
    "    prediction = model.predict(img_array)\n",
    "    predicted_label = class_names[np.argmax(prediction)]\n",
    "    return jsonify({\"predizione\": predicted_label})\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(port=5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84844a5b-7a5b-46a4-b811-6cfe7d5a5fef",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
