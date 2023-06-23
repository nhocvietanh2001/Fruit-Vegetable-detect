from flask import Flask, request, Response, jsonify, abort, send_from_directory
import tensorflow as tf
import cv2
import os
import requests

app = Flask(__name__)
model = tf.keras.models.load_model("models/FV.h5")

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}
labelsVietnamese = {0: 'tao', 1: 'chuoi', 2: 'beetroot', 3: 'ot chuong', 4: 'bap cai', 5: 'ot chuong', 6: 'ca rot',
                    7: 'sup lo', 8: 'ot', 9: 'bap', 10: 'dua leo', 11: 'ca tim', 12: 'toi',
                    13: 'gung', 14: 'nho', 15: 'ot', 16: 'kiwi', 17: 'chanh', 18: 'xa lach',
                    19: 'xoai', 20: 'hanh', 21: 'cam', 22: 'ot', 23: 'le', 24: 'dau', 25: 'thom',
                    26: 'luu', 27: 'khoai tay', 28: 'cu cai', 29: 'dau nanh', 30: 'spinach', 31: 'bap',
                    32: 'khoai lang', 33: 'ca chua', 34: 'cu cai', 35: 'dua hau'}


@app.route("/detect", methods=["POST"])
def detect():
    img_raw = request.files.get("image")
    image_name = img_raw.filename
    filepath = os.path.join(os.getcwd(), image_name)
    img_raw.save(filepath)
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255
    image_to_detect = tf.expand_dims(image, 0)

    prediction = model.predict(image_to_detect)
    y_class = prediction.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]

    responses = [{"fruit": res, "confidence": float(prediction[0][y])}]

    os.remove(image_name)

    try:
        return jsonify(responses), 200
    except FileNotFoundError:
        abort(404)


@app.route("/image_search", methods=["POST"])
def image_search():
    img_raw = request.files.get("image")
    image_name = img_raw.filename
    filepath = os.path.join(os.getcwd(), image_name)
    img_raw.save(filepath)
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255
    image_to_detect = tf.expand_dims(image, 0)

    prediction = model.predict(image_to_detect)
    y_class = prediction.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labelsVietnamese[y]
    os.remove(image_name)

    if float(prediction[0][y]) < 0.8:
        responses = [{"message": "No fruit detected!", "success": "false"}]
        return jsonify(responses), 200

    response = requests.get('https://farmhomebackend-production.up.railway.app/fruit/search?name=' + res)
    data = response.json()
    return jsonify(data), 200


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
