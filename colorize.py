from flask import Flask, request, render_template, send_file
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Paths to load the model
MODEL_DIR = r"C:\Users\UseR\PycharmProjects\pythonProject\model"
PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")

# Load Model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load and validate image
    image = cv2.imread(filepath)
    if image is None:
        return "Error processing image"

    # 🔹 Resize Image to 800 pixels width while maintaining aspect ratio
    height, width = image.shape[:2]

    # Resize only if width exceeds 1000 pixels
    if width > 1000:
        new_width = 1000
        new_height = int((new_width / width) * height)  # Maintain aspect ratio
        image = cv2.resize(image, (new_width, new_height))

    # Convert to LAB color space
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorization
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Save and return the colorized image
    result_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(result_path, colorized)

    return render_template("result.html", original_image=filepath, result_image=result_path)


if __name__ == '__main__':
    app.run(debug=True)
