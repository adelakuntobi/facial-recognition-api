from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)


@app.route('/facial_recognition', methods=['POST'])
def facial_recognition():
    # Get the request data containing two base64-encoded images
    data = request.get_json()
    image1_base64 = data['image1']
    image2_base64 = data['image2']

    # Convert the base64-encoded images to OpenCV format
    image1 = convert_base64_to_image(image1_base64)
    image2 = convert_base64_to_image(image2_base64)

    # Perform facial recognition
    confidence_value = compare_images(image1, image2)

    # Perform liveness check (you need to integrate liveness detection library for this)
    liveness_check = perform_liveness_check(image1)

    # Prepare the response
    response = {
        'confidence_value': confidence_value,
        'liveness_check': liveness_check
    }

    return jsonify(response)


def convert_base64_to_image(base64_string):
    # Decode the base64 string and convert it to an OpenCV image
    decoded_data = base64.b64decode(base64_string)
    np_array = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


def compare_images(image1, image2):
    # Perform facial recognition by comparing two images
    image1_encodings = face_recognition.face_encodings(image1)
    image2_encodings = face_recognition.face_encodings(image2)

    if len(image1_encodings) == 0 or len(image2_encodings) == 0:
        # No faces found in one or both images
        return 0.0

    # Compare the first face encoding from image1 with all face encodings from image2
    face_distances = face_recognition.face_distance(image1_encodings, image2_encodings[0])
    confidence_value = 1.0 - np.mean(face_distances)

    return confidence_value


import cv2

def perform_liveness_check(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a suitable liveness detection algorithm
    # For example, you can use eye blink detection as a simple liveness check

    # Load pre-trained eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect eyes in the grayscale image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If two eyes are detected, consider it as a live face
    if len(eyes) >= 2:
        return True
    else:
        return False


if __name__ == '__main__':
    app.run(debug=True)
