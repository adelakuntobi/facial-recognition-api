from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the trained model
trained_recognizer = cv2.face.LBPHFaceRecognizer_create()
trained_recognizer.read("face_enc.xml")


def convert_base64_to_image(base64_string):
    # Convert the base64 encoded image to a NumPy array
    nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
    # Decode the NumPy array into an OpenCV image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def compare_faces(face1, face2):
    # Perform face recognition
    label1, confidence1 = trained_recognizer.predict(face1)
    label2, confidence2 = trained_recognizer.predict(face2)

    # Calculate average confidence
    avg_confidence = (confidence1 + confidence2) / 2

    # Convert confidence to percentage
    confidence_percentage = round((100 - avg_confidence), 2)

    return confidence_percentage


def perform_liveness_check(image):
    # Load the pre-trained model for liveness detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'liveness.caffemodel')

    # Convert the image to a blob
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 177, 123))

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform forward pass and get the network output
    output = net.forward()

    # Get the predicted class label (0 for real, 1 for fake)
    predicted_class = np.argmax(output)

    # Return the liveness check result
    liveness_result = "Real" if predicted_class == 0 else "Fake"
    return liveness_result


@app.route('/authenticate', methods=['POST'])
def authenticate():
    # Get the base64 encoded images from the request
    data = request.get_json()
    image1 = data['image1']
    image2 = data['image2']

    # Convert the base64 encoded images to OpenCV images
    img1 = convert_base64_to_image(image1)
    img2 = convert_base64_to_image(image2)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect faces in the images
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces1 = face_cascade.detectMultiScale(
        gray1, scaleFactor=1.1, minNeighbors=5)
    faces2 = face_cascade.detectMultiScale(
        gray2, scaleFactor=1.1, minNeighbors=5)

    # Ensure that exactly one face is detected in each image
    if len(faces1) != 1 or len(faces2) != 1:
        return jsonify({'result': 'error', 'message': 'Exactly one face should be present in each image'})

    # Extract the face regions of interest
    (x1, y1, w1, h1) = faces1[0]
    face1 = gray1[y1:y1 + h1, x1:x1 + w1]

    (x2, y2, w2, h2) = faces2[0]
    face2 = gray2[y2:y2 + h2, x2:x2 + w2]

# Perform liveness check on both images
    liveness_result1 = perform_liveness_check(face1)
    liveness_result2 = perform_liveness_check(face2)

    # Compare faces and calculate confidence percentage
    confidence_percentage = compare_faces(face1, face2)

    # Prepare the response
    response = {
        'confidence_value': confidence_percentage,
        'liveness_result1': liveness_result1,
        'liveness_result2': liveness_result2,
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run()
