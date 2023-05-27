import cv2
import os
import numpy as np

def train_model(data_folder):
    # Create a face recognition instance
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Prepare training data
    faces = []
    labels = []
    label_map = {}

    # Iterate over the folders in the data folder
    for label, person_name in enumerate(os.listdir(data_folder)):
        person_folder = os.path.join(data_folder, person_name)
        if os.path.isdir(person_folder):
            # Iterate over the images in the person's folder
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                # Read the image
                image = cv2.imread(image_path)
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Detect faces in the grayscale image
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                # Iterate over the detected faces
                for (x, y, w, h) in faces_rect:
                    # Extract the face region of interest
                    face = gray[y:y + h, x:x + w]
                    # Append the face and its corresponding label
                    faces.append(face)
                    labels.append(label)

            # Map the label to the person's name
            label_map[label] = person_name

    # Train the face recognition model
    face_recognizer.train(faces, np.array(labels))

    return face_recognizer, label_map


# Specify the folder containing the training data
training_data_folder = "Images"

# Train the model and obtain the trained recognizer and label map
trained_recognizer, label_map = train_model(training_data_folder)

# Save the trained model to a file
trained_recognizer.save("trained_model.xml")
