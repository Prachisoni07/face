


from fastapi import FastAPI, Request, HTTPException,File,UploadFile
from mongoengine import connect, Document, StringField, DateTimeField
from pydantic import BaseModel
from cryptography.fernet import Fernet
from urllib.parse import quote_plus
from datetime import datetime
import face_recognition
import numpy as np
import cv2
import os
import time

# Decrypt password
encryption_key = "__xx3KvJbIu5irVLeGhBIhx5OGiMIdtOCLa4D8upJto="
encrypted_password = "gAAAAABnxX6TY4plt3T-TTM6rwVFSHmIim_NhQYg-mmrTCkDtbiN9EPemqYi3i_ey2ONIx0ylADJUdESsMl8HuamA8mQDRX7Uw=="
cipher_suite = Fernet(encryption_key.encode())
MONGO_USER = "prachi"
decrypted_password = quote_plus(cipher_suite.decrypt(encrypted_password.encode()).decode())

# MongoDB Connection
MONGO_URL = f"mongodb://{MONGO_USER}:{decrypted_password}@localhost/face_recognition?authSource=admin"
connect(host=MONGO_URL)

# Define the FastAPI app
app = FastAPI()

# Allowed IPs for access control
ALLOWED_IPS = {"127.0.0.1", "192.168.1.10", "203.0.113.5"}

@app.middleware("http")
async def check_ip(request: Request, call_next):
    client_ip = request.client.host
    if client_ip not in ALLOWED_IPS:
        raise HTTPException(status_code=403, detail="Access forbidden: IP not allowed")
    return await call_next(request)

@app.get("/")
async def home():
    return {"message": "Hello, World!"}

# # MongoDB Model for storing face encodings
# class Face_embedding(Document):
#     embedding = ListField(required=True)
#     name = StringField()
#     image_path = StringField()
#     timestamp = DateTimeField(default=datetime.utcnow)

# Pydantic model for response
class FaceMatchResult(BaseModel):
    recognized_faces: list

# Store recognized faces dynamically
recognized_faces_history = []

import time  # Import the time module

class FaceImage(Document):
    name = StringField(required=True)
    image_path = StringField()
    timestamp = DateTimeField(default=datetime.utcnow)

@app.post("/save_face/")
async def save_face(name: str):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    time.sleep(3)  
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame from webcam")

    # Ensure static folder exists
    if not os.path.exists("static"):
        os.makedirs("static")

    # Save captured image
    image_filename = f"{name}{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    image_path = os.path.join("static", image_filename)
    cv2.imwrite(image_path, frame)

    # Save to MongoDB
    try:
        face = FaceImage(name=name, image_path=image_path)
        face.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving face to database: {e}")

    return {"message": "Face saved successfully"}

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two face encodings."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/recognize_faces/")
async def recognize_faces():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    time.sleep(3)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame from webcam")

    # Convert captured frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face encodings from the captured frame (for multiple faces)
    captured_encodings = face_recognition.face_encodings(frame_rgb)
    if len(captured_encodings) == 0:
        return {"recognized_faces": ["No face detected"]}

    recognized_faces = []

    for captured_encoding in captured_encodings:
        best_match_name = "Unknown"
        best_similarity = 0.0

        for db_face in FaceImage.objects:
            if not os.path.exists(db_face.image_path):
                continue  # Skip if the image file is missing

            # Load the stored image from the static folder
            stored_img = cv2.imread(db_face.image_path)
            if stored_img is None:
                continue  # Skip if the image can't be read

            stored_img_rgb = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)

            # Get face encodings from the stored image
            stored_encodings = face_recognition.face_encodings(stored_img_rgb)
            if len(stored_encodings) == 0:
                continue  # Skip if no face is found in the stored image

            # Compare captured face with stored faces
            matches = face_recognition.compare_faces(stored_encodings, captured_encoding)
            similarity_scores = [cosine_similarity(stored_enc, captured_encoding) for stored_enc in stored_encodings]

            if True in matches:
                best_match_index = np.argmax(similarity_scores)  # Get the highest similarity score
                if similarity_scores[best_match_index] > best_similarity:
                    best_match_name = db_face.name
                    best_similarity = round(similarity_scores[best_match_index], 4)

        recognized_faces.append({"name": best_match_name, "similarity": best_similarity})

    return {"recognized_faces": recognized_faces}
# Pydantic model for response
class VideoRecognitionResult(BaseModel):
    message: str
    original_video_path: str
    processed_video_path: str
    recognized_faces: list

# Ensure the static/video folder exists
if not os.path.exists("static/video"):
    os.makedirs("static/video")

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two face encodings."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/capture_and_recognize_video/")
async def capture_and_recognize_video():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    # Get frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define video codec and create VideoWriter objects
    video_filename = f"video_{datetime.now().strftime('%Y%m%d%H%M%S')}.avi"
    original_video_path = os.path.join("static/video", video_filename)
    processed_video_path = os.path.join("static/video", f"processed_{video_filename}")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20.0
    out = cv2.VideoWriter(original_video_path, fourcc, fps, (frame_width, frame_height))
    processed_out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

    recognized_faces = []
    start_time = time.time()
    capture_duration = 10

    while int(time.time() - start_time) < capture_duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the original frame to the video file
        out.write(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            best_match_name = "Unknown"
            best_similarity = 0.0

            for db_face in FaceImage.objects:
                if not os.path.exists(db_face.image_path):
                    continue

                stored_img = cv2.imread(db_face.image_path)
                if stored_img is None:
                    continue

                stored_img_rgb = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)
                stored_encodings = face_recognition.face_encodings(stored_img_rgb)

                if len(stored_encodings) == 0:
                    continue

                matches = face_recognition.compare_faces(stored_encodings, face_encoding)
                similarity_scores = [cosine_similarity(stored_enc, face_encoding) for stored_enc in stored_encodings]

                if True in matches:
                    best_match_index = np.argmax(similarity_scores)
                    if similarity_scores[best_match_index] > best_similarity:
                        best_match_name = db_face.name
                        best_similarity = round(similarity_scores[best_match_index], 4)

            recognized_faces.append({"name": best_match_name, "similarity": best_similarity})

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, best_match_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        processed_out.write(frame)

    cap.release()
    out.release()
    processed_out.release()

    return {
        "message": "Video captured and processed successfully",
        "original_video_path": original_video_path,
        "processed_video_path": processed_video_path,
        "recognized_faces": recognized_faces
    }
    
@app.post("/recognize_faces_video/")
async def recognize_faces_video(file: UploadFile = File(...)):
    # Save the uploaded video file temporarily
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    recognized_faces_in_video = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert captured frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face encodings from the captured frame (for multiple faces)
        captured_encodings = face_recognition.face_encodings(frame_rgb)
        if len(captured_encodings) == 0:
            recognized_faces_in_video.append({"frame": cap.get(cv2.CAP_PROP_POS_FRAMES), "recognized_faces": ["No face detected"]})
            continue

        recognized_faces = []

        for captured_encoding in captured_encodings:
            best_match_name = "Unknown"
            best_similarity = 0.0

            for db_face in FaceImage.objects:
                if not os.path.exists(db_face.image_path):
                    continue  # Skip if the image file is missing

                # Load the stored image from the static folder
                stored_img = cv2.imread(db_face.image_path)
                if stored_img is None:
                    continue  # Skip if the image can't be read

                stored_img_rgb = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)

                # Get face encodings from the stored image
                stored_encodings = face_recognition.face_encodings(stored_img_rgb)
                if len(stored_encodings) == 0:
                    continue  # Skip if no face is found in the stored image

                # Compare captured face with stored faces
                matches = face_recognition.compare_faces(stored_encodings, captured_encoding)
                similarity_scores = [cosine_similarity(stored_enc, captured_encoding) for stored_enc in stored_encodings]

                if True in matches:
                    best_match_index = np.argmax(similarity_scores)  # Get the highest similarity score
                    if similarity_scores[best_match_index] > best_similarity:
                        best_match_name = db_face.name
                        best_similarity = round(similarity_scores[best_match_index], 4)

            if not any(face["name"] == best_match_name for face in recognized_faces):
                recognized_faces.append({"name": best_match_name, "similarity": best_similarity})

        recognized_faces_in_video.append({"frame": cap.get(cv2.CAP_PROP_POS_FRAMES), "recognized_faces": recognized_faces})

    cap.release()
    os.remove(video_path)  # Clean up the temporary video file

    return {"recognized_faces_in_video": recognized_faces_in_video}