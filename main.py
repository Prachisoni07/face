
# from mongoengine import connect, Document, StringField, ListField, DateTimeField
# from fastapi import FastAPI, Request, HTTPException
# import os
# from urllib.parse import quote_plus
# from cryptography.fernet import Fernet
# import cv2
# import numpy as np
# from deepface import DeepFace
# from pydantic import BaseModel
# from datetime import datetime

# # Get encrypted password & key from environment
# encryption_key = "__xx3KvJbIu5irVLeGhBIhx5OGiMIdtOCLa4D8upJto="
# encrypted_password = "gAAAAABnxX6TY4plt3T-TTM6rwVFSHmIim_NhQYg-mmrTCkDtbiN9EPemqYi3i_ey2ONIx0ylADJUdESsMl8HuamA8mQDRX7Uw=="

# # Decrypt password
# cipher_suite = Fernet(encryption_key.encode())
# MONGO_USER = "prachi"
# decrypted_password = quote_plus(cipher_suite.decrypt(encrypted_password.encode()).decode())

# # MongoDB Connection
# MONGO_URL = f"mongodb://{MONGO_USER}:{decrypted_password}@localhost/face_recognisation?authSource=admin"
# connect(host=MONGO_URL)

# # Define the FastAPI app
# app = FastAPI()

# # Allowed IPs for access control
# ALLOWED_IPS = {"127.0.0.1", "192.168.1.10", "203.0.113.5"}

# @app.middleware("http")
# async def check_ip(request: Request, call_next):
#     client_ip = request.client.host  # Get request's source IP
#     if client_ip not in ALLOWED_IPS:
#         raise HTTPException(status_code=403, detail="Access forbidden: IP not allowed")
#     response = await call_next(request)
#     return response

# @app.get("/")
# async def home():
#     return {"message": "Hello, World!"}

# # MongoDB Model for storing face embeddings
# class Face_embedding(Document):
#     embedding = ListField(required=True) 
#     name = StringField()  
#     timestamp = DateTimeField(default=datetime.utcnow)

# # Pydantic model for response
# class FaceMatchResult(BaseModel):
#     matched_faces: list
#     unmatched_faces: list

# @app.post("/recognize_faces/")
# async def recognize_faces():
#     # Open webcam
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         raise HTTPException(status_code=500, detail="Could not open webcam")

#     ret, frame = cap.read()
#     if not ret:
#         raise HTTPException(status_code=500, detail="Could not read frame from webcam")

#     # Release the webcam
#     cap.release()

#     # Save the frame temporarily to a file (DeepFace requires a file path)
#     temp_image_path = "tempframe.jpg"
#     cv2.imwrite(temp_image_path, frame)

#     # Initialize lists for matched and unmatched faces
#     matched_faces = []
#     unmatched_faces = []

#     # Loop through each face embedding in the database
#     for face in Face_embedding.objects:
#         stored_embedding = np.array(face.embedding)

#         # Extract the face embedding from the new frame
#         embedding = DeepFace.represent(
#             img_path=temp_image_path,
#             model_name="Facenet",
#             enforce_detection=False
#         )

#         if embedding:
#             current_embedding = np.array(embedding[0]["embedding"])
            
#             # Compute the cosine similarity
#             similarity = np.dot(current_embedding, stored_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding))

#             # Define a threshold for similarity (e.g., 0.6 or higher is a match)
#             if similarity > 0.6:
#                 matched_faces.append(face.name)
#             else:
#                 unmatched_faces.append("Unknown")

#     # Delete the temporary image file
#     os.remove(temp_image_path)

#     # Return the results
#     return FaceMatchResult(matched_faces=matched_faces, unmatched_faces=unmatched_faces)

# @app.post("/save_face/")
# async def save_face(name: str):
#     # Open webcam
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         raise HTTPException(status_code=500, detail="Could not open webcam")

#     ret, frame = cap.read()
#     if not ret:
#         raise HTTPException(status_code=500, detail="Could not read frame from webcam")

#     # Release the webcam
#     cap.release()

#     # Save the frame temporarily to a file (DeepFace requires a file path)
#     temp_image_path = "temp_frame.jpg"
#     cv2.imwrite(temp_image_path, frame)

#     # Use DeepFace to extract the face embedding
#     try:
#         embedding = DeepFace.represent(
#             img_path=temp_image_path,
#             model_name="Facenet",  # Use the Facenet model
#             enforce_detection=False,  # Skip if no face is detected
#         )

#         if not embedding:
#             raise HTTPException(status_code=400, detail="No faces found in the frame")

#         # Extract the embedding list
#         embedding_list = embedding[0]["embedding"]

#         # Save to MongoDB
#         try:
#             face = Face_embedding(name=name, embedding=embedding_list)
#             face.save()
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error saving face to database: {e}")

#         # Delete the temporary image file
#         os.remove(temp_image_path)

#         return {"message": "Face saved successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing face: {e}")
from mongoengine import connect, Document, StringField, ListField, DateTimeField
from fastapi import FastAPI, Request, HTTPException
import os
from urllib.parse import quote_plus
from cryptography.fernet import Fernet
import cv2
import numpy as np
from deepface import DeepFace
from pydantic import BaseModel
from datetime import datetime

# Get encrypted password & key from environment
encryption_key = "__xx3KvJbIu5irVLeGhBIhx5OGiMIdtOCLa4D8upJto="
encrypted_password = "gAAAAABnxX6TY4plt3T-TTM6rwVFSHmIim_NhQYg-mmrTCkDtbiN9EPemqYi3i_ey2ONIx0ylADJUdESsMl8HuamA8mQDRX7Uw=="

# Decrypt password
cipher_suite = Fernet(encryption_key.encode())
MONGO_USER = "prachi"
decrypted_password = quote_plus(cipher_suite.decrypt(encrypted_password.encode()).decode())

# MongoDB Connection
MONGO_URL = f"mongodb://{MONGO_USER}:{decrypted_password}@localhost/face_recognisation?authSource=admin"
connect(host=MONGO_URL)

# Define the FastAPI app
app = FastAPI()

# Allowed IPs for access control
ALLOWED_IPS = {"127.0.0.1", "192.168.1.10", "203.0.113.5"}

@app.middleware("http")
async def check_ip(request: Request, call_next):
    client_ip = request.client.host  # Get request's source IP
    if client_ip not in ALLOWED_IPS:
        raise HTTPException(status_code=403, detail="Access forbidden: IP not allowed")
    response = await call_next(request)
    return response

@app.get("/")
async def home():
    return {"message": "Hello, World!"}

# MongoDB Model for storing face embeddings
class Face_embedding(Document):
    embedding = ListField(required=True)
    name = StringField()
    timestamp = DateTimeField(default=datetime.utcnow)

# Pydantic model for response
class FaceMatchResult(BaseModel):
    recognized_faces: list

# Store recognized faces dynamically


@app.post("/recognize_faces/")
async def recognize_faces():
    recognized_faces_history = []
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    ret, frame = cap.read()
    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame from webcam")

    # Release the webcam
    cap.release()

    # Save the frame temporarily to a file (DeepFace requires a file path)
    temp_image_path = "tempframe.jpg"
    cv2.imwrite(temp_image_path, frame)

    recognized_faces = []

    # Extract embedding for the captured face
    try:
        captured_embedding = DeepFace.represent(
            img_path=temp_image_path,
            model_name="VGG-Face",
            enforce_detection=False
        )[0]["embedding"]
    except Exception as e:
        os.remove(temp_image_path)
        raise HTTPException(status_code=500, detail=f"Error extracting face embedding: {e}")

    # Loop through each face embedding in the database
    for face in Face_embedding.objects:
        stored_embedding = np.array(face.embedding)

        # Compute similarity (cosine similarity)
        similarity = np.dot(captured_embedding, stored_embedding) / (np.linalg.norm(captured_embedding) * np.linalg.norm(stored_embedding))

        if similarity > 0.9:  # Adjust threshold as needed
            recognized_faces.append(face.name)

    if not recognized_faces:
        recognized_faces.append("Unknown")

    # Update dynamic history
    recognized_faces_history.clear() #clear the list before each API call.
    recognized_faces_history.extend(recognized_faces)

    # Delete the temporary image file
    os.remove(temp_image_path)

    return FaceMatchResult(recognized_faces=recognized_faces_history)

@app.post("/save_face/")
async def save_face(name: str):
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    ret, frame = cap.read()
    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame from webcam")

    # Release the webcam
    cap.release()

    # Save the frame temporarily to a file (DeepFace requires a file path)
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Use DeepFace to extract the face embedding
    try:
        embedding = DeepFace.represent(
            img_path=temp_image_path,
            model_name="VGG-Face",  # Use the Facenet model
            enforce_detection=False,  # Skip if no face is detected
        )

        if not embedding:
            raise HTTPException(status_code=400, detail="No faces found in the frame")

        # Extract the embedding list
        embedding_list = embedding[0]["embedding"]

        # Save to MongoDB
        try:
            face = Face_embedding(name=name, embedding=embedding_list)
            face.save()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving face to database: {e}")

        # Delete the temporary image file
        os.remove(temp_image_path)

        return {"message": "Face saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing face: {e}")