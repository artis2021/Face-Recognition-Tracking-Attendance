# -*- coding: utf-8 -*-
from deepface import DeepFace


# detector_backend = "opencv", "ssd", "dlib", "mtcnn", "retinaface"
# model_name = "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace"
# distance_metric = "cosine", "euclidean", "euclidean_l2"


#face recognition
face_recognition = DeepFace.find(img_path="dataset/testing/modi1.jpg",
                                    db_path="dataset/training",
                                    detector_backend="opencv",
                                    model_name="VGG-Face",
                                    distance_metric="cosine")

print(face_recognition)

