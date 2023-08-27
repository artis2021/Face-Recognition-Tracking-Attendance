# -*- coding: utf-8 -*-
from deepface import DeepFace


# detector_backend = "opencv", "ssd", "dlib", "mtcnn", "retinaface"
# model_name = "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace"
# distance_metric = "cosine", "euclidean", "euclidean_l2"


#face verification
face_verified = DeepFace.verify(img1_path="dataset/testing/modi1.jpg",
                                    img2_path="dataset/testing/joe1.jpg",
                                    detector_backend="opencv",
                                    model_name="VGG-Face",
                                    distance_metric="cosine")

print(face_verified)

