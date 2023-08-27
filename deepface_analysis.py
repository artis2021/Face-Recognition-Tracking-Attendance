# -*- coding: utf-8 -*-
from deepface import DeepFace


#face analysis
face_analysis = DeepFace.analyze(img_path="dataset/testing/modi1.jpg",
                                    actions=['emotion','age','gender','race'])

print(face_analysis)

