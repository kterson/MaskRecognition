import os

import cv2
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# klasyfikatory do szukania twarzy
import prepare_dataset


# load image dla sprawdzania maski
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# funkcja do sprawdzania czy prostokąty są podobnej wielkości i w podobnym miejscu
def similar(r1, r2):
    return np.abs(r1[0] - r2[0]) < (r1[2] + r2[2]) / 8 and np.abs(r1[1] - r2[1]) < (r1[3] + r2[3]) / 8


# main
def testCrowd():
    faceCascade1 = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
    faceCascade2 = cv2.CascadeClassifier("xml/face.xml")
    faceCascade3 = cv2.CascadeClassifier("xml/face2.xml")
    faceCascade4 = cv2.CascadeClassifier("xml/face3.xml")
    faceCascade5 = cv2.CascadeClassifier("xml/face4.xml")
    faceCascade6 = cv2.CascadeClassifier("xml/face_cv2.xml")
    faceCasc1 = cv2.CascadeClassifier("xml/gitcv_frontalface_default.xml")
    faceCasc2 = cv2.CascadeClassifier("xml/gitcv_frontalface_alt.xml")
    faceCasc3 = cv2.CascadeClassifier("xml/gitcv_frontalface_alt2.xml")
    faceCasc4 = cv2.CascadeClassifier("xml/gitcv_profileface.xml")
    folder = "C:\\Users\\HP\\Desktop\\SCHOOL\\SEMESTR 4\\SI\\PRO\\pythonProject\\demo"  # folder ze zdjęciami tłumów
    model = load_model('final_model.h5')  # model sprawdzania maski

    for filename in os.listdir(folder):

        frame = cv2.imread(os.path.join(folder, filename))

        if frame is not None:

            gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            facesIn = [
                faceCasc1.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)),
                faceCasc2.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)),
                faceCasc3.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)),
                faceCasc4.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)),
                # faceCascade5.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            ]  # wszystkie twarze znalezione przez wszystkie klasyfikatory
            # scaleFactor - ile razy ma się zmniejszać filtr przeszukujący zdjęcie za każdą iteracją
            # minNeighbors - nie jestem pewien co robi, ale można się bawić tym parametrem
            # jak najwięcej twarzy i jak najmniej nie-twarzy
            faces = prepare_dataset.prepareFaces(facesIn)

            if len(faces) <= 1:
                continue

            frame2 = frame  # obraz, po którym będziemy rysować prostokąty
            for (x, y, w, h) in faces:
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (128, 128, 128), 1)  # wynik niepewny na szaro
                face = frame[y:y + h, x:x + w]  # wyodrębniamy twarz
                cv2.imwrite("temp/temp.png", cv2.resize(face, (200, 200)))  # tymczasowo zapisujemy
                img = load_image("temp/temp.png")
                result = model.predict(img)
                # result - 1elementowa tablica zawierająca wartość od 0 do 1 (0 - maska, 1 - brak maski)
                if np.abs(result[0] - 0) < 0.001:  # kolorki i podpis
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame2, "mask", (x, y + h + 10), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN,
                                color=(0, 0, 0), thickness=2)
                    cv2.putText(frame2, "mask", (x, y + h + 10), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN,
                                color=(255, 255, 255), thickness=1)
                if np.abs(result[0] - 1) < 0.001:
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(frame2, "no mask", (x, y + h + 10), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN,
                                color=(0, 0, 0), thickness=2)
                    cv2.putText(frame2, "no mask", (x, y + h + 10), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN,
                                color=(255, 255, 255), thickness=1)

            cv2.imshow("siema", frame2)
            cv2.waitKey()


# start programu
testCrowd()
