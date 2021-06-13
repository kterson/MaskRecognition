import cv2
import os
import numpy as np
import time


def prepareFaces(facesin):
    faces = []
    for i in range(len(facesin)):
        if len(faces) == 0:
            faces = facesin[i]
        else:
            if len(facesin[i]) > 0:
                faces = np.concatenate((faces, facesin[i]), axis=0)

    if len(faces) <= 1:
        return faces

    faces = faces[np.argsort(faces[:, 0])]  # sort by x

    i = 0
    while i < len(faces) - 1:  # delete duplicates and individuals
        if i + 1 < len(faces) and not similar(faces[i], faces[i + 1]):
            faces = np.delete(faces, i, axis=0)
            continue
        while i + 1 < len(faces) and similar(faces[i], faces[i + 1]):
            faces = np.delete(faces, i + 1, axis=0)
        i += 1
    return faces


def similar(r1, r2):
    if np.abs(r1[0] - r2[0]) < (r1[2] + r2[2]) / 8 and np.abs(r1[1] - r2[1]) < (r1[3] + r2[3]) / 8:
        return True
    else:
        return False


def main():
    faceCascade1 = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
    faceCascade2 = cv2.CascadeClassifier("xml/face.xml")
    faceCascade3 = cv2.CascadeClassifier("xml/face2.xml")
    faceCascade4 = cv2.CascadeClassifier("xml/face3.xml")
    faceCascade5 = cv2.CascadeClassifier("xml/face4.xml")
    faceCascade6 = cv2.CascadeClassifier("xml/face_cv2.xml")

    inputType = "negative"
    folder = "C:\\Users\\HP\\Desktop\\SCHOOL\\SEMESTR 4\\SI\\PRO\\pythonProject\\output\\pos"

    imgNo = 1

    startTime = time.time()

    for filename in os.listdir(folder):
        frame = cv2.imread(os.path.join(folder, filename))

        if frame is not None:
            print(imgNo)
            print(filename)

            height, width, channels = frame.shape
            if height > 1000:
                ratio = width / height
                height = 1000
                width = int(height * ratio)
            frame = cv2.resize(frame, (width, height))

            gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            facesIn = [faceCascade1.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)),
                       faceCascade2.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)),
                       faceCascade3.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)),
                       faceCascade4.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)),
                       faceCascade5.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)),
                       faceCascade6.detectMultiScale(gFrame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))]
            faces = prepareFaces(facesIn)

            if len(faces) <= 1:
                continue

            i = 0
            for (x, y, w, h) in faces:  # resize and save
                i += 1
                face = frame[y:y + h, x:x + w]
                cv2.imwrite("output/" + inputType + "/2_" + str(imgNo) + "_" + str(i) + ".png",
                            cv2.resize(face, (200, 200)))
            imgNo += 1

    print("Time: " + str(time.time() - startTime))


if __name__ == "__main__":
    main()
