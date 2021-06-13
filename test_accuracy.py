import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


def accuracy(subfolder, model, pos):
    mask = 0
    noMask = 0
    uncertain = 0
    i = 0
    for filename in os.listdir(subfolder):
        print(i)
        i += 1
        img = load_image(os.path.join(subfolder, filename))

        result = model.predict(img)
        if np.abs(result[0] - 0) < 0.001:
            mask += 1
        elif np.abs(result[0] - 1) < 0.001:
            noMask += 1
        else:
            uncertain += 1

    if pos:
        print("positive:")
        print("hits: " + str(mask))
        print("losses: " + str(noMask))
    else:
        print("negative:")
        print("hits: " + str(noMask))
        print("losses: " + str(mask))
    print("misses: " + str(uncertain))
    print("total: " + str(mask + noMask + uncertain))
    if pos:
        print("accuracy: " + str(mask / (mask + noMask + uncertain)))
    else:
        print("accuracy: " + str(noMask / (mask + noMask + uncertain)))


# load an image and predict the class
def main():
    # load the image
    folder = "C:\\Users\\HP\\Desktop\\SCHOOL\\SEMESTR 4\\SI\\PRO\\pythonProject\\output\\"
    model = load_model('final_model.h5')

    subfolder = os.path.join(folder, "pos")
    accuracy(subfolder, model, True)

    subfolder = os.path.join(folder, "neg")
    accuracy(subfolder, model, False)


# entry point, run the example
main()
