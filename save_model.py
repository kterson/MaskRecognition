import time
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizer_v2.gradient_descent import SGD
from keras.preprocessing.image import ImageDataGenerator


# define cnn model
def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# run the test harness for evaluating a model
def main():
    model = define_model()
    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.mean = [123.68, 116.779, 103.939]
    train_it = datagen.flow_from_directory('dataset_final/',
                                           class_mode='binary', batch_size=64, target_size=(224, 224))
    model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
    model.save('final_model.h5')


startTime = time.time()
main()
print("Time: " + str(time.time() - startTime))
