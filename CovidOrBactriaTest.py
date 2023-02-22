from keras.layers import Activation
from keras.models import Sequential
from  keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')




TestBatches=ImageDataGenerator().flow_from_directory('T3/Coronahack-Chest-XRay-Dataset/Test',target_size=(224,224),classes=['Virus','Normal','bacteria'],batch_size=10)
TriantBatches=ImageDataGenerator().flow_from_directory('T3/Coronahack-Chest-XRay-Dataset/Train',target_size=(224,224),classes=['Virus','Normal','bacteria'],batch_size=10)
ValBatches=ImageDataGenerator().flow_from_directory('T3/Coronahack-Chest-XRay-Dataset/Validate',target_size=(224,224),batch_size=10)




Model=Sequential()


for ly in VGG19().layers[:-1]:
    Model.add(ly)

for ly in Model.layers:
    ly.trainable=False



Model.add(Dense(3,activation='softmax'))

Model.compile(Adam(0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
Model.fit_generator(TriantBatches,steps_per_epoch=TriantBatches.samples/10,epochs=30,validation_data=TestBatches,validation_steps=10,verbose=2)


Model.save('NormalOrBOrVCovid19.h5')

print(TestBatches.class_indices)
print(Model.predict_generator(ValBatches,steps=10))