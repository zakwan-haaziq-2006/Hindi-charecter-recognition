from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(128,128,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Flatten())


model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))



model.add(Dense(36,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True,width_shift_range = 1,height_shift_range = 2)

val_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True)

training_set = train_datagen.flow_from_directory("Dataset/DevanagariHandwrittenCharacterDataset/Train",target_size = (128,128),batch_size = 10,class_mode = 'categorical')

label = training_set.class_indices
print(label)
val_set = val_datagen.flow_from_directory("Dataset/DevanagariHandwrittenCharacterDataset/Test",target_size = (128,128),batch_size = 10,class_mode = 'categorical')
label2 = val_set.class_indices
print(label2)
red_lr = ReduceLROnPlateau(monitor="val_loss",patience=5,verbose=1,min_lr=0.2,factor=0.3)
call_backs = [EarlyStopping(monitor='val_loss',patience=10,),ModelCheckpoint(filepath='model.weights.h5',monitor='val_loss',verbose=1,save_best_only=True),red_lr]


history = model.fit(training_set,steps_per_epoch=100,epochs=20,validation_data=val_set,validation_steps=100,callbacks=call_backs)
import matplotlib.pyplot as plt
plt.figure(0)
plt.plot(history.history['accuracy'],label = 'Accuracy')
plt.plot(history.history['val_accuracy'],label = 'Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

# model_json = model.to_json()
# with open("model.json",'w') as json:
#     json.write(model_json)
# model.save_weights("model.weights.h5")
# print("Model saved to disk")