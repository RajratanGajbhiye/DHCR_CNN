from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

train_dataGenerator = ImageDataGenerator(
		rotation_range = 5,
		width_shift_range = 0.1,
		height_shift_range = 0.1,
		rescale = 1.0/255,
		shear_range = 0.2,
		zoom_range = 0.2,		
		horizontal_flip = False,
		fill_mode = 'nearest')

test_datagenerator = ImageDataGenerator(rescale=1./255)

train_generator = train_dataGenerator.flow_from_directory(
			"DevanagariHandwrittenCharacterDataset/Train",
			target_size = (32,32),
			batch_size = 32,
			color_mode = "grayscale",
			class_mode = "categorical")


testing_generator = test_datagenerator.flow_from_directory(
			"DevanagariHandwrittenCharacterDataset/Test",
			target_size=(32,32),
			batch_size=32,
			color_mode = "grayscale",
			class_mode= 'categorical')


cnn_model = Sequential()

cnn_model.add(Conv2D(32, (3,3), strides = (1,1),activation = "relu",input_shape = (32,32,1)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2),padding="same"))

cnn_model.add(Conv2D(64, (3,3), strides = (1,1),activation = "relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2),padding="same"))

cnn_model.add(Conv2D(128, (3,3), strides = (1,1),activation = "relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2),padding="same"))

cnn_model.add(Flatten())

cnn_model.add(Dense(128, activation="relu", kernel_initializer="uniform"))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(64, activation="relu", kernel_initializer="uniform"))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(46, activation="softmax", kernel_initializer="uniform"))

cnn_model.compile(optimizer=Adam(lr=0.0001), loss="categorical_crossentropy", metrics = ["accuracy"])
		
print(cnn_model.summary())

result = cnn_model.fit(train_generator, epochs=20, validation_data=testing_generator)

training_accuracy = result.history['accuracy'][-1]
testing_accuracy = result.history['val_accuracy'][-1]

print(f'Training Accuracy: {training_accuracy}')
print(f'Testing Accuracy: {testing_accuracy}')

accuracy = result.history['accuracy']
test_acc = result.history['val_accuracy']
loss = result.history['loss']
test_loss = result.history['val_loss']

epochs_length = range(len(accuracy))

plt.plot(epochs_length, accuracy, 'r', label='Training Accuracy')
plt.plot(epochs_length, test_acc, 'g', label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(1, len(accuracy), 3))
plt.grid(True)
plt.legend()
plt.figure()


plt.plot(epochs_length, loss, 'r', label='Training Loss')
plt.plot(epochs_length, test_loss, 'g', label='Testing Loss')
plt.title('Training and Testing loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, len(loss), 3))
plt.grid(True)
plt.legend()
plt.show()


cnn_model.save("HindiCharacterRecognitionCNNModel.h5")