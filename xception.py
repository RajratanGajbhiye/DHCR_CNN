from tensorflow.keras.applications import Xception
from tensorflow.keras import models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
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
			target_size = (71,71),
			batch_size = 64,
			color_mode = "rgb",
			class_mode = "categorical")


testing_generator = test_datagenerator.flow_from_directory(
			"DevanagariHandwrittenCharacterDataset/Test",
			target_size=(71,71),
			batch_size=64,
			color_mode = "rgb",
			class_mode= 'categorical')


pretrained_model = Xception(include_top=False, weights='imagenet', input_shape=(71, 71, 3))


for layer in pretrained_model.layers:
    layer.trainable = False


xception_model = models.Sequential()
xception_model.add(pretrained_model)
xception_model.add(Flatten())
xception_model.add(Dense(512, activation='relu'))
xception_model.add(BatchNormalization())
xception_model.add(Dropout(0.5))

xception_model.add(Dense(train_generator.num_classes, activation='softmax'))

xception_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

xception_model.summary()

result = xception_model.fit(train_generator, epochs=15, validation_data=testing_generator)

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

xception_model.save("HindiCharacterRecognitionXceptionModel.h5")