from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import  Activation, Dropout, Flatten, Dense, BatchNormalization
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
			target_size = (75,75),
			batch_size = 64,
			color_mode = "rgb",
			class_mode = "categorical")


testing_generator = test_datagenerator.flow_from_directory(
			"DevanagariHandwrittenCharacterDataset/Test",
			target_size=(75,75),
			batch_size=64,
			color_mode = "rgb",
			class_mode= 'categorical')

pretrained_model = InceptionV3(input_shape=(75, 75, 3), include_top=False, weights='imagenet')

for layer in pretrained_model.layers:
    layer.trainable = False

inception_model = Sequential()
inception_model.add(pretrained_model)
inception_model.add(Flatten())
inception_model.add(Dense(512, activation='relu'))
inception_model.add(BatchNormalization())
inception_model.add(Dropout(0.5))

inception_model.add(Dense(train_generator.num_classes, activation='softmax'))

inception_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print(inception_model.summary())

result = inception_model.fit(train_generator, epochs=20, validation_data=testing_generator)

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

inception_model.save("HindiCharacterRecognitionInseprtionV3Model.h5")