import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Input


# Step 1: Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Step 2: Preprocess the data
# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Step 3: Load the pre-trained VGG16 model without the top layer (i.e., the classifier)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))


# Freeze the layers of the VGG16 model so that they are not trained
for layer in base_model.layers:
    layer.trainable = False


# Step 4: Build the custom model using the pre-trained VGG16
# Add a global average pooling layer, followed by a fully connected layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
predictions = Dense(10, activation='softmax')(x)


# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)


# Step 5: Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# Step 6: Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))


# Step 7: Plot accuracy and loss vs epochs
plt.figure(figsize=(12, 6))


# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Step 8: Evaluate the model on a test image
# Let's pick a random image from the test set and see the prediction
import random
index = random.randint(0, x_test.shape[0] - 1)
test_image = x_test[index]
print(np.argmax(y_test[index]))
true_label = np.argmax(y_test[index])


# Predict the class of the image
prediction = model.predict(np.expand_dims(test_image, axis=0))
predicted_label = np.argmax(prediction)


# Display the test image and predicted label
plt.imshow(test_image)
plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
plt.show()





