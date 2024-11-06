import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

path = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'

data = pd.read_csv(path, header=None)



data.head()


data.info()




features = data.drop(140, axis=1)
target = data[140]

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2
)

train_index = y_train[y_train == 1].index

train_data = x_train.loc[train_index]

 [markdown]
# 


min_max_scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = min_max_scaler.fit_transform(train_data.copy())

x_test_scaled = min_max_scaler.transform(x_test.copy())



class AutoEncoder(Model):
    def __init__(self, output_units, ldim=8):
        super().__init__()
        self.encoder = Sequential([
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(ldim, activation='relu')
        ])
        self.decoder = Sequential([
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded



model = AutoEncoder(output_units=x_train_scaled.shape[1])

model.compile(loss='msle', metrics=['mse'], optimizer='adam')

history = model.fit(x_train_scaled, x_train_scaled, epochs=20, batch_size=512, validation_data=(x_test_scaled, x_test_scaled), shuffle=True)



plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()


def find_threshold(model, x_train_scaled):
    recons = model.predict(x_train_scaled)
    recons_error = tf.keras.metrics.msle(recons, x_train_scaled)
    threshold = np.mean(recons_error.numpy()) + np.std(recons_error.numpy())
    return threshold

def get_predictions(model, x_test_scaled, threshold):
    predictions = model.predict(x_test_scaled)
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds

threshold = find_threshold(model, x_train_scaled)
print(f"Threshold: {threshold}")


predictions = get_predictions(model, x_test_scaled, threshold)
accuracy = accuracy_score(predictions, y_test)

print(f"Accuracy Score: {accuracy}")






