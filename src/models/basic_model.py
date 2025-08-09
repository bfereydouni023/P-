from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),

            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(64, activation='relu'),
            Dense(categories_count, activation='softmax')
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )