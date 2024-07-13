import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential
from keras.layers import Conv2D, Dropout

def build_model():
    model_Xception = tf.keras.applications.Xception(input_shape=(640, 640, 3),
                                                    include_top=False,
                                                    weights='imagenet')
    model_Xception.trainable = False

    model = Sequential([
        model_Xception,
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        Dropout(0.5),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        Dropout(0.5),
        Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
