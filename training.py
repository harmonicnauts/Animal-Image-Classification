# import tensorflow as tf

# def train_model(model, train_ds, val_ds):
#     save_best = tf.keras.callbacks.ModelCheckpoint("Model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
#     history = model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=[save_best], shuffle=True)
#     return history


import tensorflow as tf
import os

def train_model(model, train_ds, val_ds, log_dir):
    # Create a TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Create a ModelCheckpoint callback
    save_best = tf.keras.callbacks.ModelCheckpoint("Model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    
    # Train the model with the TensorBoard callback
    history = model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=[tensorboard_callback, save_best], shuffle=True)
    
    return history
