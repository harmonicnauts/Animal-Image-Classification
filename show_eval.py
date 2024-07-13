import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def evaluate_model(model, val_ds):
    evaluation = model.evaluate(val_ds)
    print(f"Loss: {evaluation[0]}, Accuracy: {evaluation[1]}")
    return evaluation

def plot_accuracies(evaluation):
    plt.figure()
    plt.bar(['Accuracy'], [evaluation[1]])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.show()

def predict_and_evaluate(model, sub_csv, path_test):
    df_sub = pd.read_csv(sub_csv)
    image_id = df_sub['ID']

    labels = []
    for i in image_id:
        image = keras.utils.load_img(os.path.join(path_test, f"{i}.jpg"), target_size=(640, 640))
        img = np.array(image) / 255.0
        img = img.reshape(1, 640, 640, 3)
        label = model.predict(img)
        label_id = label[0].tolist()
        labels.append(label_id.index(max(label_id)))

    df_sub['Label'] = labels
    df_sub.to_csv('submission_file.csv', index=False)

    return df_sub

def prepare_validation_data(path, image_size=(640, 640), batch_size=16):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
    
    val_ds = image_datagen.flow_from_directory(
        os.path.join(path, 'train'),
        subset='validation',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return val_ds

def main():
    model_path = "Model.h5"
    sub_csv = './Sample_submission.csv'
    path_test = './ofa-ai-mastery-computer-vision/test/test'
    data_path = './ofa-ai-mastery-computer-vision'

    # Load the model
    model = load_model(model_path)

    # Prepare validation data
    val_ds = prepare_validation_data(data_path)

    # Evaluate the model
    evaluation = evaluate_model(model, val_ds)

    # Plot accuracies
    plot_accuracies(evaluation)

    # Predict and evaluate on test data
    df_sub = predict_and_evaluate(model, sub_csv, path_test)
    print(df_sub.head(10))

if __name__ == "__main__":
    main()