import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import random
import os

def visualize_data(path):
    names = []
    nums = []
    data = {'Name of class':[], 'Number of samples':[]}

    for i in os.listdir(path+'/train'):
        nums.append(len(os.listdir(path+'/train/'+i)))
        names.append(i)

    data['Name of class'] += names
    data['Number of samples'] += nums

    df = pd.DataFrame(data)
    sns.barplot(x=df['Name of class'], y=df['Number of samples'])
    plt.show()

    classes = os.listdir(path+'/train')
    plt.figure(figsize=(30, 30))
    for x in range(20):
        i = random.randint(0, len(classes) - 1)
        images = os.listdir(path+'/train'+'/'+classes[i])
        j = random.randint(0, len(images) - 1)
        image = cv2.imread(path+'/train'+'/'+classes[i]+'/'+images[j])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(5, 5, x + 1)
        plt.imshow(image)
        plt.title(classes[i])
        plt.axis("off")

    plt.show()

def prepare_data(path):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.8, 1.1)
    )

    train_ds = image_datagen.flow_from_directory(
        path+'/train',
        subset='training',
        target_size=(640, 640),
        batch_size=16
    )

    val_ds = image_datagen.flow_from_directory(
        path+'/train',
        subset='validation',
        target_size=(640, 640),
        batch_size=16
    )
    return train_ds, val_ds
