import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img

def prepare_submission(model, sub_csv, path_test):
    df_sub = pd.read_csv(sub_csv)
    image_id = df_sub['ID']

    labels = []
    for i in image_id:
        image = load_img(path_test+'/'+str(i)+'.jpg', target_size=(640, 640))
        img = np.array(image) / 255.0
        img = img.reshape(1, 640, 640, 3)
        label = model.predict(img)
        label_id = label[0].tolist()
        labels.append(label_id.index(max(label_id)))

    df_sub['Label'] = labels
    df_sub.to_csv('submission_file.csv', index=False)
