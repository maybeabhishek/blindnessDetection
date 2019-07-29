import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


df_train = pd.read_csv('./input/aptos2019-blindness-detection/train.csv')
df_test = pd.read_csv('./input/aptos2019-blindness-detection/test.csv')

print(df_train.head())


def display_samples(df, columns=4, rows=4):
    fig=plt.figure(figsize=(5*columns, 4*rows))
    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'./input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        a = plt.imshow(img)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

display_samples(df_train)

