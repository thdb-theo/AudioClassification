from matplotlib import pyplot as plt
import random
import pandas as pd
import numpy as np

df = pd.read_csv("../data/Data/features_30_sec.csv", usecols=["label"])
genres = sorted(df["label"].unique())


def label_to_genre(label):
    ind = label.index(max(label))

    print(genres[ind])
    return ind

def show_image_from_each_class(imgs, genre_indices, n_genres=10):

    fig, axs = plt.subplots(2, (n_genres+1)//2)
    
    df = pd.read_csv("../data/Data/features_30_sec.csv", usecols=["label"])
    axs = [ax for row in axs for ax in row]
    for i in range(n_genres):
        ax = axs[i]
        ax.axis("off")
        ax.set_title(genres[i])
    
        prev_genre_index = 0 if i == 0 else genre_indices[i-1]
        n_imgs = genre_indices[i] - prev_genre_index
        rand_idx = random.randint(0, n_imgs)
    
        ax.imshow(imgs[prev_genre_index+rand_idx], cmap="gray")
    plt.show()



def find_genre_indices(labels):
    indices = []
    for i, label in enumerate(labels):
        if i == 0:
            continue
        if np.any(labels[i-1] != label):
            indices.append(i)
    indices.append(len(labels))
    return np.array(indices)

