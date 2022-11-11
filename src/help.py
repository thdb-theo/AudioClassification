from matplotlib import pyplot as plt
import random
import pandas as pd

df = pd.read_csv("../data/Data/features_30_sec.csv", usecols=["label"])
genres = sorted(df["label"].unique())


def label_to_genre(label):
    ind = label.index(max(label))

    print(genres[ind])
    return ind

def show_image_from_each_class(imgs, n_genres=10):

    fig, axs = plt.subplots(2, (n_genres+1)//2)
    
    df = pd.read_csv("../data/Data/features_30_sec.csv", usecols=["label"])
    axs = [ax for row in axs for ax in row]
    for i in range(n_genres):
        ax = axs[i]
        ax.axis("off")
        ax.set_title(genres[i])
        rand_idx = random.randint(0, 99)
        ax.imshow(imgs[100*i+rand_idx], cmap="gray")
    plt.show()

    # plt.figure(figsize=(22, 8))
    # for i in range(10):
    #     ax = plt.subplot(1, 10, i+1)
    #     plt.gray()
    #     plt.imshow(imgs[rand_idx])
    #     plt.title(genres[i])
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()


