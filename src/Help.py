from matplotlib import pyplot as plt
import random

def LabelToGenre(label):
    ind = label.index(max(label))
    Map = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock"
    }
    print(Map[ind])
    return ind

def ShowImageFromEachClass(imgs):
    plt.figure(figsize=(22, 8))
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    for i in range(10):
        ax = plt.subplot(1, 10, i+1)
        rand_idx = random.randint(0+i*100,99+i*100)
        plt.gray()
        plt.imshow(imgs[rand_idx])
        plt.title(genres[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


