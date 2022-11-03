from random import sample
import scipy.signal
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
from scipy.io import wavfile
from matplotlib import pyplot as plt
from tqdm import tqdm


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# extract labels
df = pd.read_csv("../data/Data/features_30_sec.csv", usecols=["filename", "label"])
df = df.sort_values("filename")
print(df.head())
# one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df["label"])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded.shape)

cuts = [1, 0.9, 0.8, 0.7]

extended_labels = onehot_encoded.repeat(repeats=len(cuts), axis=0)

np.save("../data/processed/labels.npy", extended_labels)

# extract data
audio_dict = {}
from pathlib import Path
genre_folders = Path("../data/Data/genres_original").iterdir()
for genre_folder in sorted(genre_folders):
    print(genre_folder)
    files = Path(genre_folder).iterdir()
    for file in sorted(files):
        filename = file.as_posix().rsplit('/', 1)[1]
        try:
            samplerate, data = wavfile.read(file)
            audio_dict[filename] = (samplerate, data)

        except ValueError as err:
            print(file)
            print(err)


def create_spectogram(song_name, save_image=False, cut_ratio=1.0):
    samplerate, song = audio_dict[song_name]
    song = song[:int(len(song) * cut_ratio)]

    f, t, Sxx = scipy.signal.spectrogram(song, samplerate)
    fig = plt.figure(figsize=(200, 200), dpi=1)

    plt.pcolormesh(t, f, np.log1p(Sxx), shading='gouraud')
    plt.axis('off')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)

    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(width, height, 3)

    gray_image = rgb2gray(mplimage)
    if save_image:
        plt.savefig(f"../data/processed/{song_name}-{cut_ratio}.png")
    plt.close()
    return gray_image

images = []
for song in tqdm(audio_dict.keys()):
    for cut in cuts:
        image = create_spectogram(song, save_image=False, cut_ratio=cut)
        images.append(image)
images_array = np.stack(images, axis=0)
np.save("../data/processed/images2.npy", images_array)