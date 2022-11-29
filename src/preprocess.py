from random import sample
import scipy.signal
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
from scipy.io import wavfile
from matplotlib import pyplot as plt
from tqdm import tqdm

# keeping the first one each time if exact rep
EXCLUDE_LIST: list = [
    # disco exact repetitions (50,51,70)(55,60,89)(71,74)(98,99)
    "disco.00051.wav", "disco.00070.wav",
    "disco.00060.wav", "disco.00089.wav",
    "disco.00074.wav", "disco.00099.wav",
    # hiphop (39,45) (76,78)
    "hiphop.00045.wav", "hiphop.00078.wav"
    # jazz (33,51)(34,53)(35,55)(36,58)(37,60)(38,62)(39,65)(40,67)(42,68)(43,69)(44,70)(45,71)(46,72)
    "jazz.00051.wav", "jazz.00053.wav",
    "jazz.00055.wav", "jazz.00058.wav",
    "jazz.00060.wav", "jazz.00062.wav",
    "jazz.00065.wav", "jazz.00067.wav",
    "jazz.00068.wav", "jazz.00069.wav",
    "jazz.00070.wav", "jazz.00071.wav", "jazz.00072.wav", 
    # metal (04,13)(34,94)(40,61)(41,62)(42,63)(43,64)(44,65)(45,66)
    "metal.00013.wav", "metal.00094.wav",
    "metal.00061.wav", "metal.00062.wav",
    "metal.00063.wav", "metal.00064.wav",
    "metal.00065.wav", "metal.00066.wav",
    # pop (15,22)(30,31)(45,46)(47,80)(52,57)(54,60)(56,59)(67,71)(87,90)
    "pop.00022.wav", "pop.00031.wav",
    "pop.00046.wav", "pop.00080.wav",
    "pop.00057.wav", "pop.00060.wav",
    "pop.00059.wav", "pop.00071.wav",
    "pop.00090.wav",
    # reggae (03,54)(05,56)(08,57)(10,60)(13,58)(41,69)(73,74)(80,81,82)(75,91,92), 25s distortion jazz(86)
    "reggae.00054.wav", "reggae.00056.wav",
    "reggae.00057.wav", "reggae.00060.wav",
    "reggae.00058.wav", "reggae.00069.wav",
    "reggae.00074.wav", "reggae.00081.wav",
    "reggae.00082.wav", "reggae.00091.wav",
    "reggae.00092.wav", "reggae.00086.wav"
]
import librosa, librosa.display

def rgb2gray(rgb):
    """returns grey value for rgb pic"""
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b
    

def extract_labels():
    """extracts all the labels"""
    df = pd.read_csv("../data/Data/features_30_sec.csv", usecols=["filename", "label"])
    df = df.sort_values("filename")
    return df


def exclude_labels(df, exclude_values=[]):
    """lets skip corrupted labels, exact duplicits (within genre), """
    df = df.query('filename not in @exclude_values')
    df.sort_values("filename").reindex() 
    return df


df = extract_labels()
df = exclude_labels(df, EXCLUDE_LIST)


# one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df["label"])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded.shape)

def smooth_labels(labels, factor):
	labels *= 1 - factor
	labels += factor / labels.shape[1]
	return labels


smoothed_labels = smooth_labels(onehot_encoded, 0.1)

np.save("../data/processed/labels2.npy", smoothed_labels)

# extract data
audio_dict = {}
from pathlib import Path
genre_folders = Path("../data/Data/genres_original").iterdir()
for genre_folder in sorted(genre_folders):
    print(genre_folder)
    files = Path(genre_folder).iterdir()
    for file in sorted(files):
        filename = file.as_posix().rsplit('/', 1)[1]
        if (filename in EXCLUDE_LIST): continue
        try:
            samplerate, data = wavfile.read(file)
            audio_dict[filename] = (samplerate, data)

        except ValueError as err:
            print(file)
            print(err)


def create_spectogram(song_name, save_image=False):
    samplerate, song = audio_dict[song_name]
    song = np.array([float(i) for i in song])

    # this is the number of samples in a window per fft
    n_fft = 2048# The amount of samples we are shifting after each fft
    hop_length = 512

    # f, t, Sxx = scipy.signal.spectrogram(song, samplerate)
    # fig = plt.figure(figsize=(200, 200), dpi=1)

    mel_signal = librosa.feature.melspectrogram(y=song, sr=samplerate, hop_length=hop_length, 
        n_fft=n_fft)

    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    fig = plt.figure(figsize=(100, 100), dpi=1)
    librosa.display.specshow(power_to_db, sr=samplerate, cmap="gray", 
    hop_length=hop_length)

    # plt.pcolormesh(t, f, np.log1p(Sxx), shading='gouraud')
    plt.axis('off')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)

    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(width, height, 3)

    gray_image = rgb2gray(mplimage)
    if save_image:
        filename = f"../data/processed/{song_name}.png"
        plt.savefig(filename)
    plt.close()
    return gray_image

images = []
for song in tqdm(audio_dict.keys()):
    image = create_spectogram(song, save_image=False)
    images.append(image)
images_array = np.stack(images, axis=0)
np.save("../data/processed/images2.npy", images_array)