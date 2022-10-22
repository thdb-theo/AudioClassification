import numpy as np
from matplotlib import pyplot as plt


images = np.load("../data/processed/images.npy")
print(images.shape)

plt.imshow(images[0], cmap="gray")
plt.show()
