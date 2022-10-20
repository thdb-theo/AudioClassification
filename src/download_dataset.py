import tensorflow as tf

import tensorflow_datasets as tfds

# Doesn't work unfortunately, because the source website is down :(
ds = tfds.load('gtzan', split='train', shuffle_files=True)
