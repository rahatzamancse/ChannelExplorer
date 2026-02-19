from nltk.corpus import wordnet as wn
import tensorflow as tf
import tensorflow_datasets as tfds

ds, info = tfds.load(
    'imagenette/320px-v2',
    shuffle_files=False,
    with_info=True,
    as_supervised=True,
    batch_size=None,
)
labels = list(map(lambda l: wn.synset_from_pos_and_offset(l[0], int(l[1:])).name(), info.features['label'].names))

# Now save the dataset so we can load it in pytorch folder structure
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

# Create the folder structure
if os.path.exists('pytorch-data'):
    shutil.rmtree('pytorch-data')
os.makedirs('pytorch-data')
os.makedirs('pytorch-data/train')
os.makedirs('pytorch-data/test')

splits = ds.keys()

for label in labels:
    for split in splits:
        os.makedirs(f'pytorch-data/{split}/{label}')
    
with open('pytorch-data/labels.txt', 'w') as f:
    f.write('\n'.join(labels))
    
for split in ['train', 'validation']:
    for i, (image, label) in tqdm(enumerate(ds[split])):
        label_name = labels[label.numpy()]
        plt.imsave(f'pytorch-data/{split}/{label_name}/{i}.jpg', image.numpy())