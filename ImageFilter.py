import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

hdf5_path = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/hdf5data/' +'basic' +'.hdf5'
if not os.path.isfile:
    print('ERROR: HDF5-file not found! Run create_hdf5_dataset.py first!')
hdf5_file = h5py.File(hdf5_path, "r")

feature = hdf5_file['val_features'][0, ...]
sample_excerpt = feature[:,1400:1515]
# sample_excerpt=np.flip(sample_excerpt,0)
# sample_excerpt=np.flip(sample_excerpt,1)
# print(sample_excerpt.shape)
# imgplot = plt.imshow(sample_excerpt)

sample_excerpt[:,:]=sample_excerpt[:,:]+0.2746671993100236
sample_excerpt[:,:]=sample_excerpt[:,:]/6.762975251946389
# sample_excerpt[:,0]
# # test image is normalised between 0 and 1
imgplot = plt.imshow(sample_excerpt)

row_start_slice=4
row_finish_slice=30
excerpt_filter_region=sample_excerpt[row_start_slice:row_finish_slice]
excerpt_filtered_region=gaussian_filter(excerpt_filter_region, sigma=2)
sample_excerpt[row_start_slice:row_finish_slice]=excerpt_filtered_region

# convert to image
formatted = (sample_excerpt * 255 / np.max(sample_excerpt)).astype('uint8')
ford = Image.fromarray(formatted)
# make image 3 times bigger
sizer=3
ford_resized = ford.resize((115*sizer, 80*sizer))
plt.imshow(ford_resized)
plt.show()