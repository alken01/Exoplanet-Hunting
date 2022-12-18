import sys

sys.path.append('..')
import numpy as np
from scipy.io import loadmat
from os.path import join
from matplotlib import pyplot

import os
from pickle import FALSE
import random
from time import time
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.decomposition import FastICA

from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import stft
from scipy.signal import welch
from scipy.signal import istft
from playsound import playsound
import wave


# Specify the directory that contains the audio files
directory = '/Users/barci/recordings/'

# Change the current working directory to the specified directory
os.chdir(directory)

# Get a list of all the audio files in the directory
audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

# Print the names of the audio files
print("The audio files in the directory are:")
for f in audio_files:
    print(f)

# Set the sampling rate to 44.1 kHz
sampling_rate = 44100


time_domain = []
# Loop through each audio file
for f in audio_files:
    # Construct the full path to the file
    file_path = os.path.join(directory, f)

    # Read the audio file
    data, sample_rate = sf.read(file_path)

    # Isolate the first 10 seconds of the audio file
    data = data[:20*sampling_rate]
    
    time_domain.append(data)
    print("Length of data frame: " + str(len(data)))
    

time_domain = time_domain - np.mean(time_domain)

time_domain = np.array(time_domain)
print("Shape of time domain is:" + str(time_domain.shape))
# Perform PCA on the array
pca = PCA(n_components=4)
pca.fit(time_domain.T)
# Transform the array using PCA
whitened_matrix = pca.transform(time_domain.T)

whitened_matrix = np.array(whitened_matrix)   
# Print the transformed array
print("The transoformed matrixes from PCA have this shape: " + str(whitened_matrix.shape))

ica = FastICA(max_iter=1000,whiten=FALSE,n_components=4)

# Fit the model to the data
ica.fit(time_domain.T)

# Separate the data into the independent components
sources = ica.transform(time_domain.T)
sources = sources.T

print("Sources have this shape: " + str(sources.shape))
source1_time = sources[0,:]
source2_time = sources[1,:]
source3_time = sources[2,:]
source4_time = sources[3,:]

print(len(source1_time))



sf.write("test1.wav",source1_time,sampling_rate)
sf.write("test2.wav",source2_time,sampling_rate)
sf.write("test3.wav",source3_time,sampling_rate)
sf.write("test4.wav",source4_time,sampling_rate)