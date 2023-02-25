import numpy as np
import librosa
import librosa.display as dsp
import matplotlib.pyplot as plt


data1, sample_rate = librosa.load('sample2.wav')

print('Total number of samples: ',data1.shape[0])
print('Sample rate: ',sample_rate)
print('Lenngth of file in seconds: ',librosa.get_duration(data1))

d = librosa.stft(data1)
D = librosa.amplitude_to_db(np.abs(d),ref=np.max)
fig,ax = plt.subplots(2,1,sharex=True,figsize=(10,10))

dsp.waveshow(data1, sr=sample_rate, alpha=0.6, ax=ax[0])
# img = dsp.specshow(D, y_axis='linear', x_axis='s',sr=sample_rate,ax=ax[0])
ax[0].set(title='Audio waveform')
ax[0].label_outer()

img = dsp.specshow(D, y_axis='linear',x_axis='s',sr=sample_rate,ax=ax[1])
ax[1].set(title='Log frequency power spectrogram')
ax[1].label_outer()
fig.colorbar(img, ax=ax, format='%+2.f dB')

plt.show()