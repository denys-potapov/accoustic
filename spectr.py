import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('single.wav')
# right channel only
print(sample_rate, len(samples), samples.max(), samples.min())
samples = samples[:, 1]
print(samples)
frequencies, times, spectrogram = signal.spectrogram(
	samples, sample_rate, mode="magnitude")


plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
