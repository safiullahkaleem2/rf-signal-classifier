import numpy as np
import matplotlib.pyplot as plt
import os

def create_spectrogram(signal, filename):
    plt.specgram(signal, NFFT=256, Fs=10000, noverlap=128)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_all():
    os.makedirs("data/spectrograms", exist_ok=True)
    for file in os.listdir("data/signals"):
        signal = np.load(f"data/signals/{file}")
        label = file.split("_")[0]
        save_path = f"data/spectrograms/{label}_{file.split('_')[1].replace('.npy', '.png')}"
        create_spectrogram(signal, save_path)

if __name__ == "__main__":
    process_all()
