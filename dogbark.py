import os
import soundfile as sf
import librosa
import librosa.display
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
'''
n_ftt refers to the number of Fast Fourier Transform windows when converting a time-domain signal into its frequency-domain representation. The higher or lower this number depends on the type of analysis we are performing.
Fast Fourier Transform- an algorithm that computes the Discrete Fourier Transform (DFT) of a sequence, or its inverse (IDFT). Fourier analysis converts a signal from its original domain (often time or space) to a representation in the frequency domain. 
Highest n_ftt: 2048-4096
Used with high quality sampling with complex overlapping noise that is distinct and unique. Barking in a noisey street, mid conversation, music on at the same time... etc. 
Higher n_ftt: 1024-2048
This is to be used with complex recordings that have several unique sounds or layers, but not as much complexity as the highest. Larger n_ftt values can provide higher resolution analysis and account for more unique frequencies
Lower n_ftt: 256-1024 
This is to be used with clean recordings with limited frequency differences(no background noise or music). It is useful for tracking rapid changes in audio 
Hop Length- the number of samples the analysis window moves during stft computations. Larger hop_lengths result in less detailed overviews of audio, but smaller lengths can be difficult to calculate and require decently lengthed audio files to get useful data.
A common hop length in music is n_ftt// 4 so that 75% of windows overlap with the next window. Much smaller values would increase computation time or give unreadable/unhelpful data.
STFT- Short Time Fourier Transform is a Fourier Transform used to determine frequency and phase content over a length of time. The time signal is divided into shorter equal lengths to determine change in phase and frequency over time. 
'''
# Map labels to integers
LABEL_MAP = {
    "barking": 1,
    "non_barking": 0
}

# Base directories
BASE_DIR = "data"
AUDIO_DIRS = {
    "barking": os.path.join(BASE_DIR, "barking"),
    "non_barking": os.path.join(BASE_DIR, "non_barking")
} 
SPECTROGRAM_DIR = os.path.join(BASE_DIR, "spectrograms")
TFRECORD_FILE = os.path.join(BASE_DIR, "DogBarking.tfrecord")
# Ensure spectrogram directory exists
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
def add_noise(signal, noise_level=0.005):
    #Add Gaussian noise to the signal.

    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise
def spectrogram_plot(signal, sample_rate, output_path, n_fft=4096, hop_size=None, target_size=(512, 512)):
    
    #Generate a spectrogram from an audio signal.
    
    if hop_size is None:
        hop_size = n_fft // 16
    
    # Compute the STFT and convert to dB
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_size)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Resize spectrogram to target size
    resized_spectrogram = resize(spectrogram, target_size, mode='constant', anti_aliasing=True)

    # Save spectrogram as .npy
    npy_path = output_path.replace('.png', '.npy')
    np.save(npy_path, resized_spectrogram)

    # Plot and save as .png
    plt.figure(figsize=(15, 8))
    librosa.display.specshow(
        resized_spectrogram, sr=sample_rate, hop_length=hop_size,
        x_axis='time', y_axis='log', cmap='inferno'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return resized_spectrogram, npy_path

def process_audio_file(audio_file, label, output_dir, augment_with_noise=True, noise_level=0.005):
    try:
        signal, sample_rate = sf.read(audio_file)
        if signal.ndim > 1:  # Handle multi-channel audio
            signal = np.mean(signal, axis=1)  # Convert to mono

        if augment_with_noise:
            signal = add_noise(signal, noise_level=noise_level)

        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        output_image_path = os.path.join(label_dir, f"{base_name}_spectrogram.png")

        # Generate and save spectrogram
        spectrogram, spectrogram_path = spectrogram_plot(
            signal, sample_rate, output_image_path, n_fft=4096, hop_size=256, target_size=(512, 512)
        )

        print(f"Processed: {audio_file} -> Label: {label}")
        print(f"Spectrogram saved at: {spectrogram_path}")

    except Exception as e:
        print(f"Failed to process {audio_file}: {e}")



def main():
    for label_name, dir_path in AUDIO_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"Audio directory '{dir_path}' does not exist. Skipping...")
            continue
        
        label_spectrogram_dir = os.path.join(SPECTROGRAM_DIR, label_name)
        os.makedirs(label_spectrogram_dir, exist_ok=True)
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    # Process original file
                    process_audio_file(file_path, label_name, SPECTROGRAM_DIR)
                    
                    # Augment clean barking files with noise
                    if label_name == "barking":
                        process_audio_file(
                            file_path, label_name, SPECTROGRAM_DIR,
                            augment_with_noise=True, noise_level=0.01
                        )

if __name__ == '__main__':
    main()
