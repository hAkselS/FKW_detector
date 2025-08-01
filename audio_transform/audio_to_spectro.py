'''
File:   audio_to_spectro.py

Spec:   Audio to spectro ingests one minute of audio data and transforms it into two spectrograms. 
        Each image contains ten, three second spectrogram strips separated by a small 
        black space. Images are roughly square for optimal performance with YOLO. 
        Images are not saved in gray scale format for YOLO training purposes. 

I/O:    This program expects one minute audio inputs. 
        This program outputs spetrograms images containing ten spectrogram strips.
        Spectrograms do not overlap each other.
        This program currently can ONLY ingest 1 minute audio inputs. 

Usage:  Do not run this program directly. Call from system_control/transform_and_inference.py.
'''

import matplotlib.pyplot as plt
import os
from scipy.signal import spectrogram, get_window
from scipy.io import wavfile
import numpy as np
import sys 

###################################################################
# CONFIGURATION DEFAULTS
chunk_duration = 3              # Number of seconds represented in each pane of the spectrogram
freq_min = 3500                 # Spectrogram strip's minimum sampled frequency 
freq_max = 9500                 # Spectrogram strip's maximum sampled frequency 
plot_min = 4000                 # Spectrogram strip's minumum DISPLAYED frequency
plot_max = 9000                 # Spectrogram strip's maximum DISPLAYED frequency
###################################################################

def process_audio_to_spectrograms(wave_file_path, output_directory, channel=5):
    """
    Process a single audio file and generate two spectrogram images.
    
    Args:
        wave_file_path (str): Path to the wave file
        output_directory (str): Output directory for images
        channel (int): Audio channel to process (default: 5)
    
    Returns:
        tuple: (boolean, output_files)
    """
    try:
        # Get audio file name
        audio_file_name = os.path.basename(wave_file_path)[:-4]
        
        # Read audio file
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Reached EOF prematurely.*")
                sample_rate, data = wavfile.read(wave_file_path)
        except ValueError:
            return False, "Invalid input file type. Supported file type(s): .wav", []
        
        # Select a channel if multiple 
        if len(data.shape) > 1:
            if channel >= data.shape[1]:
                return False, f"Channel {channel} not available. File has {data.shape[1]} channels", []
            data = data[:, channel]
        
        # Validate length
        length = data.shape[0] / sample_rate
        if not (58 < length < 62):
            return False, f"Length not ~60 second ({length:.1f}s), undefined behavior", []
        
        # Determine the number of whole 3 second chunks
        samples_per_chunk = int(sample_rate * chunk_duration)
        num_chunks = int(len(data) / samples_per_chunk)
        
        # Create chunks
        all_chunks = [] 
        for i in range(num_chunks):
            start_sample = i * samples_per_chunk
            end_sample = start_sample + samples_per_chunk
            chunk_data = data[start_sample:end_sample]
            all_chunks.append(chunk_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate two spectrograms
        output_files = []
        for which_plot in range(2):
            output_file = _make_spectro(
                all_chunks, audio_file_name, sample_rate, 
                output_directory, num_rows=10, which_plot=which_plot
            )
            output_files.append(output_file)
        
        return True, output_files
        
    except Exception as e:
        return False, f"Error processing {wave_file_path}: {str(e)}", []

def _make_spectro(all_chunks, audio_file_name, sample_rate, output_directory, num_rows=10, which_plot=0):
    """Create a single spectrogram image."""
    fig, axes = plt.subplots(
        nrows=num_rows, 
        ncols=1, figsize=(8, 5),
        facecolor='black',
        gridspec_kw={'hspace': -0.5},
        constrained_layout=True
    )
    
    fig.patch.set_facecolor('black')

    for i in range(num_rows):
        # Compute spectrogram
        fft_size = 1024
        hop_size = fft_size // 2
        window = get_window("hann", fft_size)

        # 10 spectros to a plot, if 2nd spectro grab 10 - 19
        f, t, Sxx = spectrogram(
            all_chunks[i + which_plot*10], 
            fs=sample_rate, 
            window=window, 
            nperseg=fft_size, 
            scaling='density'
        )

        fmin, fmax = freq_min, freq_max
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        f = f[freq_slice]
        Sxx = Sxx[freq_slice, :][0]

        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Plot
        ax = axes[i]
        pcm = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap=plt.cm.binary)
        ax.set_ylim(plot_min, plot_max)
        ax.axis('off')

    base_name = audio_file_name + '-' + str("{:04}".format(which_plot*10 + 1)) 
    image_name = os.path.join(output_directory, f"{base_name}.jpg")
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    return image_name

def main():
    print("This program should not be run directly. Use system_control/transform_and_inference.py instead.")
    sys.exit(1)

if __name__ == "__main__":
    main()