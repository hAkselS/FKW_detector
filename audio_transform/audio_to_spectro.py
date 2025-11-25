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

ID:     as 

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
        
        # Validate minimum length (at least 31 seconds required)
        length = data.shape[0] / sample_rate
        if length < 31:
            return False, f"Audio too short ({length:.1f}s). Minimum 31 seconds required.", []
        
        # Calculate required samples for 60 seconds total
        target_length_samples = int(sample_rate * 60)  # 60 seconds worth of samples
        current_length_samples = data.shape[0]
        
        # Zero-pad if necessary to reach 60 seconds
        if current_length_samples < target_length_samples:
            padding_needed = target_length_samples - current_length_samples
            padding_seconds = padding_needed / sample_rate

            print(f"as: Padding audio: adding {padding_seconds:.3f} seconds to reach 60 seconds total")
            # Add noise to the end of the audio data (instead of constant values)
            # Generate pink/brown noise in the frequency range of interest
            noise_samples = padding_needed
            
            # Create filtered noise in your frequency range (3500-9500 Hz)
            noise = np.random.normal(0, 1, noise_samples)
            
            # Apply simple bandpass by mixing frequencies
            t_noise = np.arange(noise_samples) / sample_rate
            freq_center = (freq_min + freq_max) / 2  # ~6500 Hz
            noise_filtered = noise * np.sin(2 * np.pi * freq_center * t_noise)
            
            # Scale to reasonable amplitude
            noise_filtered = noise_filtered * (np.std(data) * 0.1)  # 10% of original signal strength
            
            data = np.concatenate([data, noise_filtered.astype(data.dtype)])
    

        # If longer than 60 seconds, truncate to exactly 60 seconds
        elif current_length_samples > target_length_samples:
            truncate_seconds = (current_length_samples - target_length_samples) / sample_rate
            print(f"as: Truncating audio: removing {truncate_seconds:.1f} seconds to fit 60 seconds total")
            data = data[:target_length_samples]
        
        # Now we have exactly 60 seconds of data
        # Determine the number of whole 3 second chunks (should be exactly 20)
        samples_per_chunk = int(sample_rate * chunk_duration)
        num_chunks = int(len(data) / samples_per_chunk)  # Should be 20

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
        
        return True, f"Successfully processed {audio_file_name}: generated {len(output_files)} spectrograms: Added {padding_seconds:.3f} of padding.", output_files
        

        
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