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
# TODO: make sure that this script can handle audio files shorter than 1 minute (by up to 6 seconds)

import matplotlib.pyplot as plt
import os
from scipy.signal import spectrogram, get_window
import numpy as np
import sys 

# Try to import soundfile (better 24-bit support), fallback to scipy
try:
    import soundfile as sf
    USE_SOUNDFILE = True
except ImportError:
    from scipy.io import wavfile
    USE_SOUNDFILE = False 

###################################################################
# CONFIGURATION DEFAULTS
chunk_duration = 3              # Number of seconds represented in each pane of the spectrogram
freq_min = 3500                 # Spectrogram strip's minimum sampled frequency 
freq_max = 9500                 # Spectrogram strip's maximum sampled frequency 
plot_min = 4000                 # Spectrogram strip's minumum DISPLAYED frequency
plot_max = 9000                 # Spectrogram strip's maximum DISPLAYED frequency
###################################################################

def read_dat_file(dat_file_path, sample_width, num_channels):
    """
    Read raw audio data from a .dat file.
    
    Args:
        dat_file_path (str): Path to the .dat file
        sample_width (int): Sample width in bytes (1=8bit, 2=16bit, 3=24bit, 4=32bit)
        num_channels (int): Number of audio channels
    
    Returns:
        numpy.ndarray: Audio data
    """
    # Determine the numpy data type based on sample width
    if sample_width == 1:
        dtype = np.int8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 3:
        # 24-bit audio - read as bytes and convert to int32
        with open(dat_file_path, 'rb') as f:
            raw_data = f.read()
        
        # Convert 24-bit data to 32-bit integers
        # Each sample is 3 bytes, so we need to process in groups of 3
        samples = []
        for i in range(0, len(raw_data), 3 * num_channels):
            for ch in range(num_channels):
                if i + ch * 3 + 2 < len(raw_data):
                    # Read 3 bytes and convert to signed 24-bit integer
                    byte1 = raw_data[i + ch * 3]
                    byte2 = raw_data[i + ch * 3 + 1] 
                    byte3 = raw_data[i + ch * 3 + 2]
                    
                    # Combine bytes (little-endian) and sign-extend
                    value = byte1 | (byte2 << 8) | (byte3 << 16)
                    if value >= 0x800000:  # Sign bit set
                        value -= 0x1000000  # Convert to negative
                    
                    samples.append(value)
        
        data = np.array(samples, dtype=np.int32)
        
        # Reshape for multiple channels
        if num_channels > 1:
            data = data.reshape(-1, num_channels)
        
        return data
        
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    
    # For non-24-bit data, use numpy's fromfile
    data = np.fromfile(dat_file_path, dtype=dtype)
    
    # Reshape for multiple channels
    if num_channels > 1:
        data = data.reshape(-1, num_channels)
    
    return data

def process_audio_to_spectrograms(audio_file_path, output_directory, channel=5, 
                                 sample_rate=199000, sample_width=3, num_channels=1):
    """
    Process a single audio file (.wav or .dat) and generate two spectrogram images.
    
    Args:
        audio_file_path (str): Path to the audio file (.wav or .dat)
        output_directory (str): Output directory for images
        channel (int): Audio channel to process (default: 5)
        sample_rate (int): Sample rate for .dat files (default: 199000)
        sample_width (int): Sample width in bytes for .dat files (default: 3 for 24-bit)
        num_channels (int): Number of channels for .dat files (default: 1)
    
    Returns:
        tuple: (boolean, message, output_files)
    """
    try:
        # Get audio file name (works for both .wav and .dat)
        audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        file_extension = os.path.splitext(audio_file_path)[1].lower()
        
        # Read audio file based on extension
        if file_extension == '.dat':
            # Read .dat file directly
            data = read_dat_file(audio_file_path, sample_width, num_channels)
            # Use provided sample rate for .dat files
            actual_sample_rate = sample_rate
        elif file_extension == '.wav':
            # Read .wav file
            try:
                if USE_SOUNDFILE:
                    # Use soundfile for better 24-bit support
                    data, actual_sample_rate = sf.read(audio_file_path)
                    # Convert to int if needed (soundfile returns float by default)
                    if data.dtype == np.float64 or data.dtype == np.float32:
                        data = (data * 32767).astype(np.int16)  # Convert to 16-bit int
                else:
                    # Fallback to scipy with warning suppression
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*Reached EOF prematurely.*")
                        actual_sample_rate, data = wavfile.read(audio_file_path)
            except Exception as e:
                return False, f"Failed to read WAV file: {str(e)}", []
        else:
            return False, f"Unsupported file type: {file_extension}. Supported: .wav, .dat", []
        
        # Select a channel if multiple 
        if len(data.shape) > 1:
            if channel >= data.shape[1]:
                return False, f"Channel {channel} not available. File has {data.shape[1]} channels", []
            data = data[:, channel]
        
        # Validate length
        length = data.shape[0] / actual_sample_rate
        if not (58 < length < 62):
            return False, f"Length not ~60 second ({length:.1f}s), undefined behavior", []
        
        # Determine the number of whole 3 second chunks
        samples_per_chunk = int(actual_sample_rate * chunk_duration)
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
                all_chunks, audio_file_name, actual_sample_rate, 
                output_directory, num_rows=10, which_plot=which_plot
            )
            output_files.append(output_file)
        
        return True, f"Successfully processed {audio_file_name}: generated {len(output_files)} spectrograms from {num_chunks} audio chunks | sample rate: {actual_sample_rate}", output_files
        
    except Exception as e:
        return False, f"Error processing {audio_file_path}: {str(e)}", []

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