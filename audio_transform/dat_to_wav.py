'''
File:   audio_transform/dat_to_wav.py

Spec:   Convert a raw audio .dat file to a .wav file.

Usage:  Call this script from system_control/transform_and_inference.py.
        Do not run this program directly.

I/O:    This program expects a .dat file as input and outputs a .wav file
'''

# TODO: read dat header and extract parameters like sample rate and size
import wave
import sys

def parse_dat_header(dat_file_path):
    """
    Parses the header of a .dat file to extract audio parameters.
    
    Args:
        dat_file_path (str): Path to the .dat file
    
    Returns:
        sample_rate, sample_width,    """
    
    with open(dat_file_path, 'rb') as dat_file:

        line = dat_file.readline().decode('utf-8').strip()

        while True:
            line = dat_file.readline(32).decode('utf-8').strip()
            print(line)
            if not line:
                break
            # check for the end of ascii header, which will a null char
            if line[0] == '\0':
                break

            # parse the line
            line = line.strip(";")
            line = line.replace(" ", "")
            #line = line.replace("'", "\"")
            [var, value] = line.split("=")
            if var == "sensor_id":
               sensor_id = value
            if var == "platform_id":
               platform_id = value
            if var == "location_id":
               location_id = value
            if var == "instrument_id":
               instrument_id = value
            if var == "second":
                second = float(value)
            if var == "time":
                time = value
            if var == "file_size":
                file_size = int(value)
            if var == "buffer_size":
                buffer_size = int(value)
            if var == "samples_per_buffer":
                samples_per_buffer = int(value)
            if var == "sampling_rate":
                sampling_rate = int(value)
            if var == "channels":
                channels = int(value)
            if var == "sample_size":
                sample_size = int(value)
            if var == "gain":
                gain = int(value)
            if var == "timestamp":
                timestamp = int(value)

        # number of adc buffer in the file            
        number_buffers = file_size * 512 / buffer_size

        dt = 1.0 / sampling_rate
        
        duration = samples_per_buffer * dt

    return sample_rate, sample_width

def convert_dat_to_wav(input_dat_file, output_directory, num_channels, sample_width, frame_rate):
    """
    Converts a raw audio .dat file to a .wav file.

    Args:
        input_dat_file (str): Path to the input .dat file containing raw audio data.
        output_directory (str): Directory where the output .wav file will be saved.
        num_channels (int): Number of audio channels (e.g., 1 for mono, 2 for stereo).
        sample_width (int): Sample width in bytes (e.g., 1 for 8-bit, 2 for 16-bit).
        frame_rate (int): Frame rate (samples per second, e.g., 44100 Hz).
    
    Returns:
        tuple: (success, message, output_wav_file_path)
    """
    try:
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate output WAV filename from input DAT filename
        dat_filename = os.path.basename(input_dat_file)
        wav_filename = os.path.splitext(dat_filename)[0] + '.wav'
        output_wav_file_path = os.path.join(output_directory, wav_filename)
        
        # Read the raw audio data from the .dat file
        with open(input_dat_file, 'rb') as dat_file:
            raw_audio_data = dat_file.read()

        # Open the .wav file in write mode
        with wave.open(output_wav_file_path, 'wb') as wav_file:
            # Set the WAV file parameters
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(frame_rate)

            # Write the raw audio data to the .wav file
            wav_file.writeframes(raw_audio_data)

        success_message = f"Successfully converted '{os.path.basename(input_dat_file)}' to '{wav_filename}'"
        return True, success_message, output_wav_file_path

    except FileNotFoundError:
        error_message = f"Input file '{input_dat_file}' not found"
        return False, error_message, None
    except Exception as e:
        error_message = f"Conversion error: {str(e)}"
        return False, error_message, None

def main():
    print("This program should not be run directly. Use system_control/transform_and_inference.py instead.")
    sys.exit(1)

if __name__ == "__main__":
    main()
