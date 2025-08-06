'''
File:   audio_transform/dat_to_wav.py

Spec:   Convert a raw audio .dat file to a .wav file.

Usage:  Call this script from system_control/transform_and_inference.py.
        Do not run this program directly.

I/O:    This program expects a .dat file as input and outputs a .wav file
'''

# TODO: If sample size is 2, safeyway convert to sample size = 2 (scipy wave can only handle 16 bit sample width)
import wave
import sys
import os 

def convert_dat_to_wav(input_dat_file, output_directory):
    """
    Converts a raw audio .dat file to a .wav file by extracting header parameters.

    Args:
        input_dat_file (str): Path to the input .dat file containing raw audio data.
        output_directory (str): Directory where the output .wav file will be saved.

    Returns:
        tuple: (success, message, output_wav_file_path)
    """
    try:
        # Step 1: Parse header
        sampling_rate = None
        channels = None
        sample_size = None
        file_size = None

        with open(input_dat_file, 'rb') as dat_file:
            # Read line-by-line until header ends
            while True:
                pos = dat_file.tell()
                line = dat_file.readline()
                if not line or b'\0' in line:
                    # End of header
                    break
                line = line.decode('utf-8').strip().replace(" ", "").strip(";")
                if '=' not in line:
                    continue
                key, value = line.split('=')
                if key == "sampling_rate":
                    sampling_rate = int(value)
                elif key == "channels":
                    channels = int(value)
                elif key == "sample_size":
                    sample_size = int(value)  # in bytes
                elif key == "file_size":
                    file_size = int(value)  # in blocks (e.g., 512-byte)
            
            if channels == None:
                channels = 1 # Default to mono if not specified
            # DEBUG
            print(f"Header values: sampling_rate={sampling_rate}, channels={channels}, sample_size={sample_size}, file_size={file_size}")

            # Validate header values
            if None in (sampling_rate, channels, sample_size, file_size):
                raise ValueError("Missing required parameters in header.")

            # Step 2: Read binary audio data after header
            dat_file.seek(pos)  # go back to where binary data begins
            raw_audio_data = dat_file.read()

        # Step 3: Set output path
        os.makedirs(output_directory, exist_ok=True)
        dat_filename = os.path.basename(input_dat_file)
        wav_filename = os.path.splitext(dat_filename)[0] + '.wav'
        output_wav_file_path = os.path.join(output_directory, wav_filename)

        # Step 4: Write to WAV file
        with wave.open(output_wav_file_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_size) # In bytes
            wav_file.setframerate(sampling_rate)
            wav_file.writeframes(raw_audio_data)

        success_message = f"Successfully converted '{dat_filename}' to '{wav_filename}'"
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
