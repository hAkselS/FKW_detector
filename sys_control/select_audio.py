'''
File:   sys_control/select_audio.py

Spec:   Maintain CSV file with time ranges for each analyzed
        ascent / descent. Create a CSV file for each ascent / descent
        with a list of audio files to process and success flags.

Super Spec:   List files in the ascent / descent directories.
        Convert file names to times. 
        Save files names nad times into a json 
        Determine if there are any new files to process.
        If there are new files, determine their time ranges.
        Add all of the files to a CSV who's title reflects the time range. 
        Select images for sampling. Set true a "selected for sampling" flag.
        Pass the CSV file to transform and inference.
        Finish, return success. 

Notes:  !!! This script uses main and does not receive any arguments !!!
'''

import yaml 
import os 
import csv
import json
from datetime import datetime
from collections import OrderedDict


###################################################################
# CONFIGURATION DEFAULTS
config_file = 'config.yaml'  # Path to your configuration file
mapping_file = 'logs/analyst_logs/file_mapping.json'  # Path to the file mapping JSON
directory_date_format = "%y%m%d" # How date directories are named

###################################################################

def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the configuration file
    
    Returns:
        dict: Configuration parameters
    
    Raises:
        SystemExit: If config file cannot be loaded
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        if config is None:
            print(f"Error: Configuration file '{config_file}' is empty or invalid")
            return {}
            
        return config
    
    except Exception as e:
        print(f"Select Audio Error: Unexpected error loading '{config_file}': {str(e)}")
        return {}

def load_file_mapping(mapping_file):
    """Load existing file-to-time mapping from JSON."""
    try:
        with open(mapping_file, 'r') as f:
            data = json.load(f)
        
        # Convert string timestamps back to datetime objects
        file_time_map = OrderedDict()
        for timestamp_str, file_path in data.items():
            timestamp = datetime.fromisoformat(timestamp_str)
            file_time_map[timestamp] = file_path
        
        print(f"Loaded {len(file_time_map)} files from existing mapping")
        return file_time_map
        
    except FileNotFoundError:
        print("No existing mapping found, creating new one")
        return OrderedDict()
    except Exception as e:
        print(f"Error loading mapping: {e}")
        return OrderedDict()

def save_file_mapping(file_time_map, mapping_file):
    """Save file-to-time mapping to JSON."""
    try:
        # Convert datetime objects to ISO format strings for JSON
        data = OrderedDict()
        for timestamp, file_path in file_time_map.items():
            data[timestamp.isoformat()] = file_path
        
        with open(mapping_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(file_time_map)} files to mapping")
        return True
    except Exception as e:
        print(f"Error saving mapping: {e}")
        return False

def update_file_mapping(directory, mapping_file='file_mapping.json'):
    """Update existing mapping with any new files found."""
    # Load existing mapping
    existing_mapping = load_file_mapping(mapping_file)
    existing_files = set(existing_mapping.values())
    
    # Scan for new files only
    new_files_found = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.dat'):
                file_path = os.path.join(root, filename)
                
                # Skip if we already have this file
                if file_path in existing_files:
                    continue
                
                # Parse new file
                if filename.startswith("WISPR_"):
                    try:
                        parts = filename[6:-4]
                        if '_' in parts:
                            date_part, time_part = parts.split('_')
                            datetime_str = date_part + time_part
                            file_datetime = datetime.strptime(datetime_str, "%y%m%d%H%M%S")
                            
                            existing_mapping[file_datetime] = file_path
                            new_files_found += 1
                    except (ValueError, IndexError):
                        continue
    
    # Re-sort the mapping
    sorted_mapping = OrderedDict(sorted(existing_mapping.items()))
    
    # Save updated mapping
    save_file_mapping(sorted_mapping, mapping_file)
    
    print(f"Added {new_files_found} new files to mapping")
    return sorted_mapping


def main():
    config = load_config(config_file)
    if not config:
        print("Failed to load configuration.")
        return

    # Continue with the main logic using the loaded config
    print("Configuration loaded successfully.")

    base_audio_directory = config['base_audio_directory']
    num_files_to_analyze = config['num_files_to_analyze']

    # DEBUG 
    print(f"\nBase audio directory: {base_audio_directory}")
    print(f"Number of files to analyze: {num_files_to_analyze}")

    # Update mapping with any new files (fast for incremental updates)
    time_mapping = update_file_mapping(base_audio_directory, mapping_file)
    
    if not time_mapping:
        print("No valid audio files found.")
        return
    
    

if __name__ == "__main__":
    main()