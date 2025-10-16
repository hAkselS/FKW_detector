'''
File:   sys_control/select_audio.py

Spec:   Maintain JSON file with time ranges for each analyzed
        ascent / descent. Create a CSV file for each ascent / descent
        with a list of audio files to process and success flags.

Super Spec: This script uses update_file_mapping to look at all
        the past audio files, stored in observed_audio_and_times.json, that
        have been previously analyzed or counted and compares the list of files 
        to the contents of the active directory to determine which file are new and therefor unanalyzed.
        New files are added to a list, sorted then added to the JSON in sorted order.
        This allows the JSON file to remain sorted which provides faster and more efficient searching. 
        The list of new files ALSO goes into a CSV file who's title is the time
        range of the new files. Once in the CSV, a subset of the new
        files will be selected for sampling by setting the 'selected_for_sampling'
        flag to True in the CSV. The CSV is returned by main for use in Transform and Inference. 
        Transform and Inference uses the CSV to determine which files to analyze. 

Notes:  !!! This script uses main and does not receive any arguments !!!
        This script adds new files to the existing list of files that have been analyzed on
        the assumption that the new files all represent times AFTER the last existing file. 
    
ID:     sa 
'''
import yaml 
import os 
import sys 
import pandas as pd
import json
from datetime import datetime
from collections import OrderedDict
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

###################################################################
# CONFIGURATION DEFAULTS
config_file = project_root + '/config/config.yaml'  # Path to your configuration file
csv_directory = project_root + '/logs/dive_logs'  # Directory where CSV files will be saved
time_mapping_file = project_root + '/logs/analyst_logs/observed_audio_and_times.json'  # Path to the file mapping JSON
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
            print(f"sa: Error: Configuration file '{config_file}' is empty or invalid")
            return {}
            
        return config
    
    except Exception as e:
        print(f"sa: Select Audio Error: Unexpected error loading '{config_file}': {str(e)}")
        return {}
    
def create_dive_csv(csv_directory, csv_name):
    """
    Create a CSV file for tracking dive analysis progress using pandas.
    
    Args:
        csv_directory (str): Directory where the CSV file will be created
        csv_name (str): Name of the CSV file (with or without .csv extension)
    
    Returns:
        tuple: (success, message, csv_file_path)
    """
    
    try:
        # Ensure directory exists
        os.makedirs(csv_directory, exist_ok=True)
        
        # Create full path
        csv_file_path = os.path.join(csv_directory, csv_name)
        
        # Define CSV columns with proper data types
        columns = [
            'file_name',           # string
            'start_time',          # datetime/string
            'selected_for_sampling', # boolean
            'dat_to_wave',         # boolean
            'wave_to_spectro',     # boolean
            'image_analyzed'       # boolean
        ]
        
        # Create empty DataFrame with specified columns
        df = pd.DataFrame(columns=columns)
        
        # Set appropriate data types
        df = df.astype({
            'file_name': 'string',
            'selected_for_sampling': 'boolean',
            'dat_to_wave': 'boolean', 
            'wave_to_spectro': 'boolean',
            'image_analyzed': 'boolean'
        })
        
        # Save to CSV
        df.to_csv(csv_file_path, index=False)
        
        return True, f"Successfully created CSV file: {csv_name}", csv_file_path
    
    except Exception as e:
        return False, f"Error creating CSV file: {str(e)}", None    

def populate_dive_csv(csv_directory, csv_name, sorted_new_files):
    """
    Populate an existing dive CSV file with new file entries.

    Args:
        csv_directory (str): Directory where the CSV file is stored.
        csv_name (str): Name of the CSV file.
        sorted_new_files (list): List of tuples (datetime, file_path) for new files.

    Returns:
        tuple: (success, message)
    """
    try:
        csv_path = os.path.join(csv_directory, csv_name)

        # Check if CSV exists
        if not os.path.exists(csv_path):
            return False, f"CSV file '{csv_name}' does not exist in {csv_directory}."

        # Load existing DataFrame
        df = pd.read_csv(csv_path)

        # Prepare new rows
        new_rows = []
        for file_datetime, file_path in sorted_new_files:
            new_rows.append({
                "file_name": file_path,
                "start_time": file_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "selected_for_sampling": False,
                "dat_to_wave": False,
                "wave_to_spectro": False,
                "image_analyzed": False
            })

        # Append new rows
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Save updated CSV
        df.to_csv(csv_path, index=False)

        return True, f"Successfully added {len(new_rows)} new files to {csv_name}."

    except Exception as e:
        return False, f"Error populating CSV: {str(e)}"

def quick_sort_check_memory(file_time_map):
    """
    Check if the in-memory file mapping is properly sorted by timestamps.
    Checks: 1) First 5 entries are sorted, 2) Last 5 entries are sorted, 
            3) All first 5 are smaller than all last 5 entries
    
    Args:
        file_time_map (OrderedDict): Dictionary with datetime keys and file paths as values
    
    Returns:
        bool: True if sorted, False otherwise
    """
    try:
        if len(file_time_map) <= 1:
            return True  # Empty or single item is always sorted
        
        # Get list of datetime objects (keys)
        timestamps = list(file_time_map.keys())
        n = len(timestamps)
        
        
        # For larger mappings, check first 5, last 5, and cross-check
        first_5 = timestamps[:5]
        last_5 = timestamps[-5:]
        
        # Check if first 5 entries are sorted
        for i in range(1, len(first_5)):
            if first_5[i-1] > first_5[i]:
                print(f"sa: First 5 entries not sorted: {first_5[i-1]} > {first_5[i]}")
                return False
        
        # Check if last 5 entries are sorted
        for i in range(1, len(last_5)):
            if last_5[i-1] > last_5[i]:
                print(f"sa: Last 5 entries not sorted: {last_5[i-1]} > {last_5[i]}")
                return False
        
        # Check if ALL first 5 entries are smaller than ALL last 5 entries
        # This ensures proper ordering between beginning and end of the mapping
        max_first_5 = max(first_5)
        min_last_5 = min(last_5)
        
        if max_first_5 >= min_last_5:
            print(f"sa: Ordering violation: max of first 5 ({max_first_5}) >= min of last 5 ({min_last_5})")
            return False
        
        print(f"\nsa: Quick sort check passed: first 5 sorted, last 5 sorted, proper ordering ({n} entries)")
        return True
            
    except Exception as e:
        print(f"Error in memory sort check: {e}")
        return False

def load_file_mapping(time_mapping_file):
    """Load existing file-to-time mapping from JSON."""
    try:
        with open(time_mapping_file, 'r') as f:
            data = json.load(f)
        
        # Convert string timestamps back to datetime objects
        file_time_map = OrderedDict()
        for timestamp_str, file_path in data.items():
            timestamp = datetime.fromisoformat(timestamp_str)
            file_time_map[timestamp] = file_path
        
        print(f"sa: Loaded {len(file_time_map)} files from existing mapping in logs/analyst_logs json file")
        return file_time_map
        
    except FileNotFoundError:
        print("sa: No existing mapping found, creating new one")
        return OrderedDict()
    except Exception as e:
        print(f"sa: Error loading mapping: {e}")
        return OrderedDict()

def save_file_mapping(file_time_map, time_mapping_file):
    """Save file-to-time mapping to JSON."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(time_mapping_file), exist_ok=True)
        
        # Convert datetime objects to ISO format strings for JSON
        data = OrderedDict()
        for timestamp, file_path in file_time_map.items():
            data[timestamp.isoformat()] = file_path
        
        with open(time_mapping_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nsa: [{len(file_time_map)}] total files saved in logs/analyst_logs json")
        return True
    except Exception as e:
        print(f"sa: Error saving mapping: {e}")
        return False

def update_file_mapping(directory, time_mapping_file='observed_audio_and_times.json'):
    """
    Update existing mapping with any new files found.

    Args:
        directory (str): The directory to scan for new files.
        time_mapping_file (str): The path to the time mapping file.

    Calls:
        create_dive_csv: To create a new CSV file for the new audio files.
        populate_dive_csv: To populate the created CSV file with the new audio files.
        quick_sort_check: To perform a quick sort check on the new audio files.
    
    Returns:
        str: The path to the created CSV file, or None if no new files were found.
    """
    # Load existing mapping (already sorted)
    existing_mapping = load_file_mapping(time_mapping_file)
    existing_files = set(existing_mapping.values())
    
    # Collect new files
    new_files = {}
    new_files_found = 0

    for root, dirs, files in os.walk(directory):
        # DEBUG
        # print(f"Scanning: {root}")
        # print(f"Files: {files}")
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
                            
                            new_files[file_datetime] = file_path
                            new_files_found += 1
                    except (ValueError, IndexError):
                        continue
    
    # If no new files, return existing mapping unchanged
    if new_files_found == 0:
        print("sa: Update File Mapping: No new files found")
        return None # Check this 
    
    # Sort only the new files and append to existing mapping
    sorted_new_files = sorted(new_files.items())
    
    # Create CSV with date range name for new files
    if sorted_new_files:
        ''' Call create_dive_csv and populate_dive_csv '''
        # Get first and last timestamps from sorted new files
        first_timestamp = sorted_new_files[0][0]  # datetime object
        last_timestamp = sorted_new_files[-1][0]  # datetime object
        
        # Format timestamps for CSV filename (YYMMDD_HHMMSS format)
        first_str = first_timestamp.strftime("%y%m%d_%H%M%S")
        last_str = last_timestamp.strftime("%y%m%d_%H%M%S")
        
        # Create CSV name: first_datetime-last_datetime.csv
        csv_name = f"{first_str}-{last_str}.csv"
        
        # Create the CSV
        creation_success, creation_message, csv_path = create_dive_csv(csv_directory, csv_name)

        if creation_success:
            print(f"\nsa: ✓ Created CSV for new files: {csv_name}")
            print(f"  sa: Time range: {first_timestamp} to {last_timestamp}")
            print(f"  sa: CSV path: {csv_path}")
            # Populate the CSV with new files
            populate_success, population_message = populate_dive_csv(csv_directory, csv_name, sorted_new_files)

            if populate_success:
                print(f"\nsa:✓ Successfully populated CSV with new files: {csv_name}")
            else:
                print(f"sa: ✗ Failed to populate CSV: {population_message}")

        else:
            print(f"\nsa: ✗ Failed to create CSV: {creation_message}\n")
    

    # Append new files to existing mapping (no need to resort existing)
    for timestamp, file_path in sorted_new_files:
        existing_mapping[timestamp] = file_path

    # Do a quick sorting check
    if not quick_sort_check_memory(existing_mapping):
        print("sa: Warning: In-memory mapping is not sorted!")
        # Sort here if issue
        existing_mapping = OrderedDict(sorted(existing_mapping.items()))
    else:
        print("sa: First 5 and last 5 entries are sorted")

    # Save updated mapping
    save_file_mapping(existing_mapping, time_mapping_file)

    print(f"sa: Added [{new_files_found}] new files to logs/analyst_logs json")
    return csv_path

def select_files_for_sampling(csv_path, num_files_to_analyze):

    df = pd.read_csv(csv_path)
    num_files = len(df) 

    if num_files == 0:
        print(f"\nsa: ✗ No files found for sampling")
        return False, "No files found for sampling", 0

    if num_files <= num_files_to_analyze:
        # Sample everything if the number of files is less than the target
        df['selected_for_sampling'] = True
        # Save the file 
        df.to_csv(csv_path, index=False)
        print(f"\nsa: ✓ Selected files [{num_files}] for sampling (num files < num files to analyze)")
        return True, f"Selected all {num_files} files for sampling", {num_files}    

    else: 
        # Sample evenly across the dataset
        df['selected_for_sampling'] = False # Ensure no Trues when starting  
        step = max(1, num_files // num_files_to_analyze)
        # DEBUG
        # print(f'\nnum files to analyze = {num_files_to_analyze}')
        # print(f'num files = {num_files}')
        # print(f'step = {step}')

        files_analyzed = 0 
        for i in range(0, num_files, step): 
            if i < num_files and files_analyzed < num_files_to_analyze:
                df.iloc[i, df.columns.get_loc('selected_for_sampling')] = True
                files_analyzed += 1 

        df.to_csv(csv_path, index=False)
        print(f"\nsa: ✓ Selected files for sampling (num files > num files to analyze)")
        return True, f"Selected files for sampling", df['selected_for_sampling'].sum()

def main():
    '''
    Main function for select_audio.py
    
    Args: 
        None

    Returns: 
        tuple: (success, message, csv_path)
        - success (bool): True if successful, False if failed
        - message (str): Status message describing what happened
        - csv_path (str): Path to the created CSV file, or empty string if failed
    '''

    config = load_config(config_file)
    if not config:
        return False, "Failed to load configuration.", ''

    # Continue with the main logic using the loaded config
    # DEBUG 
    # print("\nsa: Configuration loaded successfully.")

    base_audio_directory = config['base_audio_directory']
    num_files_to_analyze = config['num_files_to_analyze']

    # DEBUG 
    print(f"\nsa: Base audio directory: {base_audio_directory}")
    print(f"sa: Maximum number of files to analyze: {num_files_to_analyze}\n")

    # Update mapping with any new files (fast for incremental updates)
    csv_path = update_file_mapping(base_audio_directory, time_mapping_file)

    if not csv_path:
        return False, "No new audio files found.", ''

    select_status, select_message, num_selected = select_files_for_sampling(csv_path, num_files_to_analyze)
    
    if not select_status:
        return False, f"File selection failed: {select_message}", csv_path
    
    success_message = f"Successfully processed audio files. Selected {num_selected} files for sampling from CSV: {csv_path}"
    return True, success_message, csv_path # CSV path is used in transform and inference! 

if __name__ == "__main__":
    main()