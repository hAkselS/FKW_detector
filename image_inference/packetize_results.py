'''
File:   image_inference/packetize_results.py

Spec:   Parse existing inference outputs and remove all information
        that is non essential for the ground team. 
        In other words, remove everything except the time and detection count. 

ID:     pr 

Usage:  Call this from another script!

Up Next: Implement a function that filters detections by their bounding box size. 
'''

import pandas as pd 
import os 
from datetime import datetime, timedelta

def packetize_inference_outputs(input_csv):
    # Find the name of the existing file 
    base_name = os.path.basename(input_csv)
    # Remove 'detection' from the base name, add 'packetized'
    new_name = base_name.replace('detection', 'packetized')
    
    # Create the output directory inside data_products/packets
    output_dir = os.path.join('data_products', 'packets')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, new_name)

    try:
        # Read the existing CSV file
        df = pd.read_csv(input_csv)

        results = []
        accounted_times = set()
        total_detections = 0
        for row in df.itertuples():
            class_name = row.class_name
            start_time = row.start_time
            # Use the file name to know if it is a 0001 (+0 seconds) or 0011 (+30 seconds)
            file_path = row.file_path
            filename = os.path.basename(file_path)
            suffix = filename.split('-')[-1].replace('.jpg', '')      
            # Add 30 seconds to start time if file name ends with '0011' 
            if suffix == '0011':  
                start_time = dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                start_time = (dt + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
                start_time = str(start_time)
            
            detections = row.number_of_detections
            if class_name == 'whistle' and start_time not in accounted_times:
                accounted_times.add(start_time)
                results.append({'start_time': start_time, 'detections': detections})
                total_detections += detections

        print(f"\npr: total detections {total_detections}")

        # Save the packetized results
        packetized_df = pd.DataFrame(results)
        packetized_df.to_csv(output_csv, index=False)
        
        message = f"Packetized {total_detections} positive detections to {output_csv}"
        # print(message)
        return True, message, output_csv

    except Exception as e:
        message = f"Failed to packetize results: {str(e)}"
        print(message)
        return False, message, ""
    

def main():
   print(f"ti: PLEASE DO NOT RUN THIS SCRIPT BY ITSELF, RUN <python3 sys_control/process_control.py> INSTEAD, EXITING...")
   import sys
   sys.exit(1)

if __name__ == "__main__":
    main()
    
