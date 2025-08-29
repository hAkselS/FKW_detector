'''
File:   detection/inference.py

Spec:   Perform inference on spectrogram images to detect FKWs using YOLO.

Usage:  Do not run this program directly. Call from system_control/transform_and_inference.py.

I/O:    This program expects one or more spectrogram images as inputs. 
        This program outputs inference results and a success boolean. 
'''

from ultralytics import YOLO
import os
import sys
import pandas as pd 
from datetime import datetime



###################################################################
# CONFIGURATION DEFAULTS
default_confidence_threshold = 0.25                           # Minimum confidence for detections
# Project root 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

###################################################################

def perform_inference(input_files, model_path, confidence_threshold, output_file_name_and_path):
    """
    Perform YOLO inference on a list of image files.
    
    Args:
        input_files (list): List of paths to image files
        model_path (str): Path to the YOLO model file
        results_file_name_and_path (str): Path to the results CSV file
        TODO: confidence (float): Confidence threshold for detections
    
    Returns:
        tuple: (success, message, output directory)
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
        
        # Check if all input files exist
        for file_path in input_files:
            if not os.path.exists(file_path):
                return False, f"Input file not found: {file_path}"
        
        # Load YOLO model
        try:
            model = YOLO(model_path, verbose=False)  # Disable verbose output
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
        
        total_detections = 0
        
        for file_path in input_files:
            try:
                # Run inference (verbose=False to suppress output)
                results = model(file_path, conf=confidence_threshold, verbose=False)
                
                # Extract results for this file
                file_results = {
                    'file_path': file_path,
                    'detections': [],
                    'detection_count': 0
                }
                
                # Process detections
                for result in results: 
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            detection = {
                                'confidence': float(box.conf.cpu().numpy()[0]),
                                'class_id': int(box.cls.cpu().numpy()[0]),
                                'class_name': result.names[int(box.cls.cpu().numpy()[0])],
                                'bbox': box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]
                            }
                            file_results['detections'].append(detection)
                
                file_results['detection_count'] = len(file_results['detections'])
                total_detections += file_results['detection_count']

                # Save results to CSV
                success, message = save_results(file_results, output_file_name_and_path) #csv_file_path)
                if not success:
                    print(f"Warning: {message}")

            except Exception as e:
                return False, f"Failed to process {file_path}: {str(e)}"
       
        success_message = f"Successfully processed {len(input_files)} files, found {total_detections} detections"
        return True, success_message
        
    except Exception as e:
        return False, f"Inference error: {str(e)}"


def save_results(results_dict, csv_file_path):
    """
    Save inference results to a CSV file using pandas with each detection as one row.
    
    Args:
        results_dict (dict): Detection results from inference
        csv_file_path (str): Path where CSV file will be saved
    
    Returns:
        tuple: (success, message)
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        
        file_path = results_dict.get('file_path', 'Unknown')
        total_detections = results_dict.get('detection_count', 0)
        detections = results_dict.get('detections', [])

        # Find the start time 
        # TODO: Consider adding +30 seconds here instead of in the packetizer. 
        file_name = os.path.basename(file_path)   # WISPR_240930_000003-0001.jpg
        parts = file_name[6:-9]
        if '_' in parts:
            date_part, time_part = parts.split('_')
            datetime_str = date_part + time_part
            file_datetime = datetime.strptime(datetime_str, "%y%m%d%H%M%S")
        start_time = file_datetime.strftime("%Y-%m-%d %H:%M:%S")

        
        # Prepare data for DataFrame
        data = []
        
        if detections:
            for detection in detections:
                # Format bounding box as "x1,y1,x2,y2"
                bbox = detection.get('bbox', [])
                bbox_str = ",".join([str(round(coord, 2)) for coord in bbox]) if bbox else ""
                
                data.append({
                    'file_path': file_path,
                    'start_time': start_time,
                    'class_name': detection.get('class_name', 'Unknown'),
                    'class_id': detection.get('class_id', -1),
                    'confidence': round(detection.get('confidence', 0.0), 4),
                    'bounding_box': bbox_str,
                    'number_of_detections': total_detections
                })
        else:
            # No detections found - create one row with zeros
            data.append({
                'file_path': file_path,
                'start_time': start_time,
                'class_name': 'None',
                'class_id': -1,
                'confidence': 0.0,
                'bounding_box': '',
                'number_of_detections': 0
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Append to existing CSV or create new one
        if os.path.exists(csv_file_path):
            # Append to existing file without headers
            df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            # Create new file with headers
            df.to_csv(csv_file_path, mode='w', header=True, index=False)
        
        return True, f"Successfully saved {len(data)} detection rows to CSV"
        
    except Exception as e:
        return False, f"Error saving detections to CSV: {str(e)}"

def main():
    print("This program should not be run directly. Use system_control/transform_and_inference.py instead.")
    sys.exit(1)

if __name__ == "__main__":
    main()