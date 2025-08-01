'''
File:   detection/inference.py

Spec:   Perform inference on spectrogram images to detect FKWs using YOLO.

Usage:  Do not run this program directly. Call from system_control/transform_and_inference.py.

I/O:    This program expects one or more spectrogram images as inputs. 
        This program outputs inference results and a success boolean. 
'''

from ultralytics import YOLO
import argparse
import os
import sys
import json
from datetime import datetime



###################################################################
# CONFIGURATION DEFAULTS
model_path = "models/fkw_whistle_classifier_2.0.pt"   # Update with your trained model path
# model_path = 'models/yolo11n.pt' # For debugging 
confidence_threshold = 0.25                           # Minimum confidence for detections

# Project root 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

###################################################################

def perform_inference(input_files, output_directory=project_root + '/images'):
    """
    Perform YOLO inference on a list of image files.
    
    Args:
        input_files (list): List of paths to image files
        model_path (str): Path to the YOLO model file
        confidence (float): Confidence threshold for detections
    
    Returns:
        tuple: (success, message, results_dict)
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}", {}
        
        # Check if all input files exist
        for file_path in input_files:
            if not os.path.exists(file_path):
                return False, f"Input file not found: {file_path}", {}
        
        # Load YOLO model
        try:
            model = YOLO(model_path)
        except Exception as e:
            return False, f"Failed to load model: {str(e)}", {}
        
        
        total_detections = 0
        
        for file_path in input_files:
            try:
                # Run inference
                results = model(file_path, conf=confidence_threshold)
                
                # Extract results for this file
                file_results = {
                    'file_path': file_path,
                    'detections': [],
                    'detection_count': 0
                }
                
                # Process detections
                for result in results: # TODO: use existing FKW tools as a template for this (just return number of positive detections and confidence level.)
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
                
            except Exception as e:
                return False, f"Failed to process {file_path}: {str(e)}", {}
       
        
        success_message = f"Successfully processed {len(input_files)} files, found {total_detections} detections"
        return True, success_message
        
    except Exception as e:
        return False, f"Inference error: {str(e)}", {}

def save_results(results_dict, output_file):
    """Save inference results to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        return True, f"Results saved to {output_file}"
    except Exception as e:
        return False, f"Failed to save results: {str(e)}"

def main():
    print("This program should not be run directly. Use system_control/transform_and_inference.py instead.")
    sys.exit(1)

if __name__ == "__main__":
    main()