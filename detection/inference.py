'''
File:   detection/inference.py

Spec:   Perform inference on spectrogram images to detect FKWs using YOLO.

Usage:  python3 detection/inference.py <input_file1> <input_file2> -o <output_file>

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
        return True, success_message, # TODO 
        
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
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Perform YOLO inference on spectrogram images')
    parser.add_argument('input_files', nargs='+', help='Path(s) to input image files')
    parser.add_argument('-o', '--output', help='Output file for results (JSON format)')
    parser.add_argument('-m', '--model', default=model_path, help='Path to YOLO model file')
    parser.add_argument('-c', '--confidence', type=float, default=confidence_threshold, 
                       help='Confidence threshold for detections')
    parser.add_argument('--save-images', action='store_true', 
                       help='Save annotated images with detections')
    
    args = parser.parse_args()
    
    print(f"\nRunning inference on {len(args.input_files)} file(s):")
    for file_path in args.input_files:
        print(f"  - {file_path}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    
    # Perform inference
    success, message, results = perform_inference(
        args.input_files, 
        args.model, 
        args.confidence
    )
    
    if success:
        print(f"\n✓ {message}")
        
        # Print summary
        for filename, file_results in results['files'].items():
            detection_count = file_results['detection_count']
            print(f"  {filename}: {detection_count} detection(s)")
            
            # Print detection details
            for i, detection in enumerate(file_results['detections']):
                conf = detection['confidence']
                class_name = detection['class_name']
                print(f"    Detection {i+1}: {class_name} (confidence: {conf:.3f})")
        
        # Save results if output file specified
        if args.output:
            save_success, save_message = save_results(results, args.output)
            if save_success:
                print(f"✓ {save_message}")
            else:
                print(f"✗ {save_message}")
        
        # Save annotated images if requested
        if args.save_images:
            try:
                model = YOLO(args.model)
                for file_path in args.input_files:
                    results_obj = model(file_path, conf=args.confidence)
                    # Save with detections drawn
                    annotated_path = file_path.replace('.jpg', '_annotated.jpg')
                    for result in results_obj:
                        result.save(annotated_path)
                    print(f"✓ Saved annotated image: {annotated_path}")
            except Exception as e:
                print(f"✗ Failed to save annotated images: {str(e)}")
        
    else:
        print(f"\n✗ {message}")
        sys.exit(1)

if __name__ == "__main__":
    main()