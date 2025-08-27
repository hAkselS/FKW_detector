'''
File:   image_inference/packetize_results.py

Spec:   Parse existing inference outputs and remove all information
        that is non essential for the ground team. 
        In other words, remove everything except the time and detection count. 

ID:     pr 

Usage:  Call this from another script!
'''

import pandas as pd 
import os 

# TODO: Fix logic so that only positive detections are packetized
# RUN THIS SCRIPT BY ITSELF FOR THE TIME BEING

# def filter_by_bbox_area(input_csv, area_threshold):
#     """
#     Exclude detections with bounding box area below the threshold.
#     Returns a filtered DataFrame.
#     """
#     import pandas as pd

#     df = pd.read_csv(input_csv)
#     if 'bounding_box' not in df.columns:
#         raise ValueError("CSV must have a 'bounding_box' column.")

#     def bbox_area(bbox_str):
#         try:
#             coords = [float(x) for x in bbox_str.split(',')]
#             if len(coords) == 4:
#                 x1, y1, x2, y2 = coords
#                 return abs((x2 - x1) * (y2 - y1))
#         except Exception:
#             return 0.0
#         return 0.0

#     df['bbox_area'] = df['bounding_box'].apply(bbox_area)
#     filtered_df = df[df['bbox_area'] >= area_threshold].copy()
#     filtered_df.drop(columns=['bbox_area'], inplace=True)
#     return filtered_df

def packetize_inference_outputs(input_csv):
    # Find the name of the existing file 
    base_name = os.path.basename(input_csv)
    # Remove 'detection' from the base name
    new_name = base_name.replace('detection', 'packetized')
    
    # Create the output directory inside data_products/packets
    output_dir = os.path.join('data_products', 'packets')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, new_name)

    try:
        # Read the existing CSV file
        df = pd.read_csv(input_csv)

        for row in df.itertuples():
            file_path = row.file_path
            total_detections = row.number_of_detections
            start_time = row.start_time
            print(f"total detections {total_detections}")



        # Save the packetized results
       # packetized_df.to_csv(output_csv, index=False)
        message = f"Packetized {total_detections} positive detections to {output_csv}"
        # print(message)
        return True, message, output_csv

    except Exception as e:
        message = f"Failed to packetize results: {str(e)}"
        print(message)
        return False, message, ""
    

def main():
    input_csv = 'data_products/inference_outputs/240930_000003-241001_000449_detections.csv'
    packetize_inference_outputs(input_csv)

if __name__ == "__main__":
    main()
    
