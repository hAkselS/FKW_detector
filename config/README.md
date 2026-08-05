# User configurations for FKW_detector
#### Mess with these parameters to change operation variables

### Allowed Runtime Minutes (allowed_runtime_minutes: 15)
How long should the Raspberry be allowed to run before 
shutting down. Note, the timer running out will NOT cause the 
forced shutdown flag to be True. 

### Base Audio Directory (base_audio_directory: path/to/directory)
Where should the detector look for .dat audio files?
This path should point to the *directory* where .dat files are expected to show up. 

### Confidence Threshold (confidence_threshold: 0.25)
How confident does the model need to be before counting a detection? 


### Mission Mode (mission_mode: True/False)
If this flag is set to True, the RPi5 will shut itself down when analysis is complete.
It is ESSENTIAL that this flag is true when the detector is deployed on a SeaGlider!!!

### Model Path (model_path: path/to/model.pt)
What YOLO model would you like to use? 

### Number of Files to Analyze (num_files_to_analyze: 60)
How many audio files should be processed?
Our initial guess is approximately 60 files. 
In lab, 60 files equates to 1Wh of energy consumption on the RPi5. 
