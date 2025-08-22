# User configurations for FKW_detector
## Mess with these parameters to change operation variables

### Forced Shutdown (forced_shutdown: True/False)
If this flag is True that means the the pi shutdown unintentionally.
This could be that the Pi froze or a similar critical error.
THIS FLAG SHOULD BE 'False' BEFORE DEPLOYMENT 

### Base Audio Directory (base_audio_directory: 'path/to/directory')
Where should the detector look for .dat audio files?
This path should point to the *directory* where .dat files are expected to show up. 


## Analyst Parameters
### Number of Files to Analyze (num_files_to_analyze: 60)
How many audio files should be processed?
Our initial guess is approximately 60 files. 
In lab, 60 files equates to 1Wh of energy consumption on the RPi5. 

### Allowed Runtime Minutes (allowed_runtime_minutes: 15)
How long should the Raspberry be allowed to run before 
shutting down. Note, the timer running out will NOT cause the 
forced shutdown flag to be True. 