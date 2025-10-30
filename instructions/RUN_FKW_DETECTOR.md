# Running the False Killer Whale Detector

### Before you run
Make sure that your virtual environment is setup and activated. See, ```FKW_detector/instructions/SETUP.md``` for instructions on setting up a virtual environment. 

Ensure that the ```base_audio_directory:``` category in ```FKW_detector/config/config.yaml``` points to where you .dat file data is or will be. 

Note, start on boot should eliminate the need to run this project manually, see, ```FKW_detector/instructions/START_ON_BOOT.md``` for instructions on setting this up. 

The following assumes that you want to test run the FKW_detector in a bench-top scenario. 

### Running the FKW_detector

Run the following command from the project root:

```bash
python3 sys_control/process_control.py
```


### How it works
sys_control/process_control.py is the main script that makes the whole sha-bang run. Under the hood, process control is managing all the other helper scripts to make the magic happen. In general, the other scripts in this project are not meant to be run induvidually. 


*Note to people running on the desktop*
If you are seeing that no new images are being analyzed, check to see what is in *logs/analyst_logs* directory. The observed audio and times json file is used to track what files have been previously seen, if your file names are in here, they will not be analyzed again.
