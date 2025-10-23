# SETUP
## How to get this repository working once downloaded.

### Creating a virtual environment
*in the project root AKA ~/FKW_detector*

Create a python 3.11 virtual environment in the FKW_detector project repo. 

```bash
python3.11 -m venv venv
```

Activate the virtual environment.

*again, in the project root* 

```bash
source venv/bin/activate
```

This will activate the virtual environement. You should see (venv) in front of your command line prompt. 

```bash
pip install -r requirements.txt
```

This will install all the required libraries to run FKW_detector. It also might take a while (2-20 miunutes).


#### Activating the virtual environment once it has been setup
/start here if you already have a working venv folder/ 
*From the project root*

```bash
source venv/bin/activate
```

This will activate the virtual environment. You should see (venv) in front of your command line prompt. 

#### Running the main program
sys_control/process_control.py is the main script that makes the whole sha-bang run. Under the hood, process control is managing all the other helper scripts to make the magic happen. In general, the other scripts in this project are not meant to be run induvidually. 

*From the project root*
<python3 sys_control/process_control.py>

*Note to people running on the desktop*
If you are seeing that no new images are being analyzed, check to see what is in *logs/analyst_logs*. The observed audio and times json file is used to track what files have been looked at, if your file names are in here, they will not be analyzed again.


## Trouble Shooting

If python3.11 is no longer on your system, try later versions. 
If you are getting into dependency hell, when you continously get dependency issues, try using miniconda to get a python3.11 virtual environment working, this make change how run_detector.sh works slightly, but Gemini can help you with that. 