# SETUP
## How to get this repository working once downloaded.

### Desktop implementation
Before you can run any code, you need to have a working conda environment. 

#### First time conda setup
Setting up a conda environment is generally straight forward and documentation can be found here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
 
You will need to run approximately these commands in this order. 

*From the project root AKA .../FKW_detector/*

<conda create --name venv python=3.11>
This will create a brand new conda environment called "venv". A new folder called venv should appear in the project root directory. 

<source venv/bin/activate>
This will activate the conda environement. You should see (venv) in front of your command line prompt. 

<pip install -r requirements.txt>
This will install all the required libraries to run FKW_detector. 

#### Activating conda once it has been setup
*From the project root*
<source venv/bin/activate>
This will activate the conda environement. You should see (venv) in front of your command line prompt. 

#### Running the main program
sys_control/process_control.py is the main script that makes the whole shabang run. Under the hood, process control is managing all the other helper scripts to make the magic happen. In general, the other scripts in this project are not meant to be run induvidually. 

*From the project root*
<python3 sys_control/process_control.py>

*Note to people running on the desktop*
If you are seeing that no new images are being analyzed, check to see what is in *logs/analyst_logs*. The observed audio and times json file is used to track what files have been looked at, if your file names are in here, they will not be analyzed again.