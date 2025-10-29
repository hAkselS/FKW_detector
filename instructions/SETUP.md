# SETUP
## How to get this repository working once downloaded.

### Creating a virtual environment

Create a python 3.11 virtual environment in the FKW_detector project repo. 

*While in the project root*

```bash
 ~/FKW_detector/
```

Run this command to make the virtual environment:
```bash
python3.11 -m venv venv
```

Activate the virtual environment.

*again, in the project root* 

```bash
source venv/bin/activate
```

This will activate the virtual environement. You should see (venv) in front of your command line prompt. Like so:

```bash
(venv) home/camera/FKW_detector$ 
```

Now that our virtual environment is active, we can install all our project dependencies into that virual environment using:

*again, in the project root* 

```bash
pip install -r requirements.txt
```

This command will install all the required libraries to run the FKW_detector. This install may take up to twenty.


### Activating the Virtual Environment Once it Has Been Setup
/start here if you already have a working venv folder/ 
*From the project root*

```bash
source venv/bin/activate
```

This will activate the virtual environment. You should see (venv) in front of your command line prompt. 

## Running the main program
sys_control/process_control.py is the main script that makes the whole sha-bang run. Under the hood, process control is managing all the other helper scripts to make the magic happen. In general, the other scripts in this project are not meant to be run induvidually. 

*From the project root*
<python3 sys_control/process_control.py>

*Note to people running on the desktop*
If you are seeing that no new images are being analyzed, check to see what is in *logs/analyst_logs* directory. The observed audio and times json file is used to track what files have been previously seen, if your file names are in here, they will not be analyzed again.


## Trouble Shooting

#### Dependency Issues: 
If you're getting errors that rhyme with:
```bash
Traceback (most recent call last):
  File "sys_control/process_control.py", line 19, in <module>
    import yaml
ModuleNotFoundError: No module named 'yaml'
```

Or something that includes, "this version of X requires this verion of Y",
then you probably have a dependency issue. In this case you should delete the virtual environment folder using:

```bash
rm -rf venv
```
From the project root. Then you should go into the requirements.txt file using:

```bash
nano requirements.txt
```
And try messing with the file versions or adding dependencies. Note that each time you do this you will need to create a virtual environment AND install its dependencies as outlined above. 

#### No Data Outputs: 
