# SETUP
## How to get this repository working once downloaded.

### Creating a virtual environment

To run the FKW_detector, you must create a python 3.11 virtual environment in the FKW_detector project directory. 

*While in the project root ``` ~/FKW_detector/```,* run this command to create the virtual environment:

```bash
python3.11 -m venv venv
```

Now, you need to activate the virtual environment that you created.

*From the project root, ```~/FKW_detector/```, run:* 

```bash
source venv/bin/activate
```

This command will activate the virtual environement. You should now see (venv) in front of your command line prompt. Like so: ```(venv) home/camera/FKW_detector$ ```

Now that the virtual environment is active, you can install all the project dependencies into that virual environment by running the following command in the project root directory:

```bash
pip install -r requirements.txt
```

This command will install all the required libraries to run the FKW_detector. This install may take up to twenty.


## Activating the Virtual Environment Once it Has Been Setup
*Start here if you already have a working venv folder.*

Run the following command from the project root to activate your FKW_detector virtual environment:

```bash
source venv/bin/activate
```

This command will activate the virtual environement. You should now see (venv) in front of your command line prompt. Like so: ```(venv) home/camera/FKW_detector$ ```

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
then you probably have a dependency issue. In this case you need to try to solve the dependency issue by installing the correct dependency into virtual environment or modified some of the dependency versions until they are compatable.

If you are missing someting, like in the error shown above, try pip installing it using:

```bash
pip install theThingImMissing # make sure your venv is active!
```

For more complex issues or when you want the changes you make to be persistent, you should go into the requirements.txt file using:

```bash
nano requirements.txt
```

You can modify dependency versions by changing the version number of the dependency that is giving you trouble. Ex:

```bash
 'scipy==1.15.1' # change this number only if necessary
 ```
 
 You can also add new dependencies if necessary by simply adding the dependency as a new line in the requirements.txt file. Like so:

```bash
scipy==1.15.1
ultralytics
myNewDependency=1.3.5
```
  
Note, each time you perform eitehr of these approaches you will need to either re-install your requirements.txt, as performed above, or completely destroy your virtual environment and rebuild it. 

To perform the later, deactivate the virtual environment, remove the ```venv``` folder then rebuild the virtual environment. 

```bash
deactivate # deactivate the virtual environment
rm -rf venv # burn down the old virtual environment
python3.11 -m venv venv # create a new venv
pip install -r requirements.txt # install your updated requirements
source venv/bin/activate # activate! 
```

For complex issues, burning down the whole virtual environment folder is prefered to ensure a fresh environment each time. 


#### No Data Outputs: 
If you're getting no data outputs, check the sys_logs to see if the detector is running and if there are any preventative errors. For example, if the detector cannot find the config file, it will shutdown immdediately, or if the detector does not see any new data it will exit without producing any data outputs.

