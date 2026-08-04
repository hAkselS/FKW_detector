# Start on Boot
### Make FKW_detector Run Automatically. 

*Skip to 'steps' to avoid learning anything.*
THIS SHOULD ONLY BE DONE A RASPBERRY PI, not on a personal or goverment computer. 

### Intro to Systemd
Ensuring that a program runs immediately after boot time is a non-trivial task. We have elected to take the Systemd approach to accomplish this.  Systemd is a power operating system level tool, and so the instructions should be followed carefully. 

### What are we trying to do?
We are created a systemd / systemclt service that becomes part of the operating system's "to-do list" at boot time. Our service, now part of the operating systems to-do list, should start the FKW_detector, and since the FKW_detector is engineered to handle everything from new data discovery to packetizing results, thats all we have to do. 

## Steps
### 1. Copy save-whales.service into systemd space

Use the 'cp' (copy) command to move ```FKW_Detector/sys_control/save-whales.service``` into the user space of the systemd directory. In English, put save_whales.service in /etc/systemd/user

*Run from project root*

```bash
sudo cp sys_control/save-whales.service /etc/systemd/user/
```

### 2. Reload the daemon, enable save-whales

Reload the the systemd user space daemon. Do not forget the '--user' or else this will not work. 

```bash
systemctl --user daemon-reload
```

Enable the new save-whales serive

```bash
systemctl --user enable save-whales.service
```

### 3. Test number one
Theoretically, in step two you added a new service to systemd's user boot time 'to-do' list. Because this is mission critical, let's make sure it works. 

Try running the FKW_detector by manually starting the service which activates the detector. 

```bash 
systemctl --user start save-whales.service
```

See if it worked by looking at the status logs

```bash
systemctl --user status save-whales.service
```

If it worked, you'll see ```Process: ...other stuff... (code=exited, status=0/SUCCESS)```

You may also see ```Main PID': #### (run_detector)```, this means it's still running and you should wait a bit and check again. 

### 4. Test number two

Power off the Raspberry Pi, wait ten seconds, power on the Raspberry Pi, and determine if the detector has started running. 

## Determine if the detector is running: 

#### Method 1: Check the FKW_detector logs.

In sys logs, there should be a new file for every time the system is run (via this method), this is probably your best indication if the system has run. Sys log files are named by the RASPBERRY PI's start time, this may be completely different from the other files who are named based on the .dat files they are analyzing. In each sys log file are the print statements you would see if you manually ran the detector using ```python3 sys_control/main.py```. 

In analyst logs, there should be a list of every file (and associated time) that the detector has seen. You'll know that the detector has run if there are files here. However, if you've ran the detector multiple times without adding files to the base audio directory (specified in the config.yaml) the list will not change. It is safe to delete files from this list or even remove the entire analyst logs directory if you want to analyze the same group of files multiple times for testing. 

In dive logs, there should be a file for every 'collection' of data analyzed. The collection is titled 'first_datetime-last_datetime.csv'. This file has the status of each file that you tried to analyze. Note, if you analyze the exact same group of files twice, they will have exact start and end datetimes, hence the new file will write over the old one and you will have the exact same number of files as before. 

What I usually do: Typically I delete the entire logs directory and data_products directory to see only new content after each run. This is perfectly safe to do... but DO NOT DO THIS IF YOU HAVE VALUABLE MISSION DATA STORED IN LOGS OR DATA_PRODUCTS. 


### Method 2: Check the systemclt status

Use the status command from earlier to see how the process is doing. 

```bash
systemctl --user status save-whales.service
```
