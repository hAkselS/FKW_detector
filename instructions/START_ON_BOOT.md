# START ON BOOT
## How to get the code to run automatically when the Raspberry Pi boots up. 

*skip to 'steps' to avoid learning anything.*
THIS SHOULD ONLY BE DONE A RASPBERRY PI, not on a personal or goverment computer. 

### Intro to Systemd
Unfortunately, ensuring that a program runs immediately after boot time is a non-trivial task. Hence, we use systemd to accomplish the task. Systemd is a power operating system level tool, and so the following instructions should be followed carefully. 

### What are we trying to do?
We are created a systemd / systemclt service that becomes part of the operating system's "to-do list" at boot time. Our service, now part of the operating systems to-do list, should start the FKW_detector, and since the FKW_detector is engineered to handle everything from new data discovery to packetizing results, thats all we have to do. 

## Steps

# 1. Copy save-whales.service into systemd space

Use the 'cp' (copy) command to move save-whales.service into the user space of the systemd directory. In English, put save_whales.service in /etc/systemd/user

*run from project root*

```bash
sudo cp sys_control/save-whales.service /etc/systemd/user/
```
TODO: this need testing!!!

# 2. Reload the daemon, enable save-whales
remove parens if not necessary(
Move to systemd user directory

```bash
cd /etc/systemd/user/
```
)

Reload the the systemd user space daemon. Do not forget the '--user' or else this will not work. 

```bash
systemctl --user daemon-reload
```

Enable the new save-whales serive

```bash
systemctl --user enable save-whales.service
```

# 3. First test 
Theoretically, in step two you added a new service to systemd's user boot time 'to-do' list. Because this is mission critical, let's make sure it works. 

Try running the FKW_detector by manually starting the service which activates the detector. 

```bash 
systemctl --user start save-whales.service
```

See if it worked by looking at the status logs

```bash
systemctl --user status save-whales.service
```

If it worked, you'll see 'Process: ...other stuff... (code=exited, status=0/SUCCESS)'

You may also see 'Main PID': #### (run_detector)', this means it's still running and you should wait a bit and check again. 

# 4. Second test

Power off the Raspberry Pi, wait ten seconds, power on the Raspberry Pi, and determine if the detector is running. 

## Determine if the detector is running: 

#### Method 1: Check the FKW_detector logs.
In analyst logs, there should be a list of every file (and associated time) that the detector has seen. You'll know that the detector has run if there are files here. However, if you've ran the detector multiple times without adding files to the base audio directory (specified in the config.yaml) the list will not change. It is safe to delete files from this list or even remove the entire analyst logs directory if you want to analyze the same group of files multiple times for testing. 

In dive logs, there should be a file for every 'collection' of data analyzed. The collection is titled 'first_datetime-last_datetime.csv'. This file has the status of each file that you tried to analyze. Note, if you analyze the exact same group of files twice, they will have exact start and end datetimes, hence the new file will write over the old one and you will have the exact same number of files as before. 

In sys logs, there should be a new file for every time the system is run, this is probably your best indication if the system is running when you want it to. sys(tem) log files are named by the RASPBERRY PI's start time, this may be completely different from the other files who are named based on the files they are analyzing. In each sys log file is all the print statements from all the programs in the FKW_detector. 

# Method 2: Check the systemclt status

Use the status command from earlier to see how the process is doing. 

```bash
systemctl --user status save-whales.service
```