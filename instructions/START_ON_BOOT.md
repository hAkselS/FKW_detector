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

# 2. Reload the Daemon, enable save-whales
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

# 3. Test (optional, but not really optional)
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