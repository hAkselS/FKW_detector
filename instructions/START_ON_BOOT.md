# START ON BOOT
## How to get the code to run automatically when the Raspberry Pi boots up. 

*skip to 'steps' to avoid learning anything.*
THIS SHOULD ONLY BE DONE A RASPBERRY PI, not on a personal or goverment computer. 

### Intro to Systemd
Unfortunately, ensuring that a program runs immediately after boot time is a non-trivial task. Hence, we use systemd to accomplish the task. Systemd is a power operating system level tool, and so the following instructions should be followed carefully. 

### What are we trying to do?
We are created a systemd / systemclt service that becomes part of the operating system's "to-do list" at boot time. Our service, now part of the operating systems to-do list, should start the FKW_detector, and since the FKW_detector is engineered to handle everything from new data discovery to packetizing results, thats all we have to do. 

## Steps

# 1. Copy save-whales.service into the operating system

Use the 'cp' (copy) command to move save-whales.service into the user space of the systemd directory. In English, put save_whales.service in /etc/systemd/user

*Run from project root*

```bash
sudo cp sys_control/save-whales.service /etc/systemd/user/
```
TODO: this need testing!!!