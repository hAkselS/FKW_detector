# START ON BOOT
## How to get the code to run automatically when the Raspberry Pi boots up. 

*skip to 'steps' to avoid learning anything.*

### Intro to Systemd
Unfortunately, ensuring that a program runs immediately after boot time is a non-trivial task. Hence, we use systemd to accomplish the task. Systemd is a power operating system level tool, and so the following instructions should be followed carefully. 

### What are we trying to do?
We are created a systemd / systemclt service that becomes part of the operating system's "to-do list" at boot time. Our service, now part of the operating systems to-do list, should start the FKW_detector, and since the FKW_detector is engineered to handle everything from new data discovery to packetizing results, thats all we have to do. 

## Steps

# 1. 