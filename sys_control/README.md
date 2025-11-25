# System Control

## Who lives here?
These scripts generally are the 'managers' of the repository. In otherwords, these scripts don't do very much work, but instead call on the other scripts to do the right work at the right time in the right order. 

## save-whales.service
This script has no business living in the project repo! While its presence is in no way disruptive, it needs to be copied into /etc/systemd/user to do anything. See instructions/start_on_boot for more details. 

## run_detector.sh VS process_control.py
Run detector simply calls process control, if you want to test this repository on your laptop or a benchtop Raspberry Pi 5, run process_control.py. More on this in the instructions folder. 