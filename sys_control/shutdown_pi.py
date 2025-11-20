'''
File:   sys_control/shutdown_pi.py

Spec:   Handle the graceful shutdown of the Raspberry Pi system.
        Raise the 'turn me off' gpio pin and manually shutdown the system. 

ID:     sp
'''

from subprocess import call
import time
import sys 
import os 
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

###################################################################
# CONFIGURATION DEFAULTS
config_file = project_root + '/config/config.yaml'
###################################################################

def raise_GPIO(pin_number, wait_time):
    # NOTE: No error checking on valid pin number.
    if (wait_time < 15):
        print(f"\nsp: minimum wait time must be 15 seconds")
        wait_time = 15

    print(f"\nsp: Raising GPIO [{pin_number}] for [{wait_time}] seconds")
    # Raise GPIO TODO
    time.sleep(wait_time)

# Shutdown the Raspberry Pi
def shutdown(reason="Not specified", config_GPIO_on_time=15, config_GPIO_pin_num=11):
    '''
    Called by process control to shutdown the pi. 
    Defaullts to the minimum 15 GPIO on time to prevent rapid power cycling.
    Defaults to GPIO pin 11, an arbitrary choice.

    args: string, reason for shutting down
    '''
   

    raise_GPIO(config_GPIO_pin_num ,config_GPIO_on_time)
   
    print(f"sp: Shutting down due to: {reason}")
    
    # TODO: THIS HAS NOT BEEN TESTED!!!!
    # Check if the system is in mission mode
    try: 
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"sp: Error reading config file: {e}")
        print("sp: Unable to verify mission mode, proceeding with shutdown.")
        # Shutdown in the event that we cannot read the config
        call("sudo shutdown -h now", shell=True)
        return
    
    # Prevent shutdown in the event of non-mission mode (aka you're bench top testing)
    if not config.get('mission_mode', False):
        print("sp: Mission mode not enabled, aborting shutdown.")
        return
    else: 
        # Shutdown the Pi 
        call("sudo shutdown -h now", shell=True)
    