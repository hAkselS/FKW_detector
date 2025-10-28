'''
File:   sys_control/shutdown_pi.py

Spec:   Handle the graceful shutdown of the Raspberry Pi system.
        Raise the 'turn me off' gpio pin and manually shutdown the system. 

ID:     sp
'''

from subprocess import call
import time


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
   
    # call("sudo shutdown -h now", shell=True)