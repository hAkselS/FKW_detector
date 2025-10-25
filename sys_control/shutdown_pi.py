'''
File:   sys_control/shutdown_pi.py

Spec:   Handle the graceful shutdown of the Raspberry Pi system.
        Raise the 'turn me off' gpio pin and manually shutdown the system. 

ID:     sp
'''


from subprocess import call
import time

# TODO: move this to config file
config_GPIO_on_time = 10
config_GPIO_pin_num = 11 # Pin 11 should be a GPIO only pin

def raise_GPIO(pin_number, wait_time):
    if (wait_time < 10):
        print(f"\nsp: minimum wait time must be 1 second")
        wait_time = 3

    print(f"\nsp: Raising GPIO [{pin_number}] for [{wait_time}] seconds")
    # Raise GPIO TODO
    time.sleep(wait_time)


def shutdown(reason="Not specified"):
    '''
    Called by process control to shutdown the pi.

    args: string, reason for shutting down
    '''
   
    raise_GPIO(config_GPIO_pin_num ,config_GPIO_on_time)
   
    print(f"sp: Shutting down due to: {reason}")
   
    # call("sudo shutdown -h now", shell=True)