'''
File:   sys_control/shutdown_pi.py

Spec:   Handle the graceful shutdown of the Raspberry Pi system.
        Raise the 'turn me off' gpio pin and manually shutdown the system. 

ID:     sp
'''

def shutdown(reason): 

    print(f"sp: Shutting down due to: {reason}")
    print("sp: Raising 'turn me off' GPIO pin...")
    # TODO: Code to raise GPIO pin goes here
    print("sp: GPIO pin raised. Shutting down now...")
    # TODO: Code to shutdown the system goes here

    print("sp: Shutting down Raspberry Pi...")
    print("sp: THIS SCRIPT IS NOT FINISHED OR TESTED!!!")