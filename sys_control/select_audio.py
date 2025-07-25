'''
File:   sys_control/select_audio.py

Spec:   Determine which audio file to process. Store all files to be
        analyzed into a csv file which is parsed by audio_to_spectro.py and
        inference.py. TODO: deciding whether to have one file per deployment or
        one file per ascent / descent. 

Usage:  python3 sys_control/select_audio.py -d <directory> -o <output_file_name>
'''

