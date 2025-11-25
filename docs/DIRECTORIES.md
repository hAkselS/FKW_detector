# Directories
### Who lives where? 

## Audio Transform
The audio tranform directory holds code used
to transorm audio into spectrograms. Audio comes in
in the .dat file format, is translated to .wav format,
then transformed into spectrograms.

## Image Inference
The image inference directory holds code that handles
running inference on spectrograms and packetizing 
inference outputs into radio transmittable products. 

## Models
AI models designed to detect False Killer Whales live here. 
This repository expects ultralytics YOLO type models,
preferably yolo8 and yolo11.  

## Sys Control
The system control directory holds code that starts the detector,
selects the audio files for analysis, calls the translation and 
transformation scripts, and handles life cycle. This code 
operates the system. 

## Data Products (not shown)
This directory holds all of the data products created by
the FKW_detector. Data products include wav files, spectrograms,
inference information, and packetized results (also inference information). 
Think of this directory as the 'output' directory.

## Logs (not shown)
Logs directory holds important information on what files 
have been seen, what files need to be analyzed, and 
the status of the files that have been analyzed. 
This directory appears when you run process_control.py.
Generally, these logs are only essential to developers,
however, if you are running bench top tests on a fixed set of files,
you may need to clear our the analyst logs to re-analyze files.

## Venv (not shown)
Make sure to run code in a python3.11 virtual environment! 
Creation: python3 -m venv venv 
Activation: source venv/bin/activate
Setup (Once only): pip intall -r requirements.txt
More in the SETUP.md file. 