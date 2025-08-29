# FKW_detector
## Code written for underwater SeaGliders. 
This repository allows the glider to analyze a subset
of audio data from the previous ascent / descent. 
Code in this repository is designed to run on a 
Raspberry Pi 5. Running on a Raspberry Pi 5, 
analyzing 1 hour's worth of data takes approximately
one WattHour. To analyze data, one minute audio files 
are transformed into spectrograms using scipy, spectrograms
are analyzed using a YOLO11 nano model. 

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

## Logs (not shown)
Logs directory holds important information on what files 
have been seen, what files need to be analyzed, and 
the status of the files that have been analyzed. 
This directory appears when you run process_control.py.

## Venv (not shown)
Make sure to run code in a python3.11 virtual environment! 
Creation: python3 -m venv venv 
Activation: source venv/bin/activate
Setup (Once only): pip intall -r requirements.txt