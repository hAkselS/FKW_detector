###FKW_detector
##Code written for underwater SeaGliders. 

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
to transorm audio into spectrograms. 

## Prediction Interface
The prediction interface directory hold code that handles selection of
audio and images for analysis. Detection also holds 
the script used to analyze spectrograms with AI. 

## Models
Transfer learning trained models live here. 

## Sys Control
The system control directory holds code that allows the 
detector to start up and ensures that all scripts run in
the right order without any errors.  

## Logs 
Logs directory holds important information on what files 
have been previously analyzed as well as how the system is
running and any error reporting. 

## Venv (not shown)
Make sure to run code in a python3.11 virtual environment! 
Creation: python3 -m venv venv 
Activation: source venv/bin/activate
Setup (Once only): pip intall -r requirements.txt