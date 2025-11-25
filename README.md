# FKW_detector
## Code written for underwater SeaGliders. 
This repository is designed as a software/hardware
addition to an autonomous sea glider that allows the 
sea glider to analyze audio data for the presence of
False Killer Whale (FKW) whistles during, approximately, two-month deployments.
This repository is designed specifically to run on a 
Raspberry Pi 5. Newer versions of the Raspberry Pi may work,
but version below the 5 tend to be too slow. 

## Overview Diagram
<img src="docs/DIAGRAM_1.png" alt="Detailed FKW_detector Diagram" width="600"/>

## Intended Usage 
This repository is designed to run on a Raspberry Pi 5 on a SeaGlider
brand automous sea glider.
The main purpose of this code is to create a small packet of suspected 
instances of False Kilelr Whale wistles in recorded audio data that is transmittable to
NOAA headquarters.
The Raspberry Pi 5 this code runs on is setup to run 
the FKW_detector code automatically at start time (instructions for
setting this up are provided in [/docs](docs)). It is up to the user
to tune mission parameters, such as the amount of data to analyze, in the 
[config.yaml](config.yaml) file.

## Performance Metrics 
#### Hardware Performance Metrics
The authors of this code have measured the performance metrics in laboratory
setting using an ~analog~ power supply and the "time" command in Linux. The following performance metrics were recorded:
- Average time to analyze 10 minutes of audio data: 
  - Real time:      3 minutes   6 seconds
  - User time:      1 minute    34 seconds
  - System time:    0 minutes   8 seconds

- Average voltage and current draw during analysis:
  - Voltage:        5.2 Volts
  - Current:        1.2 Amps

- Energy and time to analyze 60 minutes of audio data:
    - Energy:         1.06 Watt-Hours
    - Time:           10 minutes 12 seconds (real time)

- Suggested duty cycle: 
    - Analyze 1 hour of audio per each two to three hour ascend / descent cycle.

#### Detection Performance Metrics
- YOLO11n summary: 
    - 181 layers, 2,590,035 parameters, 0 gradients, 6.4 GFLOPs 
(181, 2590035, 0, 6.4406016)

Note: it is incredibly easy to switch models, as the mission progresses, a better
model will likely replace the current one ([fkw_whistle_classifier_2.0.pt](models/fkw_whistle_classifier_2.0.pt)). 

## Documentation
Further documentation, including instructions for setting up the Raspberry Pi 5,
running the repository manually, interpretting results, reading logs, and more can 
be found in the [/docs](docs) folder.