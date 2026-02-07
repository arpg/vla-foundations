# Gyanig - Rough Writeup
*(Better readability than using Google Sheets)*

* Lots of dataset available - need to find a ways to organize and structure the data
* Cut off at the chain of thought to verify if the gaze information is meaningful to

## Kali
* Radar data from Korean group K-radar (high frequency), also delft data to implement and fine-tune VLA models
* Radical - driving scenes from a robot vs car scene to build the vision encoder
* Millimeter and radar sensors - train the action prediction
* Possible - learning of the extra things like person driving capabilities
* Synthetic data with Carla
* The issue is not a ton of millimeter-level data; it relies on millimeter wave when the camera data fails
* Could be something that Carson can use as well
* Bridge to the action from mm data
* Stick to multi-modality - not work on standalone radar vs camera systems
* Orin for inference
* TX cascader radar - 4 receiver - 48 small components generating data - radar data cubes - after processed - pointcloud dependent on area of coverage - sparse - loads of data

## Callie
* Interested in Lidar, improve the performance of the perception
* Data - CLIP
* CARLA driving

## Jay/Carson
* Proximity sensors, train VLA policy
* Edge computing constraints, synthetic dataset, generate data
* Edge compute - 8x8 image, so might be possible - downsizing the encoder to train
* Looking on how to upscale from 8x8 to higher, how to tokenize things
* Reconstruct the features… a new
* Question - pointcloud tokenization, pointcloud based reasoning and feature extraction?