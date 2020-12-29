# COVID-Precaution-Monitor
This repository contains all the project files developed by of our team - RAYS, during HackNagpur 2020.

## Purpose:
The purpose for this project is to enable a user to monitor the index of COVID precautionary measures being followed or not.
Our model is a combination of Social-Distancing-Monitor and Fask-Mask-Monitor.
The model helps to monitor people violating saftey norms over video footage from CCTV cameras.

I've used YOLOv3 along with DBSCAN clustering for recognizing potential violations. A pre-trained Face Mask Classifier model (ResNet50) is used for detecting if the people are wearing face masks or not.
Links to both (1)YoloV3.weights & (2)PreTrained ResNet50 Model can be found [here](https://github.com/freAK14/COVID-Precaution-Monitor/tree/main/models).

## Requirements:
Making a virtual enviroment is strongly suggested. [Click here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to know more about virtual enviroments using Anaconda.

Following python packages would be required for running the project:
  * numpy
  * matplotlib
  * sklearn
  * Pillow
  * opencv-python(OpenCV)
  * keras
  * face-detection
  * face-recognition
  * tqdm
  
## Usage:
Run the following command in your terminal:
```
python Covid_Precaution_Monitor.py
```
After a bunch of TensorFlow warnings and stats, you will be able to see a progress bar processing the input video frame-by-frame.

It would take some time to process the video(depending upon your system specifications).

After completion of the progress bar, you would be able to see the ```result.mp4``` file as well as the various output images in the ```results``` folder.(You can comment the particular lines in the code if you don't wish to store these)

**NOTE:** You have to move/remove the result video and files before processing another video, otherwise it will throw an error saying "directory/file already exists".

## Example Output:

![Output GIF](https://github.com/freAK14/COVID-Precaution-Monitor/blob/main/readmefiles/result.gif)

## Potential Improvements:
 ~~We can improve the processing speed by not saving each frame, detected persons and detected faces after each iteration locally.~~
-   [x] Implemented! (Merged own pull request from ```improvements``` branch)


## Contact Me:
Feel free to reachout on my LinkedIn if you got any queries or need any help:

<a href = "https://www.linkedin.com/in/akash-kothare/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>
