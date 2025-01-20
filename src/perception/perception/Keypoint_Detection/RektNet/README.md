# RektNet Files

Replace the files in the MIT Repository with the files available here. The changes made are to facilitate running the model on a local computer, as opposed to Colab. Another objective was to obtain the keypoints as pixel coordinates and save them to an Excel file.

## Project Structure

* Clone the [parent repository](https://github.com/gopalaniyengar/Keypoint-Detection "Keypoint-Detection").
* Clone [MIT Driverless repository](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra) within it (say 'MITRepo').
* Replace the following files in ```MITRepo/RektNet/``` with the versions uploaded here: 
	* ```dataset.py```
	* ```detect.py```
	* ```train_eval.py```
	* ```utils.py```
* Unzip the [RektNet Dataset](https://drive.google.com/file/d/1PblXu77314Ah6SjkRu2SrCqR_oRz4dBA/view?usp=sharing "Drive Link to Version Used") in ```./MITRepo/RektNet/dataset/```.
* Copy the [Labels File](https://docs.google.com/spreadsheets/d/1kBM-sWIrZzb8jPE5Fdvs2cy0g7wDBmVceY-dTCsz1Qg/edit?usp=sharing "Drive Link to Version Used") to ```./MITRepo/RektNet/dataset/```.
* Refer "Changes Made" section below to change some absolute path variables. 

## Execution

* Execution performed in IDE Terminal.
* Navigate to ```./MITRepo/RektNet/```
* __Training Command__: Default training with Geo Loss included (optimal hyperparameters as obtained from MIT-Driverless' Paper). Also visualizes Average Keypoint MSE vs BB size distribution in ```./MITRepo/RektNet/logs/rektnet_validation.txt```.
```
 python train_eval.py --study_name <name of training instance here>
```
* Weights are saved to ```./MITRepo/RektNet/outputs/```
* __Inference Command__: Default inference with image output saved to ```./MITRepo/RektNet/outputs/visualization/```.
```
 python detect.py --model "./MITRepo/RektNet/outputs/<path to weights file>" --img <path to input image>
```

## Changes Made

* ```utils.py```: Added exception handling, functionality to return points as pixel coordinates in ```vis_hm_and_save()```.
* ```train_eval.py```: 
	* _line 143_- File opening mode changed to facilitate creation of file if it doesn't exist.
	* _lines 198, 200_- Changed default path.
	* _lines 211, 212, 217_- Changed default argument values.
* ```dataset.py```: Resolved problem in image path referencing.
* ```detect.py```:
	* _lines 40-43_- Attempting to load ```.onnx``` models as well, not fully implemented.
	* _lines 55-58_- Console display changes.
	* _lines 63-71_- Detected keypoints as Excel file output.
	* _line 84_- Changed default path.
