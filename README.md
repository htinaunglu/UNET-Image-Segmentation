## Image Segmentation with U-Net Algorithm on Carvana Dataset using AWS Sagemaker
This is a full project of image segmentation using the model built with U-Net Algorithm on Carvana competition Dataset from Kaggle using Sagemaker as Udacity's ML Nanodegree Capstone Project.

## Image Segmentation with U-Net Algorithm

Use AWS Sagemaker to train the model built with U-Net algorithm/architecture that can perform image segmentation on Carvana Dataset from Kaggle Competition.

## Project Set Up and Installation
Enter AWS through the gateway and create a Sagemaker notebook instance of your choice, `ml.t2.medium` is a sweet spot for this project as we will not use the GPU in the notebook and will use the Sagemaker Container to train the model. Wait for the instance to launch and then create a jupyter notebook with `conda_pytorch_latest_p36` kernel, this comes preinstalled with the needed modules related to pytorch we will use along the project. Set up your sagemaker roles and regions.

## Dataset
We use the **Carvana Dataset** from Kaggle Competition to use as data for the model training job. To get the Dataset. Register or Login to your Kaggle account, create new api in the user setting and get the api key and put it in the root of your sagemaker environment root location. 
After that `!kaggle competitions download carvana-image-masking-challenge -f train.zip` and 
`!kaggle competitions download carvana-image-masking-challenge -f train_masks.zip` will download the necessary files to your notebook environment. We will then unzip the data, upload it to S3 bucket with `!aws s3 sync` command.

## Script Files used
1. `hpo.py` for hyperparameter tuning jobs where we train the model for multiple time with different hyperparameters and search for the best combination based on loss metrics.
2. `training.py` for the final training of the model with the best parameters getting from the previous tuning jobs, and put debug and profiler hooks for debugging purpose and get the tensors emits during training.
3. `inference.py` for using the trained model as inference and pre-processing and serializing the data before it passes to the model for segmentaion. **Now this can be used locally and user friendly**
4. **Note** at this time, the sagemaker endpoint has an error and can't make prediction, so I have managed to create a new instance in sagemaker(`ml.g4dn.xlarge` to utilize the GPU) and used `endpoint_local.ipynb` notebook to get the inference result.
5. `requirements.txt` is use to install the dependencies in the training container, these include Albumentations, higher version of torch dependencies to utilize in the training script.

## Hyperparameter Tuning
I used U-Net Algorithm to create an image segmentation model.
The hyperparameter searchspaces are learning-rate, number of epochs and batchsize.
*Note* The batch size over 128(inclusive) can't be used as the GPU memory may run out during the training.
Deploy a hyperparameter tuning job on sagemaker and wait for the combination of hyperparameters turn out with best metric.

![hyperparameter tuning job](https://github.com/htinaunglu/UNET-Image-Segmentation/blob/main/images/HPO_Training_Jobs.png)

We pick the hyperparameters from the *best* training job to train the final model.

![best job's hyperparameters](https://github.com/htinaunglu/UNET-Image-Segmentation/blob/main/images/Best_Training_Job.png)


## Debugging and Profiling
The Debugger Hook is set to record the Loss Criterion of the process in both training and validation/testing.
The Plot of the *Dice Coefficient* is shown below. 

![Dice Coefficient](https://github.com/htinaunglu/UNET-Image-Segmentation/blob/main/images/Loss_Output_Plot.png)

we can see that the validation plot is high and this means that our model had entered a state of overtraining. We can reduce this by adding dropout or L1 L2 regularization, or added more different training data, or can early stop the model before it overfit.
by adding the metric definition, I could also managed to get the average accuracy and loss dat during the validation phase in AWS Cloudwatch(a powerful too to monitor your metrics of any kind).
![Metrics](https://github.com/htinaunglu/UNET-Image-Segmentation/blob/main/images/Validation%20Metrics.png)


### Results
Result is pretty good, as I was using ml.g4dn.xlarge to utilize the GPU of the instance, both the hpo jobs and training job did't take too much time.

## Inferenceing your data
Sagemaker Endpoint got an 500 status code error so I tried using another sagemaker instance with GPU(`ml.g4dn.xlarge`) and running the `endpoint_local.ipynb` will get you the desired output of your choice. 
![Result](https://github.com/htinaunglu/UNET-Image-Segmentation/blob/main/images/result_image.png)


**Thank You So Much For Your Time!**
*Please don't hesitate to contribute.*

Ref: [Github repo of neirinzaralwin](https://github.com/neirinzaralwin)
