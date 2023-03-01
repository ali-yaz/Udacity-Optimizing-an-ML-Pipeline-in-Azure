# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**This dataset contains bank marketting data and the goal of the project was to predict if the client will subscribe to a term deposit with the bank. (y= Yes or No)

**The best performing model is the model with the higher acuuracy.**

## Scikit-learn Pipeline
Our pipeline architecture, includes reading data, hyperparameter tuning, and classification. The data is first read as a tabular dataset, then it is split to a training and a testing dataset, we chose random sampling for parameter tuning and logistic regression as a classification algorithm.**

Random sampling supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space with two hyperparameters '--C' (Reqularization Strength) and '--max_iter' (Maximum iterations to converge). 

Early termination policy Bandit is an early termination policy that terminates any runs where the primary metric is not within the specified slack factor with respect to the best performing training run. 

## AutoML
**The best performing model was a model with an accuracy of aproximately 0.9177 .
AutoML pipeline experimented different models, and VotingEnsemle algorithm demontrated the highest accuracy.
**

## Pipeline comparison
**
MaxAbsScaler LightGBM and TruncatedSVDWrapper RandomForest models showed the same best metric, accuracy =0.9158. However, the run time is different. one tool around 30 seconds , the TruncatedSVDWrapper RandomForest took around 4 minutes to be completed. 
**

## Future work
**In the case of scikit-learn based model, a different parameter sampler (Grid search or Bayesian search) can be used. 
We also can remove the 30min time limit in AutoML run to see if algorithms can achieve higher accuracy.
Another future work can be removing Bandit policy to see if models performance can be improved.
**

## Proof of cluster clean up

![image](https://user-images.githubusercontent.com/16668953/222142582-61035691-c238-4552-a8d3-9d54b06eceb8.png)

**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
