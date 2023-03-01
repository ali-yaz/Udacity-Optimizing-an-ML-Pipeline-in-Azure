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
**This dataset contains data about bank marketting we seek to predict if the client will subscribe to a term deposit with the bank. (y= Yes or No)

**The best performing model is the model with the higher acuuracy.**

## Scikit-learn Pipeline
**Our pipeline architecture, includes reading data, hyperparameter tuning, and classification. The data is first read as a tabular dataset, then it is split to a training and a testing dataset, we chose random sampling for parameter tuning and logistic regression as a classification algorithm.

**Random sampling supports early termination of low-performance runs. Also supports both discrete and continuous hyperparameters. In random sampling, hyperparameter values are randomly selected from the defined search space with two hyperparameters '--C' (Reqularization Strength) and '--max_iter' (Maximum iterations to converge). In this experiment the defined spaces are, -C (inversion of regularization strength): uniform (0.01, 1), ie, It returns values uniformly distributed between 0.01 and 1.00. -max-iter (maximum number of iterations): choice (100, 150, 200, 250, 300), ie, It returns a value chosen among given discrete values 100, 150, 200, 250, 300.**

**Early termination policy Bandit is an early termination policy that terminates any runs where the primary metric is not within the specified slack factor with respect to the best performing training run. In our case BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5). Slack factor refers to the slack allowed with respect to the best performing training run in ratio, ie, if the best metric run is less than 0.909, it will cancel the run. evaluation_interval specifies the frequency for applying the policy, ie, everytime the training script logs the primary metric count, policy is applied. delay_evaluation specifies the number of intervals to delay the policy evaluation. The policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation. In our case after 5 intervals the policy is delayed.**

## AutoML
**The best performing model was a model with an accuracy of aproximately 0.9177 .**

## Pipeline comparison
**
MaxAbsScaler LightGBM and TruncatedSVDWrapper RandomForest models showed the same best metric, accuracy =0.9158. However, the run time is different. one tool around 30 seconds , the TruncatedSVDWrapper RandomForest took around 4 minutes to be completed. 
**

## Future work
**In the case of scikit-learn based model, a different parameter sampler (Grid search or Bayesian search) can be used. **

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
