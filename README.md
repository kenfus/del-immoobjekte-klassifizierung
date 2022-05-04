# Deep Learning - Classification of Real Estate Objects
## Requirements 
There is a requirements.txt included, which can be used to install the used packages:

`pip install -r requirements.txt`.

However, a GPU should be used if possible, as one epoch took around 15 seconds on an RTX 3060Ti. 

## Data
The data contains the type of the real estate (House, flat, villa..) and municipality-specific data. We had solved this task as part of the `Deep Learning` course at (FHNW)[www.fhnw.ch]. 

## EDA
Eda was done in the notebook `eda.ipynb`. 

## Preprocessing
The preprocessing can be found as the class `PreProcessor` here: [helper_functions_preprocessing.py](https://github.com/kenfus/del-immoobjekte-klassifizierung/blob/master/helper_functions_preprocessing.py). In it you will find a Python class that preprocesses the data in ways needed for the MLP. These are described in more detail at the beginning of the notebook [main.ipynb](https://github.com/kenfus/del-immoobjekte-klassifizierung/blob/master/main.ipynb)

## Actual Mini-Challenge 
The actual solution to the mini-challenge can be found in the notebook [main.ipynb](https://github.com/kenfus/del-immoobjekte-klassifizierung/blob/master/main.ipynb) It has been constructed to correspond as closely as possible to the task defined in `mini-challenges_SGDS_DEL_MC1.pdf`. The model and how it was created can be found in [main.ipynb](https://github.com/kenfus/del-immoobjekte-klassifizierung/blob/master/main.ipynb) 

### Rough structure NN
This is an MLP with several hidden layers. The structure of the network was also searched by hyperparameter tuning. Per hidden layer, the number of neurons decreases.

### Early Stopping
It has "early stopping" built in. Training is interrupted if the F1 score does not improve for 10 epochs on the test set.

## Hyperparameter Tuning
Hyperparameter tuning was done using [Weights and Biases](https://wandb.ai/) and their "sweep" method [Bayesian Hyperparameter Optimization](https://wandb.ai/site/articles/bayesian-hyperparameter-optimization-a-primer). 

The entire optimization can be looked up here: [vincenzo293/DEL-mini-challenge-1](https://wandb.ai/vincenzo293/DEL-mini-challenge-1?workspace=user-vincenzo293). However, this is explained in more detail in the notebook [main.ipynb](https://github.com/kenfus/del-immoobjekte-klassifizierung/blob/master/main.ipynb)

# Interesting things and discoveries
## Difficult beginning
It was interesting that I had an extremely bad start. For a long time I had a Macro-F1 score of 0.21-0.23. At some point I noticed that the class did not standardize all values correctly and it had an attribute which took values between 30'000-120'000. When I fixed this (standardize), my Macro-F1 score went up to 0.37-0.398.

## Optimizer Adam could not handle the data / is set incorrectly
With SGD I was able to train the NN well. With Adam my F1 score never got above 0.17, even with testing many different learning rates (0.00001-0.001) and hyperparameter. This can also be seen in more detail in the notebook [main.ipynb](https://github.com/kenfus/del-immoobjekte-klassifizierung/blob/master/main.ipynb)

## Conclusion
In `Machine Learning Basics` for 4 ECTS at (FHNW)[www.fhnw.ch], we had to implemented the same task with a decision tree classifier (LightGBM) and it easily beat my MLP by reaching a score of 0.41 vs. 0.39.

# Futher details
Please have a look at [main.ipynb](https://github.com/kenfus/del-immoobjekte-klassifizierung/blob/master/main.ipynb)