# Required Modules
  ### 1.Torch
  ### 2.Torch-Geometric
  ### 3.Numpy
  ### 4.tqdm

# Training and Evaluation/Test
Run the app.py file with python command
  #### python app.py
# Method and Algorithm
#### RGCN is used to encode the head and tail entities and relations are encoded with a simple embedding lookup
#### The Links that are to be predicted are removed from the message passing graph and the possibility of the links being present
#### are calculated by a scoring function similar to the DistMult with the difference that this scoring function, at least
#### in theory, can model anti-symmetric relations. 
#### At test time, the training set and the validation set are concattenated together to be the message passing graph
#### This does not lead to data leakage as this is standard procedure link prediction operation with Graph Neural Networks.
#### No hyperparameter search has been done at the time of this writing due to a lack of time :)
#### However, model hyperparameters, including the size of the vectors, and the values in the Global class can all be
#### tuned.
#### Uniform sampling was used for the negative triplet sampling, but the code also supports for multinomal sampling.
