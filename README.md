# Reinforcement Learning for Container Stowage Planning (CSPP)

This repository implements the container transportation problem using the CleanRL framework.  
The codebase serves as the main artifact, while the associated paper provides background and reference.

## Running the Code

The main entry point is:

ppo.py

Running this file will start training using the multi-crane environment (not the single-crane setting).

python ppo.py

## Understanding the Code

The most important files in the repository are:

ppo.py  
env/stowage_gym.py  
env/stowage_crane_gym.py  

Before attempting any modifications, please make sure you fully understand the MDP formulation, including:

- state representation
- action space
- reward function
- state transitions
- the meaning of each dimension

## Suggested Improvements

Currently, training and testing are not separated. Evaluation results are printed during training.  
You may improve the implementation by separating the training and testing processes.

For example:

- periodically save the agent parameters during training
- load the final trained model for independent evaluation
- visualize the results (e.g., reward curves, crane operating time)
