# Pulse Optimisation Using Gymnasium Enviornment 
I created this enviorment for the public to easily able to implement any Reinforcement Learning algorithm for Optimisation of Qubit pulses. My main motivation for this work is this paper https://arxiv.org/pdf/2105.01079.pdf. This enviorment hopes to provide an easy playground for creating policy networks for the Qubit Pulses without worrying about the Physics involved. 

## Install

Do one of the following in the source directory (preferably in a Python virtual environment set up for Qiskit Dynamics)

* `make # gnu make, we have provided a Makefile`
* `./setup.py install`
* `pip3 install .`


# How to work with this repo

## Making Enviornment

```python
import gym
import Rx_Env
env = gym.make('Rx_Env/RxEnv-v0')
```

## Using Enviornment

Right now the enviornment returns 
1. observation: A dictionary of Start, Current, Target State
2. reward: Reward directly based on Fidelity, some bonus is given for fidelity greater than 0.999
3. done: To stop the episode after a set number of evolutions(20) achieved
4. truncated: To stop in between for some reason.
5. fidelity info: Returns a Dictionary of current Fidelity


# Future
Please raise an issue/Pr if you have any suggetions or would like to contribute.
Ideally I would like for this library to contain a default Policy Network good enough to give you the ideal pulse for implementatiion of any gate given its paramenters and a hardware without requiring any Physics or RL knowledge. I am currently working on creating such policy network for X gate. If I can get it working efficiently enough I shall move to creating a unviversal Enviornment/Policy Network. 


