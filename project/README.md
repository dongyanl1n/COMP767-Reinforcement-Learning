# Actor Critic Network on TUNL simulated environment

In this project, we simulate an environment based on the Trial-Unique, Delayed Nonmatch-to-Location (TUNL) (Talpos et al., 2010) experiment setup, and an actor-critic network template where you can insert linear or LSTM hidden layers. We compare the performance of two neural networks: one with two linear hidden layers, the other one has one LSTM hidden layer followed by a linear hidden layer.The performance of a network is plotted in three graphs: the cumulative return across 500 episodes; the average return per episode; the acual return of each episode. Each experiment is run 10 times, the average and one standard deviation of the three criteria are plotted.

There are three versions of the simulated TUNL task:
* *SimpleMemoryTunlEnv*: two possible samples, drawn randomly. There is a delay period between receiving sample and making a decision.
* *SimpleTunlEnv*: two possible samples, drawn randomly. No delay period between receiving sample and making a decision.
* *SimpleTunlEnv_OneInput*: one possible samples. No delay period between receiving sample and making a decision.


## Prerequisites
```
pip install torch
pip install gym
pip install numpy
pip install matplotlib
```
