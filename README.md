# rl_ppo
The repositary consists of custom implementation of PPO agent, Atari wrapper, training scripts, experiment results + trained models.

PPO agent is able to:
- collect data via multiprocessing
- compute target values by GAE (Generalized Advantage Estimate)
- predict values by head support
- explore state space by using Random Network Distillation module

I tuned PPO agent in domain of Atari games, specially on Pong, Breakout, MsPacman and Montezuma's Revenge. My students and colleagues have also used my PPO agent in another domains like Starcraft and Doom mini games, card game Dominion and p-median optimalization problem. However, most of these experiments have not been allowable yet. :)

Results of PPO:

![Alt text](https://github.com/marrekb/rl_ppo/blob/main/plots/pong_score.png?raw=true "Title")

Few advantages in research:
- In Atari domain, othogonal initialization of model is better than Xavier one.
- Adam optimizer is good option... almost everytime. 
- In Atari domain, learning rate set to 0.00025 is good choice in most of environmnets.
- On GPU envs, PPO without multiprocessing is better otpion.
- I tried to use new reward head as a part of the predicted model in order to support learning. However, this modification approxiamted worse than baseline. Don't repeat my mistake. :)

There are many improvements and agents that have reached SOTA results in current research. Of course, I tried a lot of these improvements and agents. Although you are able to reach better score, dont' forget: your training will prolong many times. For example, I implemented efficient MuZero algorithm but the training in pong environment lasts about one day (PPO reach max score in one hour).


