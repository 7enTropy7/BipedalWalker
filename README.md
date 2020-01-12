[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![LinkedIn-profile](https://img.shields.io/badge/LinkedIn-profile-green.svg)](https://www.linkedin.com/in/unnikrishnan-menon-aa013415a/) [![GitHub followers](https://img.shields.io/github/followers/7enTropy7?label=Follow&style=social)](https://github.com/7enTropy7?tab=followers) [![GitHub stars](https://img.shields.io/github/stars/7enTropy7/BipedalWalker.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/7enTropy7/BipedalWalker/stargazers/)

# BipedalWalker

This is how I taught a dumb bot to walk on two feet.

### Environment Specifications

Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.

### Algorithms
- [x] Deep Q Learning
- [x] NeuroEvolution of Augmenting Topologies
- [x] Deep Deterministic Policy Gradients

### Installing Dependencies

```bash
$ pip3 install -r requirements.txt
```

### Testing Instructions

```bash
$ git clone https://github.com/7enTropy7/BipedalWalker.git
$ cd BipedalWalker/DDPG
$ python3 test.py
```

### DDPG Agent Demo

After training for around 2000 episodes...

![ezgif com-video-to-gif (8)](https://user-images.githubusercontent.com/36446402/72218920-65dadd00-3566-11ea-9321-6e478e0310fb.gif)

