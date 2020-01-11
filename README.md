[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![LinkedIn-profile](https://img.shields.io/badge/LinkedIn-profile-green.svg)](https://www.linkedin.com/in/unnikrishnan-menon-aa013415a/)

# BipedalWalker

This is how I taught a dumb bot to walk on two feet.

### Environment Specifications

Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.

### Algorithms
- [x] Deep Q Learning
- [x] NeuroEvolution of Augmenting Topologies
- [x] Deep Deterministic Policy Gradients

### Testing Instructions

```bash
$ git clone https://github.com/7enTropy7/BipedalWalker.git
$ cd BipedalWalker/DDPG
$ python3 test.py
```

### DDPG Agent Demo

After training for around 6 hours...

![ezgif com-video-to-gif (7)](https://user-images.githubusercontent.com/36446402/72131942-bbb35780-33a3-11ea-9db3-94651408fb92.gif)
