[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) 
[![forthebadge](https://forthebadge.com/images/badges/fuck-it-ship-it.svg)](https://forthebadge.com)

[![Python 3.6](https://img.shields.io/badge/python-3.6-teal.svg)](https://www.python.org/downloads/release/python-360/) [![LinkedIn-profile](https://img.shields.io/badge/LinkedIn-Unnikrishnan-green.svg)](https://www.linkedin.com/in/unnikrishnan-menon-aa013415a/) [![GitHub followers](https://img.shields.io/github/followers/7enTropy7?label=Follow&style=social)](https://github.com/7enTropy7?tab=followers) [![GitHub stars](https://img.shields.io/github/stars/7enTropy7/BipedalWalker.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/7enTropy7/BipedalWalker/stargazers/)

# BipedalWalker

This is how I taught a dumb bot to walk on two feet.

Bipedal Walker is an <a href="https://openai.com/systems/">OpenAI Gym</a> environment where an agent learns to control a bipedal 2D character to reach the end of an obstacle course. What makes this challenging is that the agent only receives limbs coordinates along with the Lidar information. The agent has to learn to balance,walk,run,jump on its own without any human intervention.

## Environment Specifications

Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.

## Algorithms
- [x] Deep Q Learning
- [x] NeuroEvolution of Augmenting Topologies
- [x] Deep Deterministic Policy Gradients
- [x] Augmented Random Search

## Cloning
```bash
$ git clone https://github.com/7enTropy7/BipedalWalker.git
```

## Directory Contents
```bash
$ cd BipedalWalker/
$ tree
.
├── ARS
│   ├── main.py
│   ├── normalizer.py
│   ├── policy.py
│   └── Runs
│       ├── openaigym.video.0.31413.video032000.meta.json
│       └── openaigym.video.0.31413.video032000.mp4
├── DDPG
│   ├── actor_checkpoint.pth
│   ├── critic_checkpoint.pth
│   ├── __pycache__
│   │   └── walker.cpython-37.pyc
│   ├── test.py
│   ├── train.py
│   └── walker.py
├── DQN
│   ├── ai.py
│   ├── Bipedal-dqn-testing.h5
│   └── main.py
├── NEAT
│   ├── config-bipedal-walker.txt
│   └── neat_walker.py
├── README.md
└── requirements.txt

6 directories, 18 files
```

## Installing Dependencies

```bash
$ pip3 install -r requirements.txt
```

## Testing Instructions

```bash
$ cd BipedalWalker/ARS/
$ python3 main.py
```

## Outputs

### Augmented Random Search

Trained for around 4 hours. This is Perfection!!

![ezgif com-video-to-gif (11)](https://user-images.githubusercontent.com/36446402/79070954-7a11b000-7cf6-11ea-8160-fb86e0a174e1.gif)


### Deep Deterministic Policy Gradients

After training for around **2000 episodes** (6 hours of training on my low-end CPU).

![ezgif com-video-to-gif (8)](https://user-images.githubusercontent.com/36446402/72218920-65dadd00-3566-11ea-9321-6e478e0310fb.gif)

***
## Author
[![LinkedIn-profile](https://img.shields.io/badge/LinkedIn-Profile-teal.svg)](https://www.linkedin.com/in/unnikrishnan-menon-aa013415a/)
* [**Unnikrishnan Menon**](https://github.com/7enTropy7) 
