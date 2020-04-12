import numpy as np

class Policy():
    def __init__(self,num_observations,num_actions,lr,num_dirs,num_dirs_best,noise):
        self.theta = np.zeros((num_actions,num_observations))
        self.learning_rate = lr
        self.num_directions = num_dirs
        self.num_best_directions = num_dirs_best
        self.noise = noise

    def evaluate(self,state, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(state)
        elif direction == "positive":
            return (self.theta+self.noise*delta).dot(state)
        else:
            return (self.theta-self.noise*delta).dot(state)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for i in range(self.num_directions)]

    def update(self,rollouts,sigma_r):
        step = np.zeros(self.theta.shape)
        for positive_reward,negative_reward,d in rollouts:
            step += (positive_reward-negative_reward)*d
        self.theta += self.learning_rate/(self.num_best_directions * sigma_r)*step
