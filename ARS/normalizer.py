import numpy as np

class Normalizer():
    def __init__(self,num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_difference = np.zeros(num_inputs)
        self.variance = np.zeros(num_inputs)

    def normalize(self,state):
        observation_mean = self.mean
        observation_standard_deviation = np.sqrt(self.variance)
        return (state-observation_mean)/observation_standard_deviation

    def observe(self,state):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (state - self.mean)/self.n
        self.mean_difference += (state-last_mean)*(state-self.mean)
        self.variance = (self.mean_difference/self.n).clip(min=1e-2)
