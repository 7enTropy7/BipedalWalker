import neat
import gym
import numpy as np
import pickle
import os

env = gym.make('BipedalWalker-v2')

def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main, n=1500) # n is no. of generations

def main(genomes,config):
    for genome_id,genome in genomes:

        genome.fitness = 0

        net = neat.nn.FeedForwardNetwork.create(genome,config)
        
        total_reward = 0.0

        observation = env.reset()

        done = False

        render = False

        t = 0

        r = 0
        for k in range(5):
            while not done:
                if genome_id > 100000:
                    env.render()

                action = net.activate(observation)
                action = np.clip(action,-1,1)

                observation, reward, done, info = env.step(action)
             
                t += 1
                r += reward

                if done:
                    total_reward += r
                    #print(total_reward)
                    #print("{} <-- Episode _________ {} timesteps __________ reward {} __________ Highest reward : {}".format(genome_id,t + 1, reward,max_reward))
                    #if total_reward > 1: render = True
                    env.reset()
                    break
        genome.fitness = total_reward / 5
        print("Genome : {}  Fitness : {}".format(genome_id,genome.fitness))


if __name__=="__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,'config-bipedal-walker.txt')
    run(config_path)
