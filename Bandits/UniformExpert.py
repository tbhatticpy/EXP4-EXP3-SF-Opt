import math
import random
import numpy as np

SF = np.array([7,8,9,10,11,12]) #Actions vector


def UniformExpert(packet_success):


    param_actions = SF.size #Number of actions
    param_gamma = 0.1 #Learning rate
    probs = np.zeros((SF.shape))
    rewards = np.zeros((SF.shape))
    uniform_weights_init = np.ones((SF.shape)) 
    uniform_weights_update= np.zeros((SF.shape))
    uniform_generated_reward = np.zeros((SF.shape))


    for i in range(param_actions): #Randomly generated initial rewards
        rewards[i] = random.uniform(0, 1)
    print(f"Initial rewards vector: {rewards}")
    print(f"Initial weights: {uniform_weights_init}")
    
    for t in range(5):

        W_t = np.sum(uniform_weights_init)
        for i in range(param_actions):
            probs[i] = (1-param_gamma)*(uniform_weights_init[i]/W_t) + param_gamma/param_actions
        print(f"Probability vector of Expert 3 is: {probs}")
        
        rec_index = np.argmax(probs) #Highest probability index
        rec_SF = SF[rec_index] #Best action for time step t
        if packet_success == True:
            received_reward = rewards[rec_index] #Achieved reward
            print(f"Recommended SF by Expert 3 is: {rec_SF}")
            
        else:
            received_reward = rewards[rec_index] - 1
        
        uniform_generated_reward[rec_index] = received_reward/probs[rec_index] #Fake reward generated 

        for i in range(param_actions):
            uniform_weights_update[i] = uniform_weights_init[i]*math.exp((param_gamma* uniform_generated_reward[i])/param_actions) #Weights update
        uniform_weights_init = uniform_weights_update
        rewards =  uniform_generated_reward
        SF_prob3 = probs


UniformExpert(True)