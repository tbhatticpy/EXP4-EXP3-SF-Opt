import math
import random
import numpy as np

SF = np.array([7,8,9,10,11,12]) #Actions vector
SF_prob1 = np.zeros((SF.shape))
SF_prob2 = np.zeros((SF.shape))
SF_prob3 = np.zeros((SF.shape))
def EABExpert(distance): #First expert which recommends SF based on area
    SF_index = math.floor(distance / 1000) #SF changes for every 1000m increase in a 6km range
    rec_SF = SF[SF_index]
    for i in range(SF.size):
        if i == SF_index:
            SF_prob1[i] = 1 #Probability vector of expert 1
    print(f"Recommended SF by Expert 1 is: {rec_SF}")
    print(f"Probability vector by Expert 1 is: {SF_prob1}") 
 
def TOAExpert(time): #Second expert which recommends SF based on area
    SF_index = math.floor(time / 30) #SF changes for every 30s increase air time
    rec_SF = SF[SF_index]
    for i in range(SF.size):
        if i == SF_index:
            SF_prob2[i] = 1 #Probability vector of expert 2
    print(f"Recommended SF by Expert 2 is: {rec_SF}")
    print(f"Probability vector of Expert 2 is: {SF_prob2}")

def UniformExpert(param_gamma,param_actions,probs,rewards,uniform_weights_init,uniform_weights_update,uniform_generated_reward,packet_success):
    
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



def exp4():
    #Initializations
    packet_success = True
    param_experts = 3 #Number of experts
    param_actions = SF.size #Number of actions
    param_gamma = 0.1 #Learning rate
    param_T = 5 #Time horizon
    probs = np.zeros((SF.shape))
    rewards = np.zeros((SF.shape))
    print(f"Learning rate: {param_gamma}")
    generated_reward = np.zeros((SF.shape))
    expert_gain = np.zeros((1,param_experts))
    weights_init = np.ones((1,param_experts))
    weights_update= np.zeros((1,param_experts))

    uniform_weights_init = np.ones((SF.shape)) 
    uniform_weights_update= np.zeros((SF.shape))
    uniform_generated_reward = np.zeros((SF.shape))

    EABExpert(5000)
    TOAExpert(120)

    for i in range(param_actions): #Randomly generated initial rewards
        rewards[i] = random.uniform(0, 1)
    print(f"Initial rewards vector: {rewards}")
    print(f"Initial weights: {weights_init}")


 #Time loop
 
    for t in range(param_T):
        UniformExpert(param_gamma,param_actions,probs,rewards,uniform_weights_init,uniform_weights_update,uniform_generated_reward,packet_success)
        mtx_expert = np.array([SF_prob1,SF_prob2, SF_prob3])
        W_t = np.sum(weights_init) #Sum of weights  
        for j in range(param_actions):
            #probs[j] = (1-param_gamma)*np.sum((np.dot(weights_init,mtx_expert)/W_t)+(param_gamma/param_actions))
            temp_val = [0,0,0]
            for i in range(param_experts):
                temp_val[i] = weights_init[0][i]*mtx_expert[i][j]
            tem_val_sum = np.sum(temp_val)
            probs[j] = (1-param_gamma)*(tem_val_sum/W_t)+(param_gamma/param_actions) #Probability vector calculation
        print(f"Probability vector: {probs}")
        rec_index = np.argmax(probs) #Highest probability index
        rec_SF = SF[rec_index] #Best action for time step t
        received_reward = rewards[rec_index] #Achieved reward
        print(f"Recommended SF: {rec_SF}")
        print(f"Received reward: {received_reward}")
        generated_reward[rec_index] = received_reward/probs[rec_index] #Fake reward generated
        print(f"Generated rewards: {generated_reward}")
        print(f"Experts probability matrix: {mtx_expert}")

        for i in range(param_experts):
            expert_gain[0][i] = np.sum(np.multiply(generated_reward,mtx_expert[i]))
        print(f"Experts gain: {expert_gain}")

        for i in range(param_experts):
            weights_update[0][i] = weights_init[0][i]*math.exp((param_gamma*expert_gain[0][i])/param_actions) #Weights update
        print(f"Updated weights: {weights_update}")
        weights_init = weights_update
        rewards = generated_reward

exp4()
