import math
import random
import numpy as np

SF = np.array([7,8,9,10,11,12])
SF_prob1 = np.zeros((SF.shape))
SF_prob2 = np.zeros((SF.shape))
def EABexpert(distance):
    SF_index = math.floor(distance / 1000)
    rec_SF = SF[SF_index]
    for i in range(SF.size):
        if i == SF_index:
            SF_prob1[i] = 1
    print(f"Recommended SF is: {rec_SF}")
    print(f"Probability vector is: {SF_prob1}")

def TOAexpert(time):
    SF_index = math.floor(time / 30)
    rec_SF = SF[SF_index]
    for i in range(SF.size):
        if i == SF_index:
            SF_prob2[i] = 1
    print(f"Recommended SF is: {rec_SF}")
    print(f"Probability vector is: {SF_prob2}")


class banditalgo(object):
    EABexpert(400)
    TOAexpert(90)
    N = SF.size
    param_gamma = random.uniform(0, 1)
    print(f"Gamma is: {param_gamma}")
    param_experts = 2
    param_actions = SF.size
    param_T = 1000
    mtx_expert = np.array([[SF_prob1],[SF_prob2]])
    print(mtx_expert)
    weights_init = np.ones((SF.shape))

    def __init__(self, param_gamma, param_experts, param_actions):
        self.param_gamma = param_gamma
        self.param_actions = param_actions
        self.weight_of_advices = np.ones(param_experts, dtype=np.int)
        self.weighted_reward = np.zeros(param_experts, dtype=np.int)
    
    def start_train(self, param_T, mtx_experts):
        self.count = 0
        self.param_T = param_T
        self.mtx_experts = mtx_experts
        selected_action, prob_selected_action = self._get_action(self.mtx_experts)
        return selected_action, prob_selected_action

    def iterate_reward(self, reward):
        """Please Run start_train before iterate_reward THX! 
        """
        if self.count <= self.param_T:
            self._update(selected_action, prob_selected_action, reward, self.mtx_experts)
            selected_action, prob_selected_action = self._get_action(self.mtx_experts)
            return selected_action, prob_selected_action
        else:
            return None, None
    def _get_action(self, mtx_experts):
        """
        Select an action based on the certain probability
        Args:
            mtx_experts (matrix) : experts advice matrix 
        Returns:
            selected_action        : the chosen action
            prob_selected_action   : the probability of the chosen action
        Examples:
            exp4._get_action(...)
        """
        # Maybe you should describe more on this part
        temp = np.dot(mtx_experts, self.weight_of_advices) * (1. / sum(self.weight_of_advices))
        prob_each_action = (1 - self.param_gamma) * temp + self.param_gamma / self.param_actions
        culmulative_prob = np.zeros(self.param_actions, dtype=np.int)
        val = rand.random()
        culmulative_prob[0] = prob_each_action[0]
        # Or more comments on this block
        for i in range(1, K): # K from where?
            culmulative_prob[i] = culmulative_prob[i-1] + prob_each_action[i]
        for j in range(K):
            if val < culmulative_prob[j]:
                selected_action = j
                break
        prob_selected_action = prob_each_action[selected_action]
        return selected_action, prob_selected_action

    def _update(self, selected_action, prob_selected_action, reward, mtx_experts):
        """Update weight of experts' advices.
        Args:
            param1 (int): The first parameter.
            param2 (str): The first parameter.
        Returns:
            bool: True if successful, False otherwise.
        Examples:
            exp4._get_action(...)
        """
        ind = selected_action
        param = float(self.param_gamma) / param_actions
        self.weighted_reward += (float(reward) / prob_for_each_arm[ind]) * mtx_expert[ind]
        self.weight_of_advices = np.exp(param * self.weighted_reward)
