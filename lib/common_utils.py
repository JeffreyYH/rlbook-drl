import numpy as np

class TabularUtils:
    def __init__(self, env):
        self.env = env


    def epsilon_greedy_policy(self, q_values, ε=0.2):
        """ Creating epsilon greedy probabilities to sample from """
        """ when ε=0, it is greedy """
        p = np.random.uniform(0, 1)
        if (p < ε):
            a = np.random.choice(len(q_values))
        else:
            a = np.argmax(np.array(q_values))
        return a
    

    def action_to_onehot(self, a):
        """ convert single action to onehot vector"""
        a_onehot = np.zeros(self.env.nA)
        a_onehot[a] = 1
        return a_onehot


    def onehot_policy_to_deterministic_policy(self, policy_onehot):
        policy=np.zeros(self.env.nS)
        for s in range(policy_onehot.shape[0]):
            policy[s] = np.argmax(policy_onehot[s, :])
        
        return policy


    def Q_value_to_greedy_policy(self, Q):
        """ get greedy policy from Q value"""
        policy = np.zeros((self.env.nS, self.env.nA))
        for s in range(Q.shape[0]):
            a_greedy = np.argmax(Q[s, :])
            policy[s, :] = self.action_to_onehot(a_greedy)
        
        return policy
