import numpy as np

class TabularUtils:
    def __init__(self, env):
        self.env = env


    def epsilon_greedy_policy(self, policy_s_or_Q_s, ε=0.2):
        """ 
        Creating epsilon greedy probabilities to sample from
        inputs: policy_s_or_Q_s, policy or q value at state s
        when ε=0, it is greedy 
        """
        p = np.random.uniform(0, 1)
        if (p < ε):
            a = np.random.choice(len(policy_s_or_Q_s))
        else:
            a = np.argmax(np.array(policy_s_or_Q_s))
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
    
    
    def Q_value_to_state_value(self, Q, policy):
        """ compute the state value function given action value function"""
        V = np.zeros(self.env.nS)
        for s in range(self.env.nS):
            V[s] = np.dot(Q[s, :], policy[s, :])
        
        return V
