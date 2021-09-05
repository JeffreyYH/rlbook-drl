import os 
import numpy as np
import matplotlib.pylab as plt

class TenArmedTestbed:
    def __init__(self):
        self.T = 1000
        self.num_run = 200

    def greedy(self, q_a_vec):
        q_a_vec = np.array(q_a_vec)
        a = np.argmax(q_a_vec)
        return a

    def ε_greedy(self, q_a_vec, ε):
        if (np.random.uniform(0, 1) <= ε):
            a = np.random.choice(range(10))
        else:
            a = self.greedy(q_a_vec)
        return a

    def tenArmed_Testbed(self):
        R_ave_vec = []
        for t in range(self.T):
            R_all = 0
            for run_idx in range(self.num_run):
                R = 0
                q_a_vec = []
                for i in range (10):  
                    num_sample = 1
                    # sample action value function q*(a)
                    q_a = np.random.normal(0.0, 1.0, num_sample)
                    q_a_vec.append(q_a)
                # determine action
                a = self.greedy(q_a_vec)
                # sample reward
                q_a_optimal = q_a_vec[a]
                r = np.random.normal(q_a_optimal, 1.0, num_sample)
                R += r[0]
                R_all += R
            R_ave = R_all/self.num_run
            R_ave_vec.append(R_ave)
        # plot the results
        plt.plot(R_ave_vec)
        plt.show()


if __name__ == "__main__":
    TenArmedTestbed().tenArmed_Testbed()