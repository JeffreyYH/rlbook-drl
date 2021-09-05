import os 
import numpy as np
import matplotlib.pylab as plt

class TenArmedTestbed:
    def __init__(self):
        self.T = 1000
        self.num_run = 100

    def greedy(self, q_a_vec):
        q_a_vec = np.array(q_a_vec)
        a = np.argmax(q_a_vec)
        return a

    def ε_greedy(self, q_a_vec, ε):
        p = np.random.uniform(0, 1)
        if (p <= ε):
            a = np.random.choice(range(10))
        else:
            a = self.greedy(q_a_vec)
        return a

    def tenArmed_Testbed(self):
        r_all_vec = []
        for run_idx in range(self.num_run):
            r_vec = []
            for t in range(self.T):
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
                r_vec.append(r)
            r_all_vec.append(r_vec)
        # plot the results
        r_all_vec_np = np.array(r_all_vec)
        r_all_vec_ave = np.sum(r_all_vec_np, axis=0)/self.num_run
        plt.plot(r_all_vec_ave)
        plt.show()


if __name__ == "__main__":
    TenArmedTestbed().tenArmed_Testbed()