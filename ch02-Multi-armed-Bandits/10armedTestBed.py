import os 
import numpy as np
import matplotlib.pylab as plt

class TenArmedTestbed:
    def __init__(self):
        self.num_k = 10 # 10-armed bandits
        self.num_runs = 2000
        self.T = 1000
        self.method = "sample_average"
        self.q_true = []
        for k in range(self.num_k):
            q_a = np.random.normal(0, 1, 1)
            self.q_true.append(q_a)
        self.q_true = np.array(self.q_true)

    def ε_greedy(self, q_a_vec, ε):
        """ when ε=0, it is greedy """
        p = np.random.uniform(0, 1)
        if (p <= ε):
            a = np.random.choice(range(10))
        else:
            q_a_vec = np.array(q_a_vec)
            a = np.argmax(q_a_vec)
        return a
    
    def estimate_Q(self, t):
        """ 
        choise of method:
        1. sample_average
        2. incremental_update 
        """
        if (self.method == "sample_average"):
            return 1

    def simulate(self):
        R_allRun = []
        for run in range(self.num_runs):
            q_estimated = np.zeros(self.num_k)
            a_times = np.zeros(self.num_k)
            R_thisRun = []
            r = 0
            r_sum = 0
            for t in range(self.T):
                a = self.ε_greedy(q_estimated, 0)
                # get rewards
                r = np.random.normal(self.q_true[a], 1, 1)
                R_thisRun.append(r[0])
                # update estimated q
                r_sum += r
                a_times[a] += 1
                q_estimated[a] = r_sum/a_times[a]
            R_allRun.append(R_thisRun)
        R_allRun_np = np.array(R_allRun)
        # plot results
        R_allRun_ave_np = np.sum(R_allRun_np, axis=0)/self.num_runs
        plt.plot(R_allRun_ave_np)
        plt.show()
        
if __name__ == "__main__":
    TenArmedTestbed().simulate()


