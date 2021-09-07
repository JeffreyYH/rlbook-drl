import os
from matplotlib.pyplot import xlabel 
import numpy as np
import matplotlib.pylab as plt

class TenArmedTestbed:
    def __init__(self):
        self.num_k = 10 # 10-armed bandits
        self.num_runs = 2000
        self.T = 1000
        self.method = "sample_average"
        # self.q_true = np.random.randn(self.num_k)
        self.q_true = np.random.normal(0, 1, self.num_k)

    def ε_greedy(self, q_estimated, ε):
        """ when ε=0, it is greedy """
        p = np.random.uniform(0, 1)
        if (p < ε):
            return np.random.choice(range(self.num_k))
        else:
            return np.argmax(np.array(q_estimated))
        
    def estimate_Q(self, t):
        """ 
        choise of method:
        1. sample_average
        2. incremental_update 
        """
        if (self.method == "sample_average"):
            return 1

    def simulate(self, ε_list):
        for ε in ε_list:
            R_allRun = []
            ifOptimalAction_allRun = []
            for run in range(self.num_runs):
                q_estimated = np.zeros(self.num_k)
                a_times = np.zeros(self.num_k)
                r_sum = np.zeros(self.num_k)
                R_thisRun = []
                ifOptimalAction = []
                for t in range(self.T):
                    a = self.ε_greedy(q_estimated, ε)
                    # get rewards
                    r = np.random.normal(self.q_true[a], 1)
                    R_thisRun.append(r)
                    # update estimated q
                    # method: sample_average
                    r_sum[a] += r
                    a_times[a] += 1
                    q_estimated[a] = r_sum[a]/a_times[a]
                    # see how good the policy is, in terms of picking optimal action
                    a_optimal = np.argmax(self.q_true)
                    ifOptimalAction.append(a_optimal == a)
                R_allRun.append(R_thisRun)
                ifOptimalAction_allRun.append(ifOptimalAction)
            # get final average reward, percentage of optimal action
            R_allRun_np = np.array(R_allRun) 
            R_allRun_ave_np = np.sum(R_allRun_np, axis=0)/self.num_runs  
            ifOptimalAction_allRun_np = np.array(ifOptimalAction_allRun)  
            OptimalAction_percentage = (np.sum(ifOptimalAction_allRun_np, axis=0)/self.num_runs) * 100 

            # plot fig. 2.2
            plt.subplot(2,1,1)
            plt.plot(R_allRun_ave_np, label=("ε = %.2f" %ε))
            plt.xlabel("steps")
            plt.ylabel("average rewards")
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(OptimalAction_percentage, label=("ε = %.2f" %ε))
            plt.xlabel("steps")
            plt.ylabel("% Optimal action")
            plt.legend()
        plt.show()
            
if __name__ == "__main__":
    ε_list = [0, 0.01, 0.1]
    TenArmedTestbed().simulate(ε_list)