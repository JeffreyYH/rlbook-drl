import time
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
    
    def UCB_actionSelection(self, q_estimated, t, N_a):
        c = 2.0
        return np.argmax(q_estimated + c*np.sqrt(np.log(t+1)/(N_a+1e-9)))

    def run_bandits(self, ε, update_method, action_selection, init_method):
        """ 
        update_method: "sample_average", "incremental"
        action_selection: "ε_greedy", "UCB"
        init_method: "optimistic", "realistic"
        """
        R_allRun = []
        ifOptimalAction_allRun = []
        for run in range(self.num_runs):
            # init of estimation Q
            if init_method == "realistic":
                q_estimated = np.zeros(self.num_k)
            elif init_method == "optimistic":
                mul = 5
                q_estimated = mul * np.ones(self.num_k)
            N_a = np.zeros(self.num_k)
            r_sum = np.zeros(self.num_k)
            R_thisRun = []
            ifOptimalAction = []
            for t in range(self.T):
                # get action
                if action_selection == "ε_greedy":
                    a = self.ε_greedy(q_estimated, ε)
                elif action_selection == "UCB":
                    a = self.UCB_actionSelection(q_estimated, t, N_a)
                # get rewards
                r = np.random.normal(self.q_true[a], 1)
                R_thisRun.append(r)
                # q estimation updating method
                if update_method == "sample_average":
                    r_sum[a] += r
                    N_a[a] += 1
                    if N_a[a] == 0:
                        q_estimated[a] = 0
                    else:
                        q_estimated[a] = r_sum[a]/N_a[a]
                elif update_method == "incremental":
                    if N_a[a] == 0:
                        q_estimated[a] = 0
                    else:
                        q_estimated[a] = q_estimated[a] + (r - q_estimated[a])/N_a[a]
                # see how good the policy is, in terms of picking optimal action
                a_optimal = np.argmax(self.q_true)
                ifOptimalAction.append(a_optimal == a)
            R_allRun.append(R_thisRun)
            ifOptimalAction_allRun.append(ifOptimalAction)
        # get final average reward, percentage of optimal action
        R_allRun_np = np.array(R_allRun) 
        self.R_allRun_ave_np = np.sum(R_allRun_np, axis=0)/self.num_runs  
        ifOptimalAction_allRun_np = np.array(ifOptimalAction_allRun)  
        self.OptimalAction_percentage = (np.sum(ifOptimalAction_allRun_np, axis=0)/self.num_runs) * 100 

    def test_ε_greedy_sampleAverage(self):
        """ get the results shown in figure 2.2"""
        ε_list = [0, 0.01, 0.1]
        for ε in ε_list:
            self.run_bandits(ε, "sample_average", "ε_greedy", "realistic")
            # plot fig. 2.2
            plt.subplot(2,1,1)
            plt.plot(self.R_allRun_ave_np, label=("ε = %.2f" %ε))
            plt.xlabel("steps")
            plt.ylabel("average rewards")
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(self.OptimalAction_percentage, label=("ε = %.2f" %ε))
            plt.xlabel("steps")
            plt.ylabel("% Optimal action")
            plt.legend()
        plt.show()
    
    def test_incrementalUpdate(self):
        """ section 2.4 """
        update_methods = ["sample_average", "incremental"]
        for update_method in update_methods:
            begin_time = time.time()
            self.run_bandits(0.1, update_method, "ε_greedy", "realistic")
            end_time = time.time()
            time_used = end_time - begin_time
            print("Runing time of update method %s is %f" %(update_method, time_used))

    def test_optimisticInitialValues(self):
        """ section 2.6, figure 2.3 """
        init_methods = ["optimistic", "realistic"]
        ε_list = [0.0, 0.1]
        for i in range(len(init_methods)):
            self.run_bandits(ε_list[i], "sample_average", "ε_greedy", init_methods[i])
            plt.plot(self.OptimalAction_percentage, label=("%s, ε = %.2f" %(init_methods[i], ε_list[i])))
            plt.xlabel("steps")
            plt.ylabel("% Optimal action")
            plt.legend()
        plt.show()

    def test_UCB(self):
        """ section 2.7, figure 2.4 """
        action_selections = ["ε_greedy", "UCB"]
        for action_selection in action_selections:
            self.run_bandits(0.1, "sample_average", action_selection, "realistic")
            plt.plot(self.R_allRun_ave_np, label=("%s" %action_selection))
            plt.xlabel("steps")
            plt.ylabel("% Optimal action")
            plt.legend()
        plt.show()

if __name__ == "__main__":
    TenArmedTestbed().test_ε_greedy_sampleAverage()
    TenArmedTestbed().test_incrementalUpdate()
    TenArmedTestbed().test_optimisticInitialValues()
    TenArmedTestbed().test_UCB()