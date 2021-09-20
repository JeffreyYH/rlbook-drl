import time
from matplotlib.pyplot import xlabel 
import numpy as np
import matplotlib.pylab as plt
import random
import argparse

class TenArmedTestbed:
    def __init__(self):
        self.num_k = 10 # 10-armed bandits
        self.num_runs = 20
        self.T = 1000

    def ε_greedy(self, q_estimated, ε):
        """ when ε=0, it is greedy """
        p = np.random.uniform(0, 1)
        if (p < ε):
            return np.random.choice(range(self.num_k))
        else:
            return np.argmax(np.array(q_estimated))
    
    def UCB_actionSelection(self, q_estimated, c, t, N_a):
        return np.argmax(q_estimated + c*np.sqrt(np.log(t+1)/(N_a+1e-9)))        

    def run_bandits(self, update_method, action_selection, init_method):
        """ 
        update_method: "sample_average", "incremental"
        action_selection[0] dentotes names: "ε_greedy", "UCB", "Boltzmann", "gradient_baseline", "gradient_noBaseline"
        action_selection[0] dentotes parameters: ε,       c,    temperature,    ...,                 ...
        init_method[0] denotes name: "optimistic", "realistic"
        init_method[1] denotes value: some specific value, none (0 by default)
        """
        R_allRun = []
        expected_R_allRun = []
        ifOptimalAction_allRun = []
        for run in range(self.num_runs):
            ## initialization
            self.q_true = np.random.normal(0, 1, self.num_k)
            # init of estimation Q
            if init_method[0] == "realistic":
                q_estimated = np.zeros(self.num_k)
            elif init_method[0] == "optimistic":
                mul = init_method[1]
                q_estimated = mul * np.ones(self.num_k)
            N_a = np.zeros(self.num_k)
            r_sum = np.zeros(self.num_k)
            H = np.ones(self.num_k)
            possible_actions = range(self.num_k)
            π = [0]*self.num_k
            R_thisRun = []
            expected_R_thisRun = []
            ifOptimalAction = []

            ## run T steps
            for t in range(self.T):
                # get action
                if action_selection[0] == "ε_greedy":
                    ε = action_selection[1]
                    a = self.ε_greedy(q_estimated, ε)
                elif action_selection == "UCB":
                    c = action_selection[1]
                    a = self.UCB_actionSelection(q_estimated, c, t, N_a)
                elif action_selection == "gradient_baseline" or action_selection == "gradient_noBaseline":
                    for a in range(self.num_k):
                        π[a] = np.exp(H[a]) / np.sum(H)
                    a = random.choices(possible_actions, π)[0]
                # get rewards
                r = np.random.normal(self.q_true[a], 1)
                R_thisRun.append(r)
                N_a[a] += 1
                ## some updates for gradient based method
                if action_selection[0] == "gradient_baseline" or action_selection[0] == "gradient_noBaseline":
                    α = action_selection[1]
                    if t <= 1:
                        r_ave = 0
                    else:
                        R_prevRun = R_thisRun
                        R_prevRun.pop()
                        r_ave = sum(R_prevRun)/(t-1)
                    if action_selection == "gradient_baseline":
                        H = H - α*(r - r_ave) * np.array(π)
                        H[a] = H[a] + α*(r - r_ave) * (1 - π[a])
                    if action_selection == "gradient_noBaseline":
                        H = H - α * r * np.array(π)
                        H[a] = H[a] + α * r * (1 - π[a])
                ## q estimation updating method
                if update_method == "sample_average":
                    r_sum[a] += r
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
                
                # get expected reward
                if action_selection[0] == "ε_greedy":
                    expected_r =  (1-ε)*self.q_true[np.argmax(q_estimated)]  + ε*np.mean(self.q_true)
                    expected_R_thisRun.append(expected_r)

            if action_selection[0] == "ε_greedy":
                expected_R_allRun.append(expected_R_thisRun)
            R_allRun.append(R_thisRun)
            ifOptimalAction_allRun.append(ifOptimalAction)

        ## get final average reward, percentage of optimal action
        self.R_allRun_ave_np = np.sum(np.array(R_allRun) , axis=0)/self.num_runs  
        self.expected_R_allRun_ave_np = np.sum(np.array(expected_R_allRun) , axis=0)/self.num_runs  
        self.OptimalAction_percentage = (np.sum(np.array(ifOptimalAction_allRun), axis=0)/self.num_runs) * 100 


    def test_ε_greedy_sampleAverage(self):
        """ get the results shown in section 2.3, figure 2.2"""
        ε_list = [0, 0.01, 0.1]
        init_method = ["realistic"]
        for ε in ε_list:
            action_section = ["ε_greedy", ε]
            self.run_bandits("incremental", action_section, init_method)
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
    
    def test_expectedRewards(self):
        ε_list = [0, 0.001, 0.01, 0.1, 1.0]
        init_method = ["realistic"]
        for ε in ε_list:
            action_section = ["ε_greedy", ε]
            self.run_bandits("incremental", action_section, init_method)
            # plot fig. 2.2
            plt.subplot(2,1,1)
            plt.plot(self.R_allRun_ave_np, label=("ε = %.2f" %ε))
            plt.xlabel("steps")
            plt.ylabel("average rewards")
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(self.expected_R_allRun_ave_np, label=("ε = %.2f" %ε))
            plt.xlabel("steps")
            plt.ylabel("average expected rewards")
            plt.legend()
        plt.show()

    def test_incrementalUpdate(self):
        """ section 2.4 """
        update_methods = ["sample_average", "incremental"]
        action_section = ["ε_greedy", 0.1]
        init_method = ["realistic"]
        for update_method in update_methods:
            begin_time = time.time()
            self.run_bandits("incremental", action_section, init_method)
            end_time = time.time()
            time_used = end_time - begin_time
            print("Running time of update method %s is %f" %(update_method, time_used))

    def test_optimisticInitialValues(self):
        """ section 2.6, figure 2.3 """
        init_methods = [["optimistic", 1.0], ["realistic"]]
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
        action_selections = [["ε_greedy", 0.1], ["UCB", 2.0]]
        init_method = ["realistic"]
        for action_selection in action_selections:
            self.run_bandits("sample_average", action_selection, init_method)
            plt.plot(self.R_allRun_ave_np, label=("%s" %action_selection))
            plt.xlabel("steps")
            plt.ylabel("% Optimal action")
            plt.legend()
        plt.show()

    def test_gradientBandits(self):
        # TODO: fix some numeric issues
        # FIXME: buggy plot
        α_list = [0.1, 0.4]
        action_selections = ["gradient_baseline", "gradient_noBaseline"]
        for action_selection in action_selections:
            for α in α_list:
                self.run_bandits(0.1, "sample_average", action_selection, α, "realistic")
                plt.plot(self.OptimalAction_percentage, label=("%s, %s" %(action_selection, α)))
                plt.xlabel("steps")
                plt.ylabel("% Optimal action")
                plt.legend()
        plt.show()


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        default='LunarLander-v2', help="Name of the environment") # CartPole-v0, LunarLander-v2
    parser.add_argument('--num_episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.") #50000
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # TenArmedTestbed().test_ε_greedy_sampleAverage()
    TenArmedTestbed().test_expectedRewards()
    # TenArmedTestbed().test_incrementalUpdate()
    # TenArmedTestbed().test_optimisticInitialValues()
    # TenArmedTestbed().test_UCB()
    # TenArmedTestbed().test_gradientBandits()