import sys
import gym
if "../" not in sys.path: sys.path.append("../")

env = gym.make("Blackjack-v1")

def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

# policy: Stick (action 0) if the score is 20 or 21, hit (action 1) otherwise
def policy(observation):
    score, dealer_score, usable_ace = observation
    # DO NOT write as score == 20 or 21, condition 21 is always true
    if score == 20 or score == 21:
        return 0
    else:
        return 1

num_episode = 1000
num_win = 0
num_draw = 0
num_lose = 0
for i_episode in range(num_episode):
    observation = env.reset()
    for t in range(1000):
        print_observation(observation)
        action = policy(observation)
        print("Taking action: {}".format( ["Stick", "Hit"][action]))
        observation, reward, done, _ = env.step(action)
        if done:
            print_observation(observation)
            print("Game end. Reward: {}\n".format(float(reward)))
            if reward == 1:
                num_win += 1
            elif reward == 0:
                num_draw += 1
            elif reward == -1:
                num_lose += 1
            break

# showing statistics
winning_rate = num_win/num_episode
drawing_rate = num_draw/num_episode
losing_rate = num_lose/num_episode
print("Winning rate: %.2f" % winning_rate)
print("drawing rate: %.2f" % drawing_rate)
print("losing rate: %.2f" % losing_rate)