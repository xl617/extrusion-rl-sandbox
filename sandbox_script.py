import os
import re
from matplotlib.pyplot import imshow
import torch
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics
import time
import torch.nn.init as init
from collections import namedtuple, deque
from collections import Counter
from itertools import combinations
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import subprocess
import shutil
import json
from Transition import Transition


min_temp = 190
max_temp = 230
min_fr = 30
max_fr = 250

probability_given_label = 1

firmware_state_size = 2


temp_action_interval = 10

change_interval = 5
change_interval1 = 10

temp_pid_interval = 3

max_transition = 100

previous_cls_num = 30

converge_plot = True
firmware_set_record = []
temp_set_record = []

 

flow_rate_ranges = [(min_fr-5, 90), (90, 110), (110, max_fr+5)]
temperature_ranges = [(185, 235)]



video_class_size = len(flow_rate_ranges) * len(temperature_ranges)




# Find the range index for flow rate and temperature
flow_rate_index = next((i for i, (start, end) in enumerate(flow_rate_ranges) if start <= 100 < end), None)
temperature_index = next((i for i, (start, end) in enumerate(temperature_ranges) if start <= 210 < end), None)

# Create label based on flow rate and temperature range indices
if flow_rate_index is not None and temperature_index is not None:
    label_target = (flow_rate_index+temperature_index*len(flow_rate_ranges))/len(flow_rate_ranges)
else:
    label_target = None

#print("label_target", label_target)

previous_cls =  np.ones(previous_cls_num) * label_target

#print("previous_cls: ", previous_cls)


state_size = video_class_size + firmware_state_size + previous_cls_num

#print("state_size: ", state_size, video_class_size, firmware_state_size, previous_cls_num)

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay memory for experience replay
#Transition = namedtuple(
#    "Transition", ("state", "action", "action1", "next_state", "reward", "done")
#)

action_range = 2  # the number of positive/negative values
action_range1 = 2
action_size = 2 * action_range + 1
action_size1 = 2 * action_range1 + 1

img_dir = "/home/cam/Server"

home_directory = "/home/cam/Documents/Server/Video-Swin-Transformer/rl_sim_checkpoint"


def save_DQN_checkpoint(policy_net, target_net, optimizer, episode, memory, position, memory1, position1, reward_record):

    print("save len(reward_record)", len(reward_record))
    
    file_name = f"{episode}.pth"  # Change the extension if needed

    # Join the home directory and the file name to create the full file path
    file_path = os.path.join(home_directory, file_name)
    checkpoint = {
        "episode": episode,
        "policy_net_state_dict": policy_net.state_dict(),
        "target_net_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "memory": memory,
        "memory_pos": position,
        "memory1": memory1,
        "memory_pos1": position1,
        "cumulative_reward_episode": reward_record,
        #'Transition': Transition,
        #"total_loss_deep_Q": total_loss_record,
        #"flowrate_record": firmware_record,
        #"flowrate_record_1": firmware_record_1,
    }
    torch.save(checkpoint, file_path)


def load_DQN_checkpoint(policy_net, target_net, optimizer, episode):
    
    file_name = "2500.pth"  #f"{episode}.pth"  # Change the extension if needed
    # /home/cam/Documents/Server/Video-Swin-Transformer/rl_sim_checkpoint/2500.pth
    # Join the home directory and the file name to create the full file path
    file_path = os.path.join(home_directory, file_name)
    checkpoint = torch.load(file_path, map_location=device)
    print(checkpoint.keys())
    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    target_net.load_state_dict(checkpoint["target_net_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    episode = checkpoint["episode"]
    init_memory = checkpoint["memory"]
    init_memory_pos = checkpoint["memory_pos"]
    init_memory1 = checkpoint["memory1"]
    init_memory_pos1 = checkpoint["memory_pos1"]
    reward_record = checkpoint["cumulative_reward_episode"]
    #Transition = checkpoint['Transition']

    print("load len(reward_record)", len(reward_record))
    #total_loss_record = checkpoint["total_loss_deep_Q"]
    #firmware_record = checkpoint["flowrate_record"]
    #firmware_record_1 = checkpoint["flowrate_record_1"]

    #print(checkpoint["cumulative_reward_episode"])
    #print(checkpoint["total_loss_deep_Q"])

    return policy_net, target_net, optimizer, episode, init_memory, init_memory_pos, init_memory1, init_memory_pos1, reward_record


def create_video_feat(flow_rate, temperature):


    # Find the range index for flow rate and temperature
    flow_rate_index = next((i for i, (start, end) in enumerate(flow_rate_ranges) if start <= flow_rate < end), None)
    temperature_index = next((i for i, (start, end) in enumerate(temperature_ranges) if start <= temperature < end), None)

    # Create label based on flow rate and temperature range indices
    if flow_rate_index is not None and temperature_index is not None:
        label = flow_rate_index+temperature_index*len(flow_rate_ranges)
    else:
        label = None

    #print("Original label: ", label)
    #0.5
    # Generate a random number between 0 and 1
    random_number = np.random.rand()
    # Check if the random number falls within the probability of using the provided label
    if random_number > probability_given_label:
        # Generate a random integer between 0 and 19 (inclusive)
        label = np.random.randint(0, video_class_size)
        #print("Random label ", label)

    video_feat = np.zeros((1, video_class_size))
    #print(video_feat)
    #print(label)
    #video_feat[0][label] = 1

    feat_scale = [0.5, 1, 2, 3, 4]
    for i in range(len(video_feat[0])):
        video_feat[0][i] = np.exp(-feat_scale[np.random.randint(0, 5)]*abs(i-label))
        if i != label:
            video_feat[0][i] = video_feat[0][i] + np.random.normal(0, 0.01, 1)
            if video_feat[0][i] < 0:
                video_feat[0][i] = 1e-20

    #print("video_feat: ", video_feat)

    return video_feat, label


def moving_average(data, window_size):
    """
    Calculate the moving average of data with a given window size.
    
    Args:
        data (array-like): Input data to compute the moving average.
        window_size (int): Size of the moving average window.
    
    Returns:
        numpy.ndarray: Array containing the moving average values.
    """
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')



def get_ranges_from_label(label):
    # Define ranges for flow rate and temperature

    # Calculate the index for flow rate and temperature based on label
    num_flow_rate_ranges = len(flow_rate_ranges)
    flow_rate_index = label % num_flow_rate_ranges
    temperature_index = label // num_flow_rate_ranges

    # Retrieve the corresponding ranges
    flow_rate_range = flow_rate_ranges[flow_rate_index]
    temperature_range = temperature_ranges[temperature_index]

    return flow_rate_range, temperature_range


def gen_state_(flow_rate_target, current_temp, current_temp_target):

    global previous_cls
    
    normalize_temp = (current_temp - min_temp) / (max_temp - min_temp)
    normalize_temp_target = (current_temp_target - min_temp) / (max_temp - min_temp)
    firmware_read = np.array([normalize_temp, normalize_temp_target]).reshape(
        1, firmware_state_size
    )

    video_feat, label = create_video_feat(flow_rate_target, current_temp)


    previous_cls[0:len(previous_cls)-1] = previous_cls[1:len(previous_cls)]
    previous_cls[-1] = label/len(flow_rate_ranges)
    #print("previous_cls: ", previous_cls, "label: ", label)

    #time.sleep(10)


    #print("video_feat: ", video_feat, "label", label)
    flow_rate_range, temperature_range = get_ranges_from_label(label)

    #flow_rate_range, temperature_range = get_ranges_from_label(max_index)
    if progress_show_flag:
      print("Flow rate range:", flow_rate_range, " Temperature range:", temperature_range)

    #print("previous_cls shape:", previous_cls.reshape(1, -1).shape)
    #print("video_feat shape:", video_feat.shape)
    #print("firmware_read shape:", firmware_read.shape)
    
    state_ = torch.tensor(np.concatenate([previous_cls.reshape(1, -1), video_feat, firmware_read], axis=1), dtype=torch.float32, requires_grad=True).to(device)
    #state_.requires_grad_(True)




    return state_#, reward


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(
        self, *args
    ):  # if the memory is at full capacity, it overwrites the oldest transition.
        # print(len(self.memory))
        # print(self.position)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):  # return the current size of the replay memory
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_actions1):
        super(DQN, self).__init__()
        # Determine the size of video features and indicator based on n_observations
        n_video_features = video_class_size + previous_cls_num
        n_indicator = firmware_state_size

        self.fc1_video = nn.Linear(n_video_features, 256)
        self.relu1_video = nn.ReLU()  # ReLU  Tanh

        self.fc1_indicator = nn.Linear(n_indicator, 32)
        self.relu1_indicator = nn.ReLU()

        self.fc_merged = nn.Linear(
            256 + 32, 128
        )  # Combine processed video and indicator features
        self.relu_merged = nn.ReLU()
        #self.fc_merged1 = nn.Linear(
        #    256, 128
        #)  # Combine processed video and indicator features
        #self.relu_merged1 = nn.ReLU()

        self.fc_output = nn.Linear(128, n_actions)
        self.fc_output1 = nn.Linear(128, n_actions1)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        # He initialization
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            init.zeros_(m.bias)

    def forward(self, x):
        # Separate the video features and indicator value from the input
        video_input = x[:, :-firmware_state_size]  # All columns except the last one   #
        indicator_input = x[:, -firmware_state_size:]  # Last column

        x_video = self.relu1_video(self.fc1_video(video_input))
        x_indicator = self.relu1_indicator(self.fc1_indicator(indicator_input))

        # Concatenate the processed video and indicator features
        x_merged = torch.cat((x_video, x_indicator), dim=1)
        x_merged = self.relu_merged(self.fc_merged(x_merged))
        #x_merged = self.relu_merged1(self.fc_merged1(x_merged))
        q_values = self.fc_output(x_merged)
        q_values1 = self.fc_output1(x_merged)
        return q_values, q_values1


# DQN agent class
class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        action_size1,
        memory_capacity=10000,
        batch_size= 512,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=500,  # 0.995,
        tao= 0.005,
        lr= 1e-4,
        steps_done=0,
    ):
        self.state_size = state_size  # number of state observations
        self.action_size = action_size  # number of actions
        self.action_size1 = action_size1
        self.interaction_counter = 0
        self.memory = ReplayMemory(memory_capacity)
        self.memory1 = ReplayMemory(memory_capacity)

        self.batch_size = (
            batch_size  # The number of transitions sampled from the replay buffer
        )
        self.gamma = gamma  # The discount factor
        self.epsilon = epsilon_start  # The starting value of epsilon
        self.epsilon_end = epsilon_end  # The final value of epsilon
        self.epsilon_decay = epsilon_decay  # Controls the rate of exponential decay of epsilon, higher means a slower decay
        self.tao = tao  # The update rate of the target network
        self.lr = lr  # The learning rate of the ``AdamW`` optimizer

        self.policy_net = DQN(state_size, action_size, action_size1).to(device)
        self.target_net = DQN(state_size, action_size, action_size1).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )  # Optimizer: AdamW and use the AMSGrad variant

        self.steps_done = (
            steps_done
            # steps_done  # keep track of the number of steps taken during training
        )

    def step_done_plus(self):
        self.steps_done += 1

    
    # Implement an epsilon-greedy exploration strategy. It decides whether the agent should explore (choose a random action)
    # or exploit (choose the action with the highest Q-value) based on an epsilon threshold that decays over time.
    def select_action(self, state):
        if progress_show_flag:
          print("self.steps_done: ", self.steps_done)

        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(
            -1.0 * (self.steps_done / self.epsilon_decay)
        )
        self.steps_done += 1

        if sample > eps_threshold:
            random_action_flag = False
            with torch.no_grad():
                q_values, q_values1 = self.policy_net(state)
                action = q_values.max(1)[1].view(1, 1)
                if self.interaction_counter % temp_action_interval != 0:
                    action1 = action_range1
                else:
                    action1 = q_values1.max(1)[1].view(1, 1)

        else:
            if progress_show_flag:
              print("Random Action")
            
            action = torch.tensor(
                [[random.randint(0, action_range * 2)]],
                device=device,
                dtype=torch.long,
            )
            if self.interaction_counter % temp_action_interval != 0:
                action1 = action_range1
            else:
                action1 = torch.tensor(
                    [[random.randint(0, action_range1 * 2)]],
                    device=device,
                    dtype=torch.long,
                )

        return action, action1

    # Perform the Q-learning update using a batch of experiences sampled from the replay memory.
    # It computes the loss using the Huber loss (smooth L1 loss) between the predicted Q-values (state_Q_values) and the target Q-values (expected_state_Q_values).
    # The loss is then used to update the DQN's policy network using gradient descent.
    # It also incorporates gradient clipping to prevent exploding gradients.
    def optimize_model(self):
        #global total_loss_record
        #print("****1:", self.batch_size)
        #print("****2:", self.batch_size/2)
        transitions = self.memory.sample(int(self.batch_size/2)) + self.memory1.sample(int(self.batch_size/2))

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_Q_values = torch.zeros(self.batch_size, device=device)
        next_state_Q_values1 = torch.zeros(self.batch_size, device=device)

        # Filter out None next states and concatenate the remaining tensors
        if any(
            non_final_mask
        ):  # Check if there are any non-final states before proceeding
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            )
            # print("non_final_next_states:", non_final_next_states.shape)
            with torch.no_grad():
                targetQ, targetQ1 = self.target_net(non_final_next_states)
                next_state_Q_values[non_final_mask] = targetQ.max(1)[0]
                next_state_Q_values1[non_final_mask] = targetQ1.max(1)[0]
        else:
            non_final_next_states = None

        # Check if there are any non-final states before proceeding
        if non_final_next_states is not None and len(non_final_next_states) == 0:
            return

        # print(next_state_Q_values)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        action1_batch = torch.cat(batch.action1)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        q_values, q_values1 = self.policy_net(state_batch)
        state_Q_values = q_values.gather(1, action_batch.unsqueeze(1))
        state_Q_values1 = q_values1.gather(1, action1_batch.unsqueeze(1))


        # Compute the expected Q values
        expected_state_Q_values = (next_state_Q_values * self.gamma) + reward_batch
        expected_state_Q_values1 = (next_state_Q_values1 * self.gamma) + reward_batch
        # print("expected_state_Q_values: ", expected_state_Q_values)


        #state_Q_values.requires_grad_(True)
        #expected_state_Q_values.requires_grad_(True)
        #state_Q_values1.requires_grad_(True)
        #expected_state_Q_values1.requires_grad_(True)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_Q_values, expected_state_Q_values.unsqueeze(1))
        loss1 = criterion(state_Q_values1, expected_state_Q_values1.unsqueeze(1))
        total_loss = loss + loss1

        #total_loss.requires_grad_(True)

        #print("state_Q_values ", state_Q_values)
        #print("state_Q_values1 ", state_Q_values1)
        #print("loss ",loss)
        #print("loss1 ",loss1)
        #print("total_loss ", total_loss)
        #total_loss_record.append(total_loss.item())


        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


def rl_agent(state_):

    result, result1 = agent.select_action(state_)

    actions = int(result)
    actions1 = int(result1)

    return actions, actions1


def gaussian_function(x, y):
    # Center and standard deviations
    x_center, y_center = 100, 210
    sigma_x, sigma_y = 35, 15

    # Rotation angle (in radians)
    theta = np.radians(60)  # Rotate by 45 degrees

    # Rotate coordinates
    x_rot = (x - x_center) * np.cos(theta) - (y - y_center) * np.sin(theta)
    y_rot = (x - x_center) * np.sin(theta) + (y - y_center) * np.cos(theta)

    # Compute Gaussian function
    exponent = -((x_rot)**2 / (2 * sigma_x**2) + (y_rot)**2 / (2 * sigma_y**2))
    return np.exp(exponent) * 2 - 1 #- (abs(x-x_center)/250)


def ellipse_function(x, y):
    # Center and axes lengths
    x_center, y_center = 90, 210
    a, b = 20, 10#40, 10#40, 20
    # Rotation angle (in radians)
    theta = np.radians(70)  # 70
    # Rotate coordinates
    x_rot = (x - x_center) * np.cos(theta) - (y - y_center) * np.sin(theta)
    y_rot = (x - x_center) * np.sin(theta) + (y - y_center) * np.cos(theta)
    # Compute distance from center
    distance = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    # Compute ellipse equation with inverse distance
    return 1 / (1 + distance) * 2 - 1 #- (abs(x-x_center)/250)


def reward_fun(current_fr, current_temp, current_temp_target):

    reward = ellipse_function(current_fr, current_temp_target)
    reward = torch.tensor([reward], device=device)

    return reward


agent = DQNAgent(state_size, action_size, action_size1)
load_checkpoint_flag = True

episode_num = 0
reward_record = []

episode_load = 2000
if load_checkpoint_flag:
    (agent.policy_net,agent.target_net,agent.optimizer,episode_num,agent.memory.memory,agent.memory.position,agent.memory1.memory,agent.memory1.position,reward_record) = load_DQN_checkpoint(agent.policy_net, agent.target_net, agent.optimizer, episode_load)

print("Load checkpoint for episode ", episode_num)
print("len of reward: ", len(reward_record))


#agent.memory.memory = []
#agent.memory.position = 0
#agent.memory1.memory = []
#agent.memory1.position = 0

agent.steps_done = 100*20000

done = False
state = None
next_state = None
next_action = None
next_action1 = None
episode_steps = 0
reward_cumulate = 0
temp_target_record = []
temp_current_record = []

progress_show_flag = False


while episode_num <= 25000:
    #print(episode_num)
    #if episode_num % 500 == 0 and episode_num != 0:
        #converge_plot = True
        #progress_show_flag = True
        #save_DQN_checkpoint(agent.policy_net,agent.target_net,agent.optimizer,episode_num,agent.memory.memory,agent.memory.position,agent.memory1.memory,agent.memory1.position,reward_record)
    #else:
    #    progress_show_flag = True

    progress_show_flag = True

    current_fr = random.randint(min_fr, max_fr)
    current_temp_target = random.randint(min_temp, max_temp)
    current_temp = current_temp_target

    firmware_set = current_fr
    temp_set = current_temp_target
    for transition_i in range(max_transition):
        if progress_show_flag:
            print("***********************episode ", episode_num, " transition ", transition_i,"***************************")  
            print("current_fr: ", current_fr, "current_temp_target: ", current_temp_target, "current_temp: ", current_temp)

        temp_target_record.append(current_temp_target)
        temp_current_record.append(current_temp)

        if next_state is not None:
            state = next_state
            action = torch.tensor([next_action], device=device, dtype=torch.long)
            action1 = torch.tensor([next_action1], device=device, dtype=torch.long)


        next_state = gen_state_(current_fr, current_temp, current_temp_target)

        print("***********next state***************")
        print(next_state)

        print("***********next state without previous cls***************")
        print(next_state[0, previous_cls_num:previous_cls_num+3])
        #time.sleep(10)


        next_action, next_action1 = rl_agent(next_state)

        reward = reward_fun(firmware_set, current_temp, temp_set)


        change = int(next_action - action_range) * change_interval
        change1 = int(next_action1 - action_range1) * change_interval1

        if progress_show_flag:
            print(f"***Change={change:.0f}, {change1:.0f}***")

        # track the firmware setting
        firmware_set = current_fr + change
        temp_set = int(current_temp_target + change1)

        if converge_plot:
            firmware_set_record.append(firmware_set)
            temp_set_record.append(current_temp)

        if firmware_set > max_fr:
            firmware_set = max_fr
        elif firmware_set < min_fr:
            firmware_set = min_fr


        if temp_set > max_temp:
            temp_set = max_temp
        elif temp_set < min_temp:
            temp_set = min_temp




        # these two updates of interacton_counter and steps_done are implemented here instead of selec_action function because we will make multiple decisions to determine one action
        agent.interaction_counter += 1
        if agent.interaction_counter == max_transition:
            agent.interaction_counter = 0

        #agent.steps_done += 1

        #agent.step_done_plus()

        if transition_i == max_transition - 1:
            done = True
            #episode_num += 1
        else:
            done = False
            #episode_steps[ip] += 1



        if state is not None:
                        
            if current_fr > 100:
                #print("memory")
                agent.memory.push(
                    state, action, action1, next_state, reward, done
                )
                #firmware_record.push(current_fr)
            else:
                #print("memory1")
                agent.memory1.push(
                state, action, action1, next_state, reward, done
                )
                #firmware_record_1.push(current_fr)

            #print("len(firmware_record.memory) ", len(firmware_record.memory), "len(firmware_record_1.memory)", len(firmware_record_1.memory))
            #print(firmware_record.memory[-1], firmware_record_1.memory[-1])

            if done:
                reward_record.append(reward_cumulate)
                reward_cumulate = 0

                
            else:
                reward_cumulate += reward.item()

            if progress_show_flag:
                
                print(
                    "reward:",
                    round(reward.item(), 2),
                    ", done:",
                    done,
                    ", memory_len:",
                    len(agent.memory),
                    ", memory1_len:",
                    len(agent.memory1),
                    ", episode:",
                    episode_num,
                )
            #print("reward_record: ", reward_record)

            if len(agent.memory) < agent.batch_size/2 or len(agent.memory1) < agent.batch_size/2:
                pass
            else:
                if progress_show_flag:
                    print("OPTIMIZE")
                
                # Perform a single optimization step on the agent's policy network
                agent.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * agent.tao + target_net_state_dict[key] * (1 - agent.tao)
                agent.target_net.load_state_dict(target_net_state_dict)


        current_fr = firmware_set
        current_temp_target = temp_set
        add_temp = (current_temp_target - current_temp) / temp_pid_interval 
        #print("add_temp: ", add_temp)
        current_temp = current_temp + add_temp



    episode_num += 1


  #time.sleep(20)
  
