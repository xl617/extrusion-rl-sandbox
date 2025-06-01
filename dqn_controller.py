import random
import numpy as np
import torch
import logging
from typing import List, Tuple

from config import (
    device, MODEL_CONFIG, TEMPERATURE_CONFIG, 
    FLOW_RATE_CONFIG, TRAINING_CONFIG, MODE_CONFIG
)
from dqn_agent import DQNAgent
from utils import (
    gen_state, reward_fun, plot_training_progress, plot_video_features,
    plot_flow_rate_and_temperature, plot_convergence
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(
    checkpoint_episode,
    num_episodes: int = 50000,
    load_checkpoint: bool = True,  
) -> Tuple[DQNAgent, List[float]]:
    """Train the DQN agent.
    
    Args:
        num_episodes (int): Number of episodes to train for
        load_checkpoint (bool): Whether to load from checkpoint
        checkpoint_episode (int): Episode to load checkpoint from
        
    Returns:
        Tuple[DQNAgent, List[float]]: Trained agent and reward record
    """
    # Initialize agent
    agent = DQNAgent(
        state_size=MODEL_CONFIG['state_size'],
        action_size=MODEL_CONFIG['action_size'],
        action_size1=MODEL_CONFIG['action_size1']
    )
    
    # Load checkpoint if requested
    episode_num = 0
    reward_record = []
    if load_checkpoint:
        episode_num, reward_record = agent.load_checkpoint(checkpoint_episode)
        logger.info(f"Loaded checkpoint from episode {episode_num}")
        logger.info(f"Reward record length: {len(reward_record)}")
    
    # Initialize previous class labels
    previous_cls = np.ones(MODEL_CONFIG['previous_cls_num'])
    
    # Training loop
    while episode_num < num_episodes:
        # Initialize episode
        current_fr = random.randint(
            FLOW_RATE_CONFIG['min_fr'], 
            FLOW_RATE_CONFIG['max_fr']
        )
        current_temp_target = random.randint(
            TEMPERATURE_CONFIG['min_temp'], 
            TEMPERATURE_CONFIG['max_temp']
        )
        current_temp = current_temp_target
        
        firmware_set = current_fr
        temp_set = current_temp_target
        reward_cumulate = 0
        state = None
        next_state = None
        next_action = None
        next_action1 = None

        # Lists to store video features for this episode
        episode_video_feat1 = []
        episode_video_feat2 = []
        episode_video_feat3 = []
        episode_current_fr = []
        episode_current_temp = []
        episode_current_temp_target = []
        
        # Episode loop
        for transition_i in range(MODEL_CONFIG['max_transition']):
            
            # Print transition header
            print(f"{'*' * 80}")
            print(f"Episode {episode_num}, Transition {transition_i}, Accuracy {MODEL_CONFIG['probability_given_label']}")
            print(f"{'*' * 80}")

            
            # Print current state
            print(f"set: flow rate={current_fr}; target temperature={current_temp_target}")
            print(f"current: flow rate={current_fr}; nozzle temperature={current_temp:.2f}; target temperature={current_temp_target}")

            # Convert previous_cls values to class numbers
            class_numbers = np.zeros_like(previous_cls)
            class_numbers[np.isclose(previous_cls, 1/3)] = 1
            class_numbers[np.isclose(previous_cls, 2/3)] = 2
            print(f"previous_cls = {class_numbers}")




            
            # Generate next state
            if next_state is not None:
                state = next_state
                action = torch.tensor([next_action], device=device, dtype=torch.long)
                action1 = torch.tensor([next_action1], device=device, dtype=torch.long)
            
            next_state = gen_state(
                current_fr, current_temp, current_temp_target, previous_cls
            )

            # Extract video features from next_state
            # Convert to numpy for easier handling
            next_state_np = next_state.cpu().detach().numpy()
            start_idx = MODEL_CONFIG['previous_cls_num']
            end_idx = start_idx + 3
            


            video_cls_prob = [
                float(next_state_np[0, start_idx]),
                float(next_state_np[0, start_idx+1]),
                float(next_state_np[0, start_idx+2])
            ]
            print(f"video_cls_prob = [{video_cls_prob[0]:.2f}, {video_cls_prob[1]:.2f}, {video_cls_prob[2]:.2f}]")
            
            episode_video_feat1.append(video_cls_prob[0])
            episode_video_feat2.append(video_cls_prob[1])
            episode_video_feat3.append(video_cls_prob[2])
            episode_current_fr.append(current_fr)
            episode_current_temp.append(current_temp)
            episode_current_temp_target.append(current_temp_target)

            
            # Select actions
            next_action, next_action1 = agent.select_action(next_state)
            
            # Calculate reward
            reward = reward_fun(firmware_set, current_temp, temp_set)
            
            # Update firmware settings
            change = int(next_action - MODEL_CONFIG['action_range']) * FLOW_RATE_CONFIG['change_interval']
            change1 = int(next_action1 - MODEL_CONFIG['action_range1']) * FLOW_RATE_CONFIG['change_interval1']

            print(f"***Change={change}, {change1}***")
            
            firmware_set = current_fr + change
            temp_set = int(current_temp_target + change1)
            
            # Clamp values to valid ranges
            firmware_set = np.clip(
                firmware_set, 
                FLOW_RATE_CONFIG['min_fr'], 
                FLOW_RATE_CONFIG['max_fr']
            )
            temp_set = np.clip(
                temp_set, 
                TEMPERATURE_CONFIG['min_temp'], 
                TEMPERATURE_CONFIG['max_temp']
            )
            
            # Update interaction counter
            agent.interaction_counter += 1
            if agent.interaction_counter == MODEL_CONFIG['max_transition']:
                agent.interaction_counter = 0
            
            # Check if episode is done
            done = transition_i == MODEL_CONFIG['max_transition'] - 1
            
            # Store transition in memory
            if state is not None:
                if current_fr > 100:
                    agent.memory.push(
                        state, action, action1, next_state, reward, done
                    )
                else:
                    agent.memory1.push(
                        state, action, action1, next_state, reward, done
                    )
                
                if done:
                    reward_record.append(reward_cumulate)
                    reward_cumulate = 0
                else:
                    reward_cumulate += reward.item()
                
                # Optimize model if enough samples
                if (len(agent.memory) >= agent.batch_size/2 and 
                    len(agent.memory1) >= agent.batch_size/2):
                    loss = agent.optimize_model()
                    #if loss is not None:
                    #    logger.info(f"Loss: {loss:.4f}")
                    
                    # Update target network
                    target_net_state_dict = agent.target_net.state_dict()
                    policy_net_state_dict = agent.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = (
                            policy_net_state_dict[key] * agent.tao + 
                            target_net_state_dict[key] * (1 - agent.tao)
                        )
                    agent.target_net.load_state_dict(target_net_state_dict)
            
            # Update current values
            current_fr = firmware_set
            current_temp_target = temp_set
            add_temp = (current_temp_target - current_temp) / TEMPERATURE_CONFIG['temp_pid_interval']
            current_temp = current_temp + add_temp
        
        # Save checkpoint periodically
        if episode_num % 500 == 0 and episode_num != 0:
            agent.save_checkpoint(episode_num, reward_record)
            plot_training_progress(reward_record)
            # Plot video features
            plot_video_features(episode_video_feat1, episode_video_feat2, episode_video_feat3, episode_num)
            plot_flow_rate_and_temperature(
                episode_current_fr,
                episode_current_temp_target,
                episode_current_temp,
                episode_num
            )
            # Plot convergence
            plot_convergence(
                episode_current_fr,
                episode_current_temp_target,
                episode_current_temp,
                episode_num
            )
        
        episode_num += 1
    
    return agent, reward_record


def test_episode(checkpoint_episode: int) -> None:
    """Test a single episode using a loaded checkpoint.
    
    Args:
        checkpoint_episode (int): Episode to load checkpoint from
    """
    # Initialize agent
    agent = DQNAgent(
        state_size=MODEL_CONFIG['state_size'],
        action_size=MODEL_CONFIG['action_size'],
        action_size1=MODEL_CONFIG['action_size1']
    )
    
    # Load checkpoint
    episode_num, _ = agent.load_checkpoint(checkpoint_episode)
    logger.info(f"Loaded checkpoint from episode {episode_num}")
    
    # Initialize previous class labels
    previous_cls = np.ones(MODEL_CONFIG['previous_cls_num'])
    
    # Initialize episode
    current_fr = FLOW_RATE_CONFIG['test']
    current_temp_target = TEMPERATURE_CONFIG['test']
    current_temp = current_temp_target
    
    firmware_set = current_fr
    temp_set = current_temp_target
    state = None
    next_state = None
    next_action = None
    next_action1 = None


    # Lists to store video features for this episode
    episode_video_feat1 = []
    episode_video_feat2 = []
    episode_video_feat3 = []
    episode_current_fr = []
    episode_current_temp = []
    episode_current_temp_target = []

    
    # Episode loop
    for transition_i in range(MODEL_CONFIG['max_transition']):
        # Print transition header
        print(f"{'*' * 80}")
        print(f"Test Episode, Transition {transition_i}, Accuracy {MODEL_CONFIG['probability_given_label']}")
        print(f"{'*' * 80}")
        
        # Print current state
        print(f"set: flow rate={current_fr}; target temperature={current_temp_target}")
        print(f"current: flow rate={current_fr}; nozzle temperature={current_temp:.2f}; target temperature={current_temp_target}")
        
        # Convert previous_cls values to class numbers
        class_numbers = np.zeros_like(previous_cls)
        class_numbers[np.isclose(previous_cls, 1/3)] = 1
        class_numbers[np.isclose(previous_cls, 2/3)] = 2
        print(f"previous_cls = {class_numbers}")
        
        # Generate next state
        if next_state is not None:
            state = next_state
            action = torch.tensor([next_action], device=device, dtype=torch.long)
            action1 = torch.tensor([next_action1], device=device, dtype=torch.long)
        
        next_state = gen_state(
            current_fr, current_temp, current_temp_target, previous_cls
        )
        
        # Extract video features from next_state
        next_state_np = next_state.cpu().detach().numpy()
        start_idx = MODEL_CONFIG['previous_cls_num']
        
        video_cls_prob = [
            float(next_state_np[0, start_idx]),
            float(next_state_np[0, start_idx+1]),
            float(next_state_np[0, start_idx+2])
        ]
        print(f"video_cls_prob = [{video_cls_prob[0]:.2f}, {video_cls_prob[1]:.2f}, {video_cls_prob[2]:.2f}]")

        episode_video_feat1.append(video_cls_prob[0])
        episode_video_feat2.append(video_cls_prob[1])
        episode_video_feat3.append(video_cls_prob[2])
        episode_current_fr.append(current_fr)
        episode_current_temp.append(current_temp)
        episode_current_temp_target.append(current_temp_target)
        
        # Select actions
        next_action, next_action1 = agent.select_action(next_state)
        
        # Update firmware settings
        change = int(next_action - MODEL_CONFIG['action_range']) * FLOW_RATE_CONFIG['change_interval']
        change1 = int(next_action1 - MODEL_CONFIG['action_range1']) * FLOW_RATE_CONFIG['change_interval1']
        
        print(f"***Change={change}, {change1}***")
        
        firmware_set = current_fr + change
        temp_set = int(current_temp_target + change1)
        
        # Clamp values to valid ranges
        firmware_set = np.clip(
            firmware_set, 
            FLOW_RATE_CONFIG['min_fr'], 
            FLOW_RATE_CONFIG['max_fr']
        )
        temp_set = np.clip(
            temp_set, 
            TEMPERATURE_CONFIG['min_temp'], 
            TEMPERATURE_CONFIG['max_temp']
        )

        # Update interaction counter
        agent.interaction_counter += 1
        if agent.interaction_counter == MODEL_CONFIG['max_transition']:
            agent.interaction_counter = 0
        
        # Update current values
        current_fr = firmware_set
        current_temp_target = temp_set
        add_temp = (current_temp_target - current_temp) / TEMPERATURE_CONFIG['temp_pid_interval']
        current_temp = current_temp + add_temp

    plot_video_features(episode_video_feat1, episode_video_feat2, episode_video_feat3, episode_num)
    plot_flow_rate_and_temperature(
        episode_current_fr,
        episode_current_temp_target,
        episode_current_temp,
        episode_num
        )
    # Plot convergence
    plot_convergence(
        episode_current_fr,
        episode_current_temp_target,
        episode_current_temp,
        episode_num
    )


if __name__ == "__main__":

    if MODE_CONFIG['mode'] == 'train':
        agent, reward_record = train(checkpoint_episode=MODE_CONFIG['phase_checkpoints'][MODE_CONFIG['phase']] )
        plot_training_progress(reward_record)
    else:  # test mode
        test_episode(checkpoint_episode=MODE_CONFIG['phase_checkpoints'][MODE_CONFIG['phase']])
   


    


    


