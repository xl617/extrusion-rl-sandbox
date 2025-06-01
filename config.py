import torch
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Temperature and flow rate ranges
TEMPERATURE_CONFIG = {
    'min_temp': 190,
    'max_temp': 230,
    'temp_action_interval': 10,
    'temp_pid_interval': 3,  # change for heat step response 
    'temperature_ranges': [(185, 235)],
    'test': 213.3
}

FLOW_RATE_CONFIG = {
    'min_fr': 30,
    'max_fr': 250,
    'change_interval': 5,
    'change_interval1': 10,
    'flow_rate_ranges': [(25, 90), (90, 110), (110, 255)],
    'test': 191
}


# Model configuration
MODEL_CONFIG = {
    'probability_given_label': 1,
    'firmware_state_size': 2,
    'max_transition': 100,
    'previous_cls_num': 30,
    'video_class_size': len(FLOW_RATE_CONFIG['flow_rate_ranges']) * len(TEMPERATURE_CONFIG['temperature_ranges']),
    'state_size': None,  # Will be set dynamically
    'action_range': 2,
    'action_range1': 2,
    'action_size': 5,  # 2 * action_range + 1
    'action_size1': 5  # 2 * action_range1 + 1
}

# Training configuration
TRAINING_CONFIG = {
    'memory_capacity': 10000,
    'batch_size': 512,
    'gamma': 0.99,
    'epsilon_start': 0.9,
    'epsilon_end': 0.01,
    'epsilon_decay': 500,
    'tao': 0.005,
    'learning_rate': 1e-4
}


# Base directory
HOME_DIR = "/home/cam/Documents/Server/Video-Swin-Transformer"

# File paths
PATHS = {
    'home_directory': HOME_DIR,
    'plot_directory': os.path.join(HOME_DIR, "sandbox_codes", "plots"),
    'checkpoint_directory': os.path.join(HOME_DIR, "sandbox_codes", "checkpoints"),
    'new_checkpoints_dir': os.path.join(HOME_DIR, "sandbox_codes", "checkpoints", "new"),
    'benchmark_checkpoints_dir': os.path.join(HOME_DIR, "sandbox_codes", "checkpoints", "benchmark")
}


# Update state_size based on other parameters
MODEL_CONFIG['state_size'] = (
    MODEL_CONFIG['video_class_size'] + 
    MODEL_CONFIG['firmware_state_size'] + 
    MODEL_CONFIG['previous_cls_num']
) 


# Mode configuration
MODE_CONFIG = {
    'mode': 'test',  # 'train' or 'test'
    'phase': 'phase_3',
    'phase_checkpoints': {
        'phase_1': 2500,
        'phase_2': 6500,
        'phase_3': 20500,
        'phase_4': 33500
    }
}


# Reward function configuration
REWARD_CONFIG = {
    'center': {
        'flow_rate': 90,
        'temperature': 210
    },
    'phase_1': {
        'semi_long_axis': 40,
        'semi_short_axis': 20
    },
    'phase_2': {
        'semi_long_axis': 40,
        'semi_short_axis': 10
    },
    'phase_3': {
        'semi_long_axis': 20,
        'semi_short_axis': 10
    },
    'angle': 70
}






