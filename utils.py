import numpy as np
import torch
from typing import Tuple, List
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline



from config import (
    device, MODEL_CONFIG, TEMPERATURE_CONFIG, 
    FLOW_RATE_CONFIG, PATHS, REWARD_CONFIG, MODE_CONFIG
)

def create_video_feat(flow_rate: float, temperature: float) -> Tuple[np.ndarray, int]:
    """Create video features based on flow rate and temperature.
    
    Args:
        flow_rate (float): Current flow rate
        temperature (float): Current temperature
        
    Returns:
        Tuple[np.ndarray, int]: Video features and label
    """
    # Find the range index for flow rate and temperature
    flow_rate_index = next(
        (i for i, (start, end) in enumerate(FLOW_RATE_CONFIG['flow_rate_ranges']) 
         if start <= flow_rate < end), 
        None
    )
    temperature_index = next(
        (i for i, (start, end) in enumerate(TEMPERATURE_CONFIG['temperature_ranges']) 
         if start <= temperature < end), 
        None
    )

    # Create label based on flow rate and temperature range indices
    if flow_rate_index is not None and temperature_index is not None:
        label = flow_rate_index + temperature_index * len(FLOW_RATE_CONFIG['flow_rate_ranges'])
    else:
        label = None

    # Generate random label with probability
    random_number = np.random.rand()
    if random_number > MODEL_CONFIG['probability_given_label']:
        label = np.random.randint(0, MODEL_CONFIG['video_class_size'])

    # Create video features
    video_feat = np.zeros((1, MODEL_CONFIG['video_class_size']))
    feat_scale = [0.5, 1, 2, 3, 4]
    
    for i in range(len(video_feat[0])):
        video_feat[0][i] = np.exp(-feat_scale[np.random.randint(0, 5)] * abs(i - label))
        if i != label:
            video_feat[0][i] = video_feat[0][i] + np.random.normal(0, 0.01, 1)
            if video_feat[0][i] < 0:
                video_feat[0][i] = 1e-20

    return video_feat, label

def get_ranges_from_label(label: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get flow rate and temperature ranges from a label.
    
    Args:
        label (int): Label to convert to ranges
        
    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]: Flow rate and temperature ranges
    """
    num_flow_rate_ranges = len(FLOW_RATE_CONFIG['flow_rate_ranges'])
    flow_rate_index = label % num_flow_rate_ranges
    temperature_index = label // num_flow_rate_ranges

    flow_rate_range = FLOW_RATE_CONFIG['flow_rate_ranges'][flow_rate_index]
    temperature_range = TEMPERATURE_CONFIG['temperature_ranges'][temperature_index]

    return flow_rate_range, temperature_range

def gen_state(
    flow_rate_target: float, 
    current_temp: float, 
    current_temp_target: float,
    previous_cls: np.ndarray
) -> torch.Tensor:
    """Generate state tensor from current conditions.
    
    Args:
        flow_rate_target (float): Target flow rate
        current_temp (float): Current temperature
        current_temp_target (float): Target temperature
        previous_cls (np.ndarray): Previous class labels
        
    Returns:
        torch.Tensor: State tensor
    """
    # Normalize temperature values
    normalize_temp = (current_temp - TEMPERATURE_CONFIG['min_temp']) / (
        TEMPERATURE_CONFIG['max_temp'] - TEMPERATURE_CONFIG['min_temp']
    )
    normalize_temp_target = (current_temp_target - TEMPERATURE_CONFIG['min_temp']) / (
        TEMPERATURE_CONFIG['max_temp'] - TEMPERATURE_CONFIG['min_temp']
    )
    
    # Create firmware state
    firmware_read = np.array([normalize_temp, normalize_temp_target]).reshape(
        1, MODEL_CONFIG['firmware_state_size']
    )

    # Create video features
    video_feat, label = create_video_feat(flow_rate_target, current_temp)

    # Update previous class labels
    previous_cls[0:len(previous_cls)-1] = previous_cls[1:len(previous_cls)]
    previous_cls[-1] = label / len(FLOW_RATE_CONFIG['flow_rate_ranges'])

    # Combine all features
    state = torch.tensor(
        np.concatenate([previous_cls.reshape(1, -1), video_feat, firmware_read], axis=1),
        dtype=torch.float32,
        requires_grad=True
    ).to(device)

    return state

def reward_fun(
    current_fr: float, 
    current_temp: float, 
    current_temp_target: float
) -> torch.Tensor:
    """Calculate reward based on current conditions.
    
    Args:
        current_fr (float): Current flow rate
        current_temp (float): Current temperature
        current_temp_target (float): Target temperature
        
    Returns:
        torch.Tensor: Reward value
    """
    # Center and axes lengths
    

    x_center, y_center = REWARD_CONFIG['center']['flow_rate'], REWARD_CONFIG['center']['temperature']
    a, b = REWARD_CONFIG[MODE_CONFIG['phase']]['semi_long_axis'], REWARD_CONFIG[MODE_CONFIG['phase']]['semi_short_axis']

    theta = np.radians(REWARD_CONFIG['angle'])

    # Rotate coordinates
    x_rot = (current_fr - x_center) * np.cos(theta) - (current_temp_target - y_center) * np.sin(theta)
    y_rot = (current_fr - x_center) * np.sin(theta) + (current_temp_target - y_center) * np.cos(theta)

    # Compute distance from center
    distance = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)

    # Compute reward
    reward = 1 / (1 + distance) * 2 - 1
    return torch.tensor([reward], device=device)

def plot_training_progress(
    reward_record: List[float],
    window_size: int = 100,
    episode_num: int = None
) -> None:
    """Plot training progress.
    
    Args:
        reward_record (List[float]): Record of rewards
        window_size (int): Size of moving average window
        episode_num (int, optional): Current episode number for plot filename
    """
    # Create plot directory if it doesn't exist
    os.makedirs(PATHS['plot_directory'], exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(reward_record, label='Raw Rewards', alpha=0.3)
    
    # Calculate moving average
    if len(reward_record) >= window_size:
        moving_avg = np.convolve(
            reward_record, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        plt.plot(
            range(window_size-1, len(reward_record)), 
            moving_avg, 
            label=f'{window_size}-Episode Moving Average',
            linewidth=2
        )
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    if episode_num is not None:
        filename = f"training_progress_episode_{episode_num}.png"
    else:
        filename = "training_progress_latest.png"
    
    plot_path = os.path.join(PATHS['plot_directory'], filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory 


def plot_video_features(episode_video_feat1, episode_video_feat2, episode_video_feat3, episode_num):
    """Plot video features in a 3D visualization.
    
    Args:
        episode_video_feat1 (List[float]): First video feature values
        episode_video_feat2 (List[float]): Second video feature values
        episode_video_feat3 (List[float]): Third video feature values
        episode_num (int): Current episode number
    """

    
    # Save original font size
    original_font_size = plt.rcParams['font.size']
    
    # Define colors for each class
    class_prob0 = (45/255, 124/255, 142/255)
    class_prob1 = (35/255, 168/255, 133/255)
    class_prob2 = (122/255, 209/255, 81/255)

    # Create figure with more space for labels
    fig = plt.figure(figsize=(35, 25))
    ax = fig.add_subplot(111, projection='3d')

    plt.rcParams.update({'font.size': 30})

    # Converting 2D step plot to 3D line plot with scatter dots
    adjust_k = 3
    adjust_b = 1000
    steps_y = range(len(episode_video_feat1))
    steps_x = [0] * len(steps_y)

    # Increase line thickness and scatter point size
    line_width = 3
    scatter_size = 100

    # Plot for video_feat1
    line0, = ax.plot(steps_x, steps_y, episode_video_feat1, label='low flow rate', color=class_prob0, linestyle='--', linewidth=line_width)
    dot0 = ax.scatter(steps_x, steps_y, episode_video_feat1, color=class_prob0, s=scatter_size)

    # Plot for video_feat2
    line1, = ax.plot([step + adjust_b / 2 for step in steps_x], steps_y, episode_video_feat2, label='good flow rate', color=class_prob1, linestyle='--', linewidth=line_width)
    dot1 = ax.scatter([step + adjust_b / 2 for step in steps_x], steps_y, episode_video_feat2, color=class_prob1, s=scatter_size)

    # Plot for video_feat3
    line2, = ax.plot([step + adjust_b for step in steps_x], steps_y, episode_video_feat3, label='high flow rate', color=class_prob2, linestyle='--', linewidth=line_width)
    dot2 = ax.scatter([step + adjust_b for step in steps_x], steps_y, episode_video_feat3, color=class_prob2, s=scatter_size)

    # Setting labels and ticks with increased font sizes and padding
    ax.set_xlabel('Flow rate class', fontsize=50, labelpad=120)
    ax.set_ylabel('Steps', fontsize=50, labelpad=50)
    ax.set_zlabel('Probability', fontsize=50, labelpad=50)
    
    # Increase tick label sizes and adjust padding
    ax.tick_params(axis='both', which='major', labelsize=35, pad=20)
    ax.tick_params(axis='both', which='minor', labelsize=35, pad=20)
    
    # Set axis limits with more space
    ax.set_xlim(-200, adjust_b + 200)
    ax.set_ylim(-10, 110)
    ax.set_zlim(-0.1, 1.1)
    ax.set_zticks([0, 0.25, 0.50, 0.75, 1.00])

    # Rotate the plot for better visibility
    ax.view_init(elev=25, azim=45)

    # Set box aspect ratio
    ax.set_box_aspect([4, 3, 1])

    # Customizing x-axis to show specific values with increased font size and spacing
    x_ticks = [steps_x[0], steps_x[0] + adjust_b / 2, steps_x[0] + adjust_b]
    x_labels = ['<90%', '         90%-110%', '    >110%']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=35)

    # Turn off grid and hide x-axis line and values
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Set the background to pure white
    ax.set_facecolor((1.0, 1.0, 1.0))

    legend_elements = [
        (line0, 'low flow rate (<90%)'),
        (line1, 'good flow rate (90%-110%)'),
        (line2, 'high flow rate (>110%)')
    ]

    # Adding the legend with increased font size and adjusted position
    ax.legend([h for h, l in legend_elements], [l for h, l in legend_elements], 
              loc='upper right', fontsize=35, bbox_to_anchor=(1.05, 0.85))  # Adjusted position

    # Adjust layout to prevent label cutoff
    fig.tight_layout(pad=4.0)

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot with high DPI for better quality
    output_path = os.path.join('plots', f'video_features_episode_{episode_num}.jpg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Restore original font size
    plt.rcParams.update({'font.size': original_font_size})
    
    print(f"Saved video features plot at: {output_path}")
















def plot_flow_rate_and_temperature(flow_rates, target_temps, current_temps, episode_num):
    """Plot flow rate and temperature data.
    
    Args:
        flow_rates (List[float]): List of flow rate values
        target_temps (List[float]): List of target temperature values
        current_temps (List[float]): List of current temperature values
        episode_num (int): Current episode number
    """
    
    
    # Save original font size
    original_font_size = plt.rcParams['font.size']
    
    # Define colors
    targettemp_rgb = (68/255, 1/255, 84/255)
    flowrate_rgb = (59/255, 82/255, 139/255)
    
    # Create a figure and axis for the first plot (Steps vs Flow Rate and Temperature)
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Increase the font size
    plt.rcParams.update({'font.size': 25})

    # Plot current_fr_plot on the left y-axis using a step plot
    ax1.step(range(len(flow_rates)), flow_rates, label='Current flow rate', color=flowrate_rgb, where='mid')
    ax1.set_xlabel('Steps', fontsize=30)
    ax1.set_ylabel('Flow rate (%)', color=flowrate_rgb, fontsize=30)
    ax1.tick_params(axis='x', labelsize=25)  # Set font size for x-axis tick labels
    ax1.tick_params(axis='y', labelsize=25, labelcolor=flowrate_rgb)  # Set font size for y-axis tick labels
    ax1.set_ylim(min(flow_rates)-5, max(flow_rates)+50)

    # Smooth current_temps using make_interp_spline
    x_steps = np.arange(len(current_temps))
    x_smooth_steps = np.linspace(x_steps.min(), x_steps.max(), 300)  # 300 points for smooth curve
    spl = make_interp_spline(x_steps, current_temps, k=3)  # Cubic spline
    current_temps_smooth = spl(x_smooth_steps)

    # Create a second y-axis for target_temps and smoothed current_temps
    ax2 = ax1.twinx()
    ax2.step(range(len(target_temps)), target_temps, label='Target temperature', color=targettemp_rgb, where='mid')
    ax2.plot(x_smooth_steps, current_temps_smooth, label='Current temperature', color='red')
    ax2.set_ylabel('Temperature (°C)', color='red', fontsize=30)
    ax2.tick_params(axis='x', labelsize=25)  # Set font size for x-axis tick labels
    ax2.tick_params(axis='y', labelsize=25, labelcolor='red')
    ax2.set_ylim(min(target_temps)-10, max(target_temps)+15)

    # Add title and legends
    fig.tight_layout()  # To ensure the right y-label is not clipped

    # Add legends for both axes separately
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    # Place the legend for flow rate on the left
    ax1.legend(lines_1, labels_1, loc='upper left')

    # Place the legend for temperature on the right
    ax2.legend(lines_2, labels_2, loc='upper right')

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot with high DPI for better quality
    output_path = os.path.join('plots', f'flow_rate_and_temperature_episode_{episode_num}.jpg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Restore original font size
    plt.rcParams.update({'font.size': original_font_size})
    
    print(f"Saved flow rate and temperature plot at: {output_path}")


def ellipse_function(x, y):
    """Compute the reward surface using an elliptical function.
    
    Args:
        x (np.ndarray): Flow rate values
        y (np.ndarray): Temperature values
        
    Returns:
        np.ndarray: Reward values
    """
    # Center and axes lengths
    x_center, y_center = REWARD_CONFIG['center']['flow_rate'], REWARD_CONFIG['center']['temperature']
    a, b = REWARD_CONFIG[MODE_CONFIG['phase']]['semi_long_axis'], REWARD_CONFIG[MODE_CONFIG['phase']]['semi_short_axis']  # returns 20

    
    # Rotation angle (in radians)
    theta = np.radians(REWARD_CONFIG['angle'])
    
    # Rotate coordinates
    x_rot = (x - x_center) * np.cos(theta) - (y - y_center) * np.sin(theta)
    y_rot = (x - x_center) * np.sin(theta) + (y - y_center) * np.cos(theta)
    
    # Compute distance from center
    distance = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    
    # Compute ellipse equation with inverse distance
    return 1 / (1 + distance) * 2 - 1













def plot_convergence(
    flow_rates: List[float],
    target_temps: List[float],
    current_temps: List[float],
    episode_num: int
) -> None:
    """Plot the convergence of flow rates and temperatures on the reward surface.
    
    Args:
        flow_rates (List[float]): List of flow rates
        target_temps (List[float]): List of target temperatures
        current_temps (List[float]): List of current temperatures
        episode_num (int): Current episode number
    """
    
    # Create plots directory if it doesn't exist
    os.makedirs(PATHS['plot_directory'], exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(15, 7))
    plt.rcParams.update({'font.size': 25})
    
    # Define ranges
    min_temp = TEMPERATURE_CONFIG['min_temp']
    max_temp = TEMPERATURE_CONFIG['max_temp']
    min_fr = FLOW_RATE_CONFIG['min_fr']
    max_fr = FLOW_RATE_CONFIG['max_fr']
    
    # Create mesh grid
    x = np.linspace(min_fr-30, max_fr+70, 300)
    y = np.linspace(min_temp-10, max_temp+10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute reward surface
    Z = ellipse_function(X, Y)
    
    # Plot background contour
    plt.contourf(X, Y, Z, cmap='viridis', alpha=1)
    plt.colorbar(label='Reward')
    
    # Plot trajectory
    plt.plot(flow_rates, current_temps, color='red', linestyle=':', linewidth=3)
    
    # Plot start and target points
    plt.scatter(flow_rates[0], target_temps[0], color='black', marker='o', s=40)
    plt.scatter(100, 210, color='red', marker='o', s=40)
    
    # Set labels and limits
    plt.xlabel('Flow rate (%)', fontsize=30)
    plt.ylabel('Temperature (°C)', fontsize=30)
    plt.xlim(min_fr-30, max_fr+70)
    plt.ylim(min_temp-10, max_temp+10)
    plt.grid(True)
    
    # Set aspect ratio
    x_min, x_max = plt.xlim()
    y_min, y_max = 0.3 * x_min, 0.3 * x_max
    plt.gca().set_aspect((x_max - x_min) / (2 * (y_max - y_min)), adjustable='box')
    
    # Save plot
    convergence_plot_path = os.path.join(
        PATHS['plot_directory'],
        f"convergence_episode_{episode_num}.jpg"
    )
    plt.savefig(convergence_plot_path)
    plt.close()
    
    print("Saved convergence plot at:", convergence_plot_path)







