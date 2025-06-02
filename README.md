# DQN Controller

A Deep Q-Network (DQN) based controller for temperature and flow rate optimization.

## Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Running the Code

Run the controller:
```bash
python3 dqn_controller.py
```

### 3. Configuration Options

Edit `config.py` to modify the controller behavior:

1. **Switch between Training and Testing**
   ```python
   MODE_CONFIG = {
       'mode': 'test',  # Change to 'train' for training mode
       'phase': 'phase_3'  # Current phase
   }
   ```

2. **Select Training Phase**
   ```python
   MODE_CONFIG = {
       'phase': 'phase_1'  # Choose from:
                          # 'phase_1' (2500 episodes)
                          # 'phase_2' (6500 episodes)
                          # 'phase_3' (20500 episodes)
                          # 'phase_4' (33500 episodes)
   }
   ```

3. **Set Test Starting Point**
   ```python
   TEMPERATURE_CONFIG = {
       'test': 213.3  # Initial temperature for testing
   }
   
   FLOW_RATE_CONFIG = {
       'test': 191  # Initial flow rate for testing
   }
   ```

### 4. Output

All plots and logs are saved in the `plots` directory:
- Training progress plots
- Video feature plots
- Flow rate and temperature plots
- Convergence plots
- Detailed log files

## Project Structure

```
├── config.py              # Configuration settings
├── dqn_agent.py          # DQN agent implementation
├── dqn_controller.py     # Main controller script
├── utils.py             # Utility functions
├── requirements.txt     # Required packages
└── plots/              # Output directory
```

## Requirements

Main dependencies:
- PyTorch
- NumPy
- Matplotlib
- SciPy

See `requirements.txt` for complete list. 