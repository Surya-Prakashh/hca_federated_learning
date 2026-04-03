import torch
import torch.nn as nn
import joblib
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

try:
    data = torch.load('rl_triage_model.pth', map_location='cpu', weights_only=False)
    model = DQN(4, 5)
    model.load_state_dict(data['model_state_dict'])
    
    # Extract to numpy
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
        
    numpy_model = {
        "weights": weights,
        "means": data['means'],
        "stds": data['stds'],
        "features": data['features'],
        "action_dim": 5
    }
    
    joblib.dump(numpy_model, 'rl_triage_weights.joblib')
    print("SUCCESS: Exported as rl_triage_weights.joblib")
except Exception as e:
    print(f"FAILURE: {e}")
