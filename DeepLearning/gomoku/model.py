import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuNet(nn.Module):
    """Neural network for Gomoku with policy and value heads"""
    
    def __init__(self):
        super().__init__()
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 9 * 9, 81)  # 81 possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """Forward pass through network"""
        # Input shape: (batch_size, 1, 9, 9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 2 * 9 * 9)
        p = F.log_softmax(self.policy_fc(p), dim=1)
        
        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 1 * 9 * 9)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Value between -1 and 1
        
        return p, v
        
    def predict(self, board_state):
        """Predict policy and value for a single board state"""
        # Convert board state to tensor
        board = torch.FloatTensor(board_state['board']).unsqueeze(0).unsqueeze(0)
        
        # Normalize for current player
        if board_state['current_player'] == -1:
            board *= -1
            
        # Forward pass
        with torch.no_grad():
            policy_logits, value = self.forward(board)
            
        # Convert policy logits to probabilities
        policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()
        value = value.item()
        
        return policy, value
