import numpy as np
import math
from typing import Dict, Optional
from board import GomokuBoard
from model import GomokuNet

class Node:
    """Node in the Monte Carlo Tree"""
    
    def __init__(self, parent: Optional['Node'] = None, action: Optional[int] = None):
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children: Dict[int, 'Node'] = {}  # action: Node
        self.visit_count = 0
        self.total_value = 0.0  # Sum of all backpropagated values
        self.prior = 0.0  # Prior probability from policy network
        
    def expanded(self) -> bool:
        """Check if node has been expanded (has children)"""
        return len(self.children) > 0
        
    def value(self) -> float:
        """Get average value of node"""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0
        
    def ucb_score(self, c_puct: float = 1.5) -> float:
        """Calculate UCB score for node"""
        if self.visit_count == 0:
            return float('inf')
            
        # Exploration term
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return self.value() + exploration

class MCTS:
    """Monte Carlo Tree Search for Gomoku"""
    
    def __init__(self, model: GomokuNet, c_puct: float = 1.5, num_simulations: int = 800):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        
    def search(self, board: GomokuBoard) -> Dict[int, float]:
        """
        Perform MCTS search from current board state
        Returns action probabilities dictionary {action: probability}
        """
        root = Node()
        
        # Get prior probabilities from neural network
        policy, _ = self.model.predict(board.get_state())
        for action in range(81):
            row, col = action // 9, action % 9
            if board.board[row, col] == 0:  # Only valid moves
                root.children[action] = Node(parent=root, action=action)
                root.children[action].prior = policy[action]
        
        # Perform simulations
        for _ in range(self.num_simulations):
            node = root
            sim_board = GomokuBoard()
            sim_board.board = np.copy(board.board)
            sim_board.current_player = board.current_player
            
            # Selection
            while node.expanded():
                action, node = max(node.children.items(), key=lambda item: item[1].ucb_score(self.c_puct))
                row, col = action // 9, action % 9
                sim_board.make_move(row, col)
                
            # Expansion and Evaluation
            if not sim_board.winner:
                # Get policy and value from neural network
                policy, value = self.model.predict(sim_board.get_state())
                
                # Expand node
                for action in range(81):
                    row, col = action // 9, action % 9
                    if sim_board.board[row, col] == 0:  # Only valid moves
                        node.children[action] = Node(parent=node, action=action)
                        node.children[action].prior = policy[action]
            else:
                value = 1 if sim_board.winner == sim_board.current_player else -1
                
            # Backpropagation
            while node:
                node.visit_count += 1
                node.total_value += value
                value = -value  # Alternate perspective for opponent
                node = node.parent
                
        # Calculate action probabilities
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        action_probs = visit_counts / np.sum(visit_counts)
        
        return {action: prob for action, prob in zip(root.children.keys(), action_probs)}
        
    def get_move(self, board: GomokuBoard, temperature: float = 1.0) -> int:
        """
        Get best move from current board state
        temperature: Controls exploration (1.0 for training, 0.1 for testing)
        """
        action_probs = self.search(board)
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        if temperature == 0:
            return actions[np.argmax(probs)]
            
        # Apply temperature
        probs = np.power(probs, 1.0 / temperature)
        probs /= np.sum(probs)
        
        return np.random.choice(actions, p=probs)
