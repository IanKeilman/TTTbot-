import random
import math
import numpy as np

# Define winning combinations
winning_combos = [
    (0, 1, 2),  # Rows
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),  # Columns
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),  # Diagonals
    (2, 4, 6)
]

def check_winner(board, player):
    """Check if the given player has won."""
    for combo in winning_combos:
        if all(board[i] == player for i in combo):
            return True
    return False

def is_draw(board):
    """Check if the game is a draw."""
    return ' ' not in board

def available_moves(board):
    """Return a list of available moves."""
    return [i for i, spot in enumerate(board) if spot == ' ']

def print_board(board):
    """Print the current board state."""
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--+---+--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--+---+--")
    print(f"{board[6]} | {board[7]} | {board[8]}")

class QLearningAgent:
    """Q-Learning agent that learns from MCTS simulations."""
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.Q = {}  # State-action value table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_Q(self, state, action):
        """Get Q-value for a state-action pair."""
        return self.Q.get((state, action), 0.0)

    def choose_action(self, board, player):
        """Choose an action using an epsilon-greedy policy."""
        state = tuple(board)
        actions = available_moves(board)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        else:
            Q_values = [self.get_Q(state, action) for action in actions]
            max_Q = max(Q_values)
            # Handle multiple actions with the same max Q-value
            best_actions = [action for action, Q in zip(actions, Q_values) if Q == max_Q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_player):
        """Update Q-value based on the reward received and the next state."""
        old_Q = self.get_Q(state, action)
        if next_state:
            next_actions = available_moves(list(next_state))
            if next_actions:
                next_Q_values = [self.get_Q(next_state, a) for a in next_actions]
                max_next_Q = max(next_Q_values)
            else:
                max_next_Q = 0.0
        else:
            max_next_Q = 0.0  # Terminal state

        # Q-Learning update rule
        new_Q = old_Q + self.alpha * (reward + self.gamma * max_next_Q - old_Q)
        self.Q[(state, action)] = new_Q

class MCTSNode:
    """Node in the MCTS tree."""
    def __init__(self, board, parent=None, move=None, player=None):
        self.board = board  # Current board state
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.visits = 0 if parent else 1  # Number of times node has been visited
        self.wins = 0  # Number of wins from this node
        self.move = move  # Move that led to this node
        self.player = player  # Player who made the move

    def ucb1(self, exploration=1.41):
        """Calculate the UCB1 score for this node."""
        if self.visits == 0 or self.parent.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term

    def best_child(self, exploration=1.41):
        """Select the child with the highest UCB1 score."""
        return max(self.children, key=lambda child: child.ucb1(exploration))

def mcts_qlearning(board, player, agent, simulations=1000):
    """Perform MCTS with Q-Learning to select the best move."""
    root = MCTSNode(board=board, player=player)
    # Ensure root visits is at least 1
    root.visits = 1

    for _ in range(simulations):
        node = root
        state_action_history = []
        current_player = player
        # Selection
        while node.children:
            node = node.best_child()
            current_player = 'O' if current_player == 'X' else 'X'
            state_action_history.append((tuple(node.board), node.move, current_player))

        # Expansion
        if not (check_winner(node.board, 'X') or check_winner(node.board, 'O') or is_draw(node.board)):
            moves = available_moves(node.board)
            for move in moves:
                new_board = node.board[:]
                new_board[move] = current_player
                child_node = MCTSNode(board=new_board, parent=node, move=move, player=current_player)
                node.children.append(child_node)

        # Simulation
        if node.children:
            node = random.choice(node.children)
            current_player = 'O' if current_player == 'X' else 'X'
            state_action_history.append((tuple(node.board), node.move, current_player))

        board_sim = node.board[:]
        sim_player = current_player
        while True:
            if check_winner(board_sim, 'X'):
                reward = 1 if sim_player == 'X' else -1
                break
            if check_winner(board_sim, 'O'):
                reward = 1 if sim_player == 'O' else -1
                break
            if is_draw(board_sim):
                reward = 0
                break
            move = agent.choose_action(board_sim, sim_player)
            board_sim[move] = sim_player
            sim_player = 'O' if sim_player == 'X' else 'X'
            state_action_history.append((tuple(board_sim), move, sim_player))

        # Backpropagation
        while node is not None:
            node.visits += 1  # Increment visits
            if node.player == player:
                if reward == 1:
                    node.wins += 1
            elif node.player is not None:
                if reward == -1:
                    node.wins += 1
            reward = -reward  # Switch reward for the opponent
            node = node.parent if node.parent else node

def play_game_against_bot():
    """Function to play against the MCTS + Q-Learning bot."""
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.1)
    print("Training the bot...")
    # Simulate games to train the agent
    train_agent(agent, episodes=50000)
    print("Training completed.\n")

    # Set epsilon to 0 for exploitation during the game
    agent.epsilon = 0

    board = [' '] * 9  # Initialize an empty board
    current_player = 'X'  # Human starts first
    print("Welcome to Tic-Tac-Toe! You are 'X' and the bot is 'O'.")
    print_board(board)

    while True:
        if current_player == 'X':
            # Human's turn
            while True:
                try:
                    move = int(input("Enter your move (1-9): ")) - 1
                    if move in available_moves(board):
                        board[move] = 'X'
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number between 1 and 9.")
        else:
            # Bot's turn
            print("Bot is thinking...")
            move = mcts_qlearning(board, 'O', agent, simulations=1000)
            board[move] = 'O'
            print(f"Bot plays move {move + 1}")

        print_board(board)

        if check_winner(board, current_player):
            if current_player == 'X':
                print("Congratulations! You win!")
            else:
                print("Bot wins!")
            break
        elif is_draw(board):
            print("It's a draw!")
            break
        else:
            current_player = 'O' if current_player == 'X' else 'X'

def train_agent(agent, episodes=50000):
    """Train the agent by playing games against itself."""
    for _ in range(episodes):
        board = [' '] * 9
        current_player = 'X'
        state_action_history = []
        while True:
            state = tuple(board)
            action = agent.choose_action(board, current_player)
            board[action] = current_player
            next_state = tuple(board)
            state_action_history.append((state, action, current_player))

            if check_winner(board, current_player):
                reward = 1
                for s, a, p in reversed(state_action_history):
                    agent.learn(s, a, reward, None, None)
                    reward = -reward  # Alternate reward for the opponent
                break
            elif is_draw(board):
                reward = 0
                for s, a, p in reversed(state_action_history):
                    agent.learn(s, a, reward, None, None)
                break
            else:
                current_player = 'O' if current_player == 'X' else 'X'

# Run the game
if __name__ == "__main__":
    play_game_against_bot()
