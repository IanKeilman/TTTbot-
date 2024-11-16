import random
import math

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

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board[:]  # Copy the board state
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.move = move  # The move that led to this board state
        self.wins = 0  # Number of wins for this node
        self.visits = 0  # Number of times this node has been visited
        self.untried_moves = available_moves(board)  # Moves that have not been tried yet

    def get_current_player(self):
        """Determine whose turn it is at this node."""
        num_moves = len([i for i in self.board if i != ' '])
        return 'X' if num_moves % 2 == 0 else 'O'

    def uct_select_child(self):
        """Select a child node using UCB, prioritizing loss prevention."""
        best_child = None
        best_ucb_score = -float('inf')  # Start with a very low score
        
        for child in self.children:
            # Opponent is the next player after the bot's move
            opponent = 'O' if self.get_current_player() == 'X' else 'X'
            
            # Simulate the opponent's possible moves
            opponent_wins = False
            simulated_board = child.board[:]
            for move in available_moves(simulated_board):
                simulated_board[move] = opponent
                if check_winner(simulated_board, opponent):
                    opponent_wins = True
                    break
                simulated_board[move] = ' '  # Undo move
            
            if opponent_wins:
                continue  # Skip child nodes where opponent can win immediately
            
            # Calculate UCB score
            win_ratio = child.wins / child.visits if child.visits > 0 else 0
            exploration_term = math.sqrt(2 * math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            ucb_score = win_ratio + exploration_term

            # Update the best child based on the highest UCB score
            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_child = child

        # If all child nodes lead to opponent's immediate win, fallback to UCB selection
        if best_child is None:
            # Select based on UCB without considering opponent's immediate win
            best_child = max(
                self.children,
                key=lambda c: (c.wins / c.visits if c.visits > 0 else 0) +
                              (math.sqrt(2 * math.log(self.visits) / c.visits) if c.visits > 0 else float('inf'))
            )
        
        return best_child

    def add_child(self, move, board):
        """Add a child node for a given move."""
        child = MCTSNode(board, self, move)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def update(self, result):
        """Update node's win/visit statistics."""
        self.visits += 1
        self.wins += result

def heuristic_simulation(board, player):
    """Simulate a game with basic heuristics to avoid losses."""
    current_player = player
    while True:
        available = available_moves(board)
        if not available:
            return 0.5  # Draws are better than losses
        
        # Try to make a winning move first
        for move in available:
            board[move] = current_player
            if check_winner(board, current_player):
                return 1 if current_player == player else -10
            board[move] = ' '  # Undo the move
        
        # Block opponent's winning move if possible
        opponent = 'O' if current_player == 'X' else 'X'
        blocked = False
        for move in available:
            board[move] = opponent
            if check_winner(board, opponent):
                board[move] = current_player  # Block
                blocked = True
                break
            board[move] = ' '  # Undo the move
        if not blocked:
            # Otherwise, make a random move
            move = random.choice(available)
            board[move] = current_player

        if check_winner(board, current_player):
            return 1 if current_player == player else -10  # Win or loss

        current_player = opponent  # Switch player

def backpropagate(node, result):
    """Backpropagate the result, favoring draws and penalizing losses heavily."""
    while node is not None:
        node.visits += 1
        node.wins += result
        result = -result  # Invert result for opponent
        node = node.parent

def prevent_immediate_loss(board, player):
    """Check if the opponent has a winning move, and block it."""
    opponent = 'O' if player == 'X' else 'X'
    available = available_moves(board)
    
    for move in available:
        # Simulate opponent's move
        board[move] = opponent
        if check_winner(board, opponent):
            board[move] = ' '  # Undo move
            return move  # Block this move
        board[move] = ' '  # Undo move
    return None  # No immediate loss to prevent

def mcts(board, player, iterations=1000):
    """Monte Carlo Tree Search biased towards avoiding losses and preferring draws."""
    # Check for immediate loss prevention
    prevent_loss_move = prevent_immediate_loss(board, player)
    if prevent_loss_move is not None:
        return prevent_loss_move  # Block opponent's winning move

    root = MCTSNode(board)

    for _ in range(iterations):
        node = root
        state = board[:]
        current_player = player  # Keep track of the player during simulation

        # Selection
        while node.untried_moves == [] and node.children != []:
            node = node.uct_select_child()
            state[node.move] = current_player
            current_player = 'O' if current_player == 'X' else 'X'

        # Expansion
        if node.untried_moves != []:
            move = random.choice(node.untried_moves)
            state[move] = current_player
            node = node.add_child(move, state)
            current_player = 'O' if current_player == 'X' else 'X'

        # Simulation with heuristics
        result = heuristic_simulation(state[:], current_player)

        # Backpropagation
        backpropagate(node, result)

    # Choose the move with the highest visit count
    return max(root.children, key=lambda c: c.visits).move

# Game loop for playing against MCTS
def play_game():
    board = [' '] * 9
    human_player = 'X'
    mcts_player = 'O'

    while True:
        print_board(board)
        
        # Human player's turn
        move = int(input("Enter your move (0-8): "))
        if board[move] != ' ':
            print("Invalid move. Try again.")
            continue
        board[move] = human_player
        if check_winner(board, human_player):
            print("Human wins!")
            print_board(board)
            break
        if is_draw(board):
            print("It's a draw!")
            print_board(board)
            break

        # MCTS player's turn
        move = mcts(board, mcts_player)
        board[move] = mcts_player
        print(f"MCTS chooses move {move}")
        if check_winner(board, mcts_player):
            print("MCTS wins!")
            print_board(board)
            break
        if is_draw(board):
            print("It's a draw!")
            print_board(board)
            break

play_game()
