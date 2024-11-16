import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import numpy as np
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

# Define winning combinations
winning_combos = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical
    (0, 4, 8), (2, 4, 6)              # Diagonal
]

def check_winner(board, player):
    for combo in winning_combos:
        if all(board[i] == player for i in combo):
            return True
    return False

# Function to simulate a random game between two bots
def simulate_random_game():
    board = [' '] * 9
    players = ['X', 'O']
    current_player = 'X'
    move_history = []

    for turn in range(9):
        available_moves = [i for i, spot in enumerate(board) if spot == ' ']
        if not available_moves:
            break

        move = random.choice(available_moves)
        board[move] = current_player
        move_history.append((board.copy(), move, current_player))

        if turn >= 4:
            if check_winner(board, current_player):
                return move_history, current_player  # Winner

        current_player = 'O' if current_player == 'X' else 'X'

    return move_history, 'Draw'  # Draw if no winner

# Function to let the RF model (bot) choose the best move
def get_best_move(board_state, model):
    # Convert board to numeric format: X=1, O=-1, ' '=0
    board_numeric = [1 if x == 'X' else -1 if x == 'O' else 0 for x in board_state]
    
    best_move = None
    best_score = -1  # Initialize with a low score
    
    # Evaluate all possible moves
    available_moves = [i for i in range(9) if board_state[i] == ' ']
    
    for move in available_moves:
        # Create a feature vector for the move
        feature = board_numeric + [move]
        
        # Convert the feature into a DataFrame with the same column names as training
        feature_df = pd.DataFrame([feature], columns=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'Move'])
        
        # Predict the probability of each class (0: loss, 1: draw, 2: win)
        proba = model.predict_proba(feature_df)[0]
        
        # Calculate expected reward
        # Assign higher value to winning (2), medium to draw (1), and low to loss (0)
        expected_reward = proba[0]*0 + proba[1]*1 + proba[2]*2
        
        # Select the move with the highest expected reward
        if expected_reward > best_score:
            best_score = expected_reward
            best_move = move
    
    if best_move is None:
        # Fallback to a random move if no move is predicted as good
        best_move = random.choice(available_moves)
    
    return best_move

# Function to simulate a self-play game between two bots
def simulate_self_play_game(model_x, model_o):
    board = [' '] * 9
    current_player = 'X'
    move_history = []

    for turn in range(9):
        available_moves = [i for i, spot in enumerate(board) if spot == ' ']
        if not available_moves:
            break

        if current_player == 'X':
            move = get_best_move(board, model_x)
        else:
            move = get_best_move(board, model_o)
        
        board[move] = current_player
        move_history.append((board.copy(), move, current_player))
        
        if turn >= 4:
            if check_winner(board, current_player):
                return move_history, current_player
        
        current_player = 'O' if current_player == 'X' else 'X'
    
    return move_history, 'Draw'

# Function to train the model with self-play
def train_with_self_play(df, model, n_iterations=1000):
    for i in range(n_iterations):
        move_history, result = simulate_self_play_game(model, model)  # Both bots use the same model
        
        for board, move, player in move_history:
            if result == 'Draw':
                label = 1  # Reward for draw
            elif result == player:
                label = 2  # Higher reward for win
            else:
                label = 0  # No reward for loss
            
            # Convert board to numeric values: X=1, O=-1, ' '=0
            board_numeric = [1 if x == 'X' else -1 if x == 'O' else 0 for x in board]
            data_row = board_numeric + [move, label]
            df.loc[len(df)] = data_row  # Append to DataFrame
        
        # Periodically retrain the model
        if (i + 1) % 100 == 0:
            X = df.drop(columns=['Label'])
            y = df['Label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Reinitialize and retrain the model to prevent overfitting
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Iteration {i + 1}: Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# Function to print the current board
def print_live_board(board):
    live_board = (
        f"{board[0]} | {board[1]} | {board[2]}\n"
        "---------\n"
        f"{board[3]} | {board[4]} | {board[5]}\n"
        "---------\n"
        f"{board[6]} | {board[7]} | {board[8]}"
    )
    print(live_board)

# Function for the human to play against the RF bot
def play_game_against_rf(model):
    board = [' '] * 9  # Initialize an empty board
    current_player = 'X'  # Human player starts as 'X'
    
    print("Welcome to Tic-Tac-Toe! You are 'X' and the RF bot is 'O'.")
    print_live_board(board)
    
    for turn in range(9):
        if current_player == 'X':  # Human's turn
            while True:
                try:
                    user_input = int(input("Choose a position (1-9): "))
                    if 1 <= user_input <= 9 and board[user_input - 1] == ' ':
                        board[user_input - 1] = 'X'
                        break
                    else:
                        print("Invalid move, please choose an empty position.")
                except ValueError:
                    print("Please enter a valid number between 1 and 9.")
        else:  # Bot's turn
            print("Bot is making its move...")
            move = get_best_move(board, model)
            board[move] = 'O'
            print(f"Bot chose position {move + 1}")
        
        print_live_board(board)
        
        # Check for a winner after at least 5 turns
        if turn >= 4:
            if check_winner(board, current_player):
                winner = "You" if current_player == 'X' else "Bot"
                print(f"{winner} wins!")
                return
        
        # Switch players
        current_player = 'O' if current_player == 'X' else 'X'
    
    print("It's a draw!")

# Main Execution Flow

# Step 1: Simulate multiple random games to create initial dataset
n_initial_games = 10000
print("Simulating initial random games...")
game_results = [simulate_random_game() for _ in range(n_initial_games)]

# Prepare labeled data
data = []

for move_history, result in game_results:
    for board, move, player in move_history:
        if result == 'Draw':
            label = 1  # Reward for draw
        elif result == player:
            label = 2  # Higher reward for win
        else:
            label = 0  # No reward for loss

        # Convert board to numeric values: X=1, O=-1, ' '=0
        board_numeric = [1 if x == 'X' else -1 if x == 'O' else 0 for x in board]
        data.append(board_numeric + [move, label])

# Create DataFrame
df = pd.DataFrame(data, columns=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'Move', 'Label'])

# Display sample data
print("\nSample Data:")
print(df.head())

# Step 2: Initial Model Training
# Features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Check label distribution
print("\nLabel Distribution:")
print(y.value_counts())

# Initialize and train the Random Forest Classifier with balanced class weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)

# Evaluate the model
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"\nInitial Model Accuracy on Training Data: {accuracy * 100:.2f}%")

# Display classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Step 3: Self-Play Training to Improve the Model
print("\nStarting self-play training...")
model = train_with_self_play(df, model, n_iterations=1000)

# Step 4: Visualize Feature Importance
print("\nVisualizing Feature Importance...")
importances = model.feature_importances_
features = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'Move']
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
sns.barplot(x=[features[i] for i in indices], y=importances[indices])
plt.title('Feature Importance in Tic-Tac-Toe Random Forest')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Step 5: Visualize a Single Decision Tree
print("\nVisualizing a single decision tree from the Random Forest...")
tree = model.estimators_[0]

dot_data = export_graphviz(
    tree, 
    out_file=None, 
    feature_names=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'Move'],
    class_names=['Loss', 'Draw', 'Win'],
    filled=True, 
    rounded=True, 
    special_characters=True
)  

graph = graphviz.Source(dot_data)
graph.render("tic_tac_toe_tree", format="png")  # Save the tree as an image file
graph.view()  # Display the tree

# Step 6: Play Against the Trained Bot
print("\nYou can now play against the trained bot.")
play_game_against_rf(model)
