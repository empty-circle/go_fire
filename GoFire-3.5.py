import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sgfmill import sgf, sgf_moves
import copy
from sgfmill import ascii_boards
from sgfmill import boards

def parse_sgf_file(sgf_content):
    game = sgf.Sgf_game.from_bytes(sgf_content)
    board_size = game.get_size()

    try:
        initial_board, move_list = sgf_moves.get_setup_and_moves(game)
    except ValueError as e:
        if "setup properties after the root node" in str(e):
            # Fallback to initial empty board and process all nodes
            initial_board = boards.Board(board_size)
            move_list = list(game.main_sequence_iter())
        else:
            raise e

    board_states = [initial_board]
    board = boards.Board(board_size)

    moves = []

    for node in move_list:
        if hasattr(node, 'has_move') and node.has_move():
            color, move = node.get_move()
            if move is not None:
                row, col = move
                try:
                    board.play(row, col, color)
                    board_states.append(copy.deepcopy(board))
                    moves.append(move)
                except ValueError:
                    # Ignore illegal moves
                    pass

    return board_states, moves

def preprocess_board_states(board_states):
    preprocessed_data = []
    for board_state in board_states:
        board_array = np.array(board_state, dtype=str)
        black_stones = (board_array == 'b').astype(np.float32)
        white_stones = (board_array == 'w').astype(np.float32)
        empty_spaces = (board_array == '.').astype(np.float32)
        preprocessed_data.append(np.stack([black_stones, white_stones, empty_spaces], axis=-1))
    return np.array(preprocessed_data)


def preprocess_moves(moves, board_size):
    return np.array([row * board_size + col for row, col in moves], dtype=np.int64)

# Load data and preprocess
def load_data(data_folder):
    sgf_files = glob(os.path.join(data_folder, "*.sgf"))
    all_board_states = []
    all_moves = []

    for sgf_file in sgf_files:
        with open(sgf_file, "rb") as f:
            sgf_content = f.read()
        board_states, moves = parse_sgf_file(sgf_content)
        print(f"Processed {sgf_file}: {len(board_states)} board states, {len(moves)} moves")
        all_board_states.extend(board_states)
        all_moves.extend(moves)

    # Preprocess board states and moves
    data = preprocess_board_states(all_board_states)
    labels = preprocess_moves(all_moves, 19)

    return data, labels


train_data_folder = 'train_data'
train_data, train_labels = load_data(train_data_folder)
print("train_data shape:", train_data.shape)
print("train_labels shape:", train_labels.shape)

validation_data_folder = 'validation'
validation_data, validation_labels = load_data(validation_data_folder)

# Define the custom dataset class
class GoDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        board_state = self.data[index]
        move = self.labels[index]
        return board_state, move

train_dataset = GoDataset(train_data, train_labels)
validation_dataset = GoDataset(validation_data, validation_labels)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define the model
class GoCNN(nn.Module):
    def __init__(self):
        super(GoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 19 * 19, 256)
        self.fc2 = nn.Linear(256, 361)  # 19 x 19 board positions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 19 * 19)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = GoCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation accuracy: {100 * correct / total}%")
    model.train()

# Save the trained model
torch.save(model.state_dict(), "go_model.pth")

# Load the saved model and use it for playing Go
loaded_model = GoCNN()
loaded_model.load_state_dict(torch.load("go_model.pth"))

# Use the loaded_model for playing Go
