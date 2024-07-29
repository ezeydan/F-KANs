!pip install pykan  # For PyPI installation

#Federated Learning with KAN

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from kan import KAN
import matplotlib.pyplot as plt
import copy
import time

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Split train_dataset for federated learning into 2 clients
num_clients = 2
rounds = 20
client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)

# Create DataLoader for clients
client_loaders = [DataLoader(dataset, batch_size=16, shuffle=True) for dataset in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Define the KANClassifier with Dropout layers
class KANClassifier(nn.Module):
    def __init__(self, kan_model, num_classes):
        super(KANClassifier, self).__init__()
        self.kan_model = kan_model
        last_layer_width = kan_model.width[-1]
        if isinstance(last_layer_width, list):
            if len(last_layer_width) == 2:
                last_layer_width = last_layer_width[0]
            else:
                raise ValueError("Unexpected format for kan_model.width[-1]")
        self.fc = nn.Linear(last_layer_width, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.kan_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def train_kan(self, dataset, steps, lamb, lamb_entropy):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
        loss_fn = nn.CrossEntropyLoss()
        for step in range(steps):
            optimizer.zero_grad()
            outputs = self(dataset['train_input'].float())
            loss = loss_fn(outputs, dataset['train_label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

# Define the MLP model with the same depth as KAN
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.5))  # Dropout with 50% probability
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def train_model(self, dataset, steps, learning_rate=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
        loss_fn = nn.CrossEntropyLoss()
        for step in range(steps):
            optimizer.zero_grad()
            outputs = self(dataset['train_input'].float())
            loss = loss_fn(outputs, dataset['train_label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

def create_dataset(loader):
    inputs = []
    labels = []
    for data in loader:
        inputs.append(data[0])
        labels.append(data[1])
    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)
    return {
        'train_input': inputs,
        'train_label': labels,
        'test_input': inputs.clone(),
        'test_label': labels.clone()
    }

def fed_avg(global_model, client_models):
    global_state_dict = global_model.state_dict()
    for param_tensor in global_state_dict:
        global_state_dict[param_tensor].zero_()
    for client_model in client_models:
        client_state_dict = client_model.state_dict()
        for param_tensor in global_state_dict:
            global_state_dict[param_tensor] += client_state_dict[param_tensor] / len(client_models)
    global_model.load_state_dict(global_state_dict)
    return global_model

def compute_metrics(model, loader):
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    model.training = False
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.float())
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    model.training = True
    return avg_loss, accuracy, precision, recall, f1

def federated_learning_rounds(global_model, client_loaders, test_loader, rounds=rounds, steps=20):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_precisions = []
    test_precisions = []
    train_recalls = []
    test_recalls = []
    train_f1s = []
    test_f1s = []

    for round_num in range(rounds):
        client_models = [copy.deepcopy(global_model) for _ in range(len(client_loaders))]
        for i, client_loader in enumerate(client_loaders):
            client_dataset = create_dataset(client_loader)
            print(f"Training client {i+1}/{len(client_loaders)}")
            if isinstance(client_models[i], KANClassifier):
                client_models[i].train_kan(client_dataset, steps, lamb=0.01, lamb_entropy=10.0)
            elif isinstance(client_models[i], MLP):
                client_models[i].train_model(client_dataset, steps)

        global_model = fed_avg(global_model, client_models)

        train_loss, train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(global_model, client_loaders[0])
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = compute_metrics(global_model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_precisions.append(train_precision)
        test_precisions.append(test_precision)
        train_recalls.append(train_recall)
        test_recalls.append(test_recall)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)

        print(f'Round {round_num+1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, '
              f'Train Precision: {train_precision:.4f}, Test Precision: {test_precision:.4f}, '
              f'Train Recall: {train_recall:.4f}, Test Recall: {test_recall:.4f}, '
              f'Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}')

    return global_model, train_losses, test_losses, train_accuracies, test_accuracies, train_precisions, test_precisions, train_recalls, test_recalls, train_f1s, test_f1s

# Initialize KAN model with 3 hidden layers and dropout
kan_width = [4, 20, 20, 20]
base_kan_model = KAN(width=kan_width, grid=5, k=3, seed=0)
last_layer_width = base_kan_model.width[-1]
if isinstance(last_layer_width, list):
    if len(last_layer_width) == 2:
        last_layer_width = last_layer_width[0]
    else:
        raise ValueError("Unexpected format for kan_model.width[-1]")
kan_global_model = KANClassifier(base_kan_model, num_classes=3)

# Measure the training time for KAN model
start_time = time.time()
kan_global_model, kan_train_losses, kan_test_losses, kan_train_accuracies, kan_test_accuracies, kan_train_precisions, kan_test_precisions, kan_train_recalls, kan_test_recalls, kan_train_f1s, kan_test_f1s = federated_learning_rounds(
    kan_global_model, client_loaders, test_loader, rounds=rounds)
kan_training_time = time.time() - start_time
print(f"KAN model training time: {kan_training_time:.2f} seconds")

# Initialize MLP model with 3 hidden layers and the same number of units as KAN
input_size = 4
hidden_sizes = [20, 20, 20]
output_size = 3
mlp_model = MLP(input_size, hidden_sizes, output_size)

# Measure the training time for MLP model
start_time = time.time()
mlp_global_model, mlp_train_losses, mlp_test_losses, mlp_train_accuracies, mlp_test_accuracies, mlp_train_precisions, mlp_test_precisions, mlp_train_recalls, mlp_test_recalls, mlp_train_f1s, mlp_test_f1s = federated_learning_rounds(
    mlp_model, client_loaders, test_loader, rounds=rounds)
mlp_training_time = time.time() - start_time
print(f"MLP model training time: {mlp_training_time:.2f} seconds")

# Plot the evolution of training and test metrics for KAN and MLP
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.plot(kan_train_losses, 'o-', label='KAN Train Loss')
plt.plot(kan_test_losses, 'x-', label='KAN Test Loss')
plt.plot(mlp_train_losses, 's-', label='MLP Train Loss')
plt.plot(mlp_test_losses, 'd-', label='MLP Test Loss')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.legend()
plt.title('Federated Learning: Loss Evolution Over Rounds')

plt.subplot(2, 3, 2)
plt.plot(kan_train_accuracies, 'o-', label='KAN Train Accuracy')
plt.plot(kan_test_accuracies, 'x-', label='KAN Test Accuracy')
plt.plot(mlp_train_accuracies, 's-', label='MLP Train Accuracy')
plt.plot(mlp_test_accuracies, 'd-', label='MLP Test Accuracy')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Federated Learning: Accuracy Evolution Over Rounds')

plt.subplot(2, 3, 3)
plt.plot(kan_train_precisions, 'o-', label='KAN Train Precision')
plt.plot(kan_test_precisions, 'x-', label='KAN Test Precision')
plt.plot(mlp_train_precisions, 's-', label='MLP Train Precision')
plt.plot(mlp_test_precisions, 'd-', label='MLP Test Precision')
plt.xlabel('Rounds')
plt.ylabel('Precision')
plt.legend()
plt.title('Federated Learning: Precision Evolution Over Rounds')

plt.subplot(2, 3, 4)
plt.plot(kan_train_recalls, 'o-', label='KAN Train Recall')
plt.plot(kan_test_recalls, 'x-', label='KAN Test Recall')
plt.plot(mlp_train_recalls, 's-', label='MLP Train Recall')
plt.plot(mlp_test_recalls, 'd-', label='MLP Test Recall')
plt.xlabel('Rounds')
plt.ylabel('Recall')
plt.legend()
plt.title('Federated Learning: Recall Evolution Over Rounds')

plt.subplot(2, 3, 5)
plt.plot(kan_train_f1s, 'o-', label='KAN Train F1')
plt.plot(kan_test_f1s, 'x-', label='KAN Test F1')
plt.plot(mlp_train_f1s, 's-', label='MLP Train F1')
plt.plot(mlp_test_f1s, 'd-', label='MLP Test F1')
plt.xlabel('Rounds')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Federated Learning: F1 Score Evolution Over Rounds')

plt.tight_layout()
plt.show()

print(f"KAN model training time: {kan_training_time:.2f} seconds")
print(f"MLP model training time: {mlp_training_time:.2f} seconds")
