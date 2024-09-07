import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt
import os
import ast
from sklearn.metrics import precision_score, recall_score

# Load and preprocess data
df = pd.read_csv('TMDB_balanced_movies.csv')
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x))
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['combined_text'].tolist(), y, test_size=0.2, random_state=42
)

# Tokenize the entire dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_labels = y.shape[1]
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

full_tokens = tokenizer(df['combined_text'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=50)
full_labels = torch.tensor(y, dtype=torch.float32)

# Prepare DataLoader
train_dataset = TensorDataset(full_tokens['input_ids'], full_tokens['attention_mask'], full_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Create DataLoader for validation
val_tokens = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt", max_length=50)
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'], torch.tensor(val_labels, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=8)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Define CyclicLR scheduler
scheduler = CyclicLR(
    optimizer, 
    base_lr=1e-5,     # Minimum learning rate
    max_lr=2e-4,      # Maximum learning rate
    step_size_up=2000, # Number of training iterations to reach the max_lr
    mode='triangular2', # Mode for the scheduler
    cycle_momentum=False # BERT does not use momentum, so disable it
)

criterion = torch.nn.BCEWithLogitsLoss()
writer = SummaryWriter()

# Function to calculate accuracy
def calculate_accuracy(logits, labels):
    predictions = torch.sigmoid(logits) > 0.5   
    correct = (predictions == labels).float()
    accuracy = correct.sum().item() / labels.numel()
    return accuracy

# Training loop
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

num_epochs = 12
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_accuracies = []
    
    for step, batch in enumerate(train_loader):
        input_ids_batch, attention_mask_batch, labels = batch
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate and log training accuracy
        train_accuracy = calculate_accuracy(logits, labels)
        epoch_train_accuracies.append(train_accuracy)
        global_step = epoch * len(train_loader) + step

        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('Accuracy/train', train_accuracy, global_step)

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}, Train Accuracy: {train_accuracy}")

    # Validation loop after each epoch
    model.eval()
    epoch_val_accuracies = []
    val_loss = 0
    
    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            val_input_ids_batch, val_attention_mask_batch, val_labels = val_batch
            val_input_ids_batch = val_input_ids_batch.to(device)
            val_attention_mask_batch = val_attention_mask_batch.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_input_ids_batch, attention_mask=val_attention_mask_batch)
            val_logits = val_outputs.logits
            val_loss += criterion(val_logits, val_labels).item()

            # Calculate validation accuracy for each movie
            val_accuracy = calculate_accuracy(val_logits, val_labels)
            epoch_val_accuracies.append(val_accuracy)

            # Log validation accuracy per movie
            writer.add_scalar('Accuracy/val', val_accuracy, epoch * len(val_loader) + idx)
            print(f"Epoch {epoch}, Movie {idx}, Validation Accuracy: {val_accuracy}")
    
    avg_val_accuracy = sum(epoch_val_accuracies) / len(epoch_val_accuracies)
    avg_train_accuracy = sum(epoch_train_accuracies) / len(epoch_train_accuracies)
    
    writer.add_scalar('Accuracy/val_epoch', avg_val_accuracy, epoch)
    writer.add_scalar('Accuracy/train_epoch', avg_train_accuracy, epoch)

    # Update learning rate scheduler
    scheduler.step(val_loss / len(val_loader))

    print(f"Epoch {epoch}, Average Train Accuracy: {avg_train_accuracy}, Average Validation Accuracy: {avg_val_accuracy}")

    # Save model weights at the end of each epoch
    model_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

# Close TensorBoard writer
writer.close()
del full_labels, val_tokens, y, train_dataset, val_dataset, train_loader, val_loader
torch.cuda.empty_cache()
gc.collect()

