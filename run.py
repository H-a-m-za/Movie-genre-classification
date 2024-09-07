import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import random
from colorama import Fore, Style, init

init()
colors = [
    Fore.BLACK,
    Fore.RED,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.BLUE,
    Fore.MAGENTA,
    Fore.CYAN,
    Fore.WHITE,
    Fore.LIGHTBLACK_EX,
    Fore.LIGHTRED_EX,
    Fore.LIGHTGREEN_EX,
    Fore.LIGHTYELLOW_EX,
    Fore.LIGHTBLUE_EX,
    Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTCYAN_EX,
    Fore.LIGHTWHITE_EX,
]

# Load and preprocess data
df = pd.read_csv('TMDB_balanced_movies.csv')
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x))

# Initialize MultiLabelBinarizer and tokenizer
mlb = MultiLabelBinarizer()
mlb.fit_transform(df['genres'])  # Fitting to the entire dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained model with the correct number of labels
num_labels = mlb.classes_.shape[0]
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model_save_path = 'checkpoints/model_epoch_8.pth'  # Path to your saved weights
model.load_state_dict(torch.load(model_save_path))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Model weights loaded.")

# Function to test a single movie
def test_single_movie(movie_index):
    # Example single movie text and actual genres
    
    single_movie_text = df['combined_text'].iloc[movie_index]  # Get the movie text by index
    single_movie_actual_genres = df['genres'].iloc[movie_index]  # Get the actual genres

    # Tokenize the single movie text
    single_movie_token = tokenizer(single_movie_text, padding=True, truncation=True, return_tensors="pt", max_length=50)
    single_movie_token = {key: value.to(device) for key, value in single_movie_token.items()}  # Move to device

    # Predict categories
    with torch.no_grad():
        outputs = model(**single_movie_token)
        logits = outputs.logits
        predicted_categories = torch.sigmoid(logits) > 0.5  # Threshold at 0.5

    # Convert back to label names
    predicted_labels = mlb.inverse_transform(predicted_categories.cpu().numpy())
    color = random.choice(colors)
    # Print actual vs predicted genres
    print(f"{color}Movie: {single_movie_text}{Style.RESET_ALL}")
    print(f"{color}Actual Genres: {single_movie_actual_genres}{Style.RESET_ALL}")
    print(f"{color}Predicted Genres: {predicted_labels}{Style.RESET_ALL}")

for i in range(0,20):    
    movie_index = random.randint(0, len(df) - 1)
# Test a specific movie by index (change index as needed)
    test_single_movie(movie_index)  # You can change the index to test different movies
