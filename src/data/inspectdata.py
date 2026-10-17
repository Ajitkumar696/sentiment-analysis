
import pandas as pd
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Build the dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nFirst 5 reviews:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nReview length statistics:")
print(df["review"].str.len().describe())
