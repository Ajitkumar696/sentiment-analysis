
import pandas as pd
import re
import emoji
from pathlib import Path
from bs4 import BeautifulSoup
import string

# Locate the project root and dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "IMDB_cleaned.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

def clean_text(text):
    # Handle missing values
    if pd.isna(text):
        return ""

    # Ensure the review is a string
    text = str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 4. Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # 5. Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning only to the review column
df["cleaned_review"] = df["review"].apply(clean_text)

# Create the output directory if needed
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save cleaned data
df.to_csv(OUTPUT_PATH, index=False)

# Compare examples
for i in range(3):
    print(f"\nOriginal review {i + 1}:")
    print(df.loc[i, "review"])

    print(f"\nCleaned review {i + 1}:")
    print(df.loc[i, "cleaned_review"])

print("\nDataset shape:", df.shape)
print("Cleaned file saved to:", OUTPUT_PATH)
