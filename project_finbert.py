from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# Load FinBERT model
classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# Load dataset
df = pd.read_csv(
    "all-data.csv",
    encoding="latin-1",
    names=["sentiment", "headline"]
)

# Use first 50 rows
df = df.head(50)

# Predict sentiment
df["predicted_sentiment"] = df["headline"].apply(
    lambda x: classifier(x)[0]["label"]
)

# Print results
print(df.head())

# Count sentiments
# Count sentiments
sentiment_counts = df["predicted_sentiment"].value_counts()

# Better graph size
plt.figure(figsize=(8,5))

# Plot graph
sentiment_counts.plot(
    kind="bar",
    color=["green", "blue", "red"]
)

# Titles
plt.title("Financial News Sentiment Analysis", fontsize=16)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Count", fontsize=12)

# Rotate labels
plt.xticks(rotation=0)

# Add numbers on top of bars
for i, v in enumerate(sentiment_counts):
    plt.text(i, v + 0.5, str(v), ha='center')


plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()