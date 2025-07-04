import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load original dataset
df = pd.read_csv(r"C:\Users\linda\Desktop\test.csv")

# Define analyzer
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update({"meeting": 0, "update": 0, "urgent": -1.5, "approved": 1.2})


# Define function to get sentiment
def classify_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.3:
        return "Positive", score
    elif score <= -0.3:
        return "Negative", score
    else:
        return "Neutral", score

# Apply sentiment classification and polarity extraction
df[["sentiment", "polarity"]] = df["body"].apply(
    lambda x: pd.Series(classify_sentiment(x))
)

# Post-process neutral phrases
neutral_phrases = ["please review", "kind regards", "as per"]
df["sentiment"] = df.apply(
    lambda row: "Neutral" if any(
        phrase in row["body"].lower() for phrase in neutral_phrases
    ) else row["sentiment"],
    axis=1
)

# Save to another CSV dataset (now includes both 'sentiment' and 'polarity' columns)
output_file = "labeled_emails_with_polarity.csv"
df.to_csv(output_file, index=False)
print(f"Labeled data saved to '{output_file}'")

# Validation: Check distribution and borderline cases
print("\nSentiment Distribution")
print(df["sentiment"].value_counts())

# Sample borderline cases for manual review
borderline = df[
    (df["polarity"].between(0.2, 0.3)) |
    (df["polarity"].between(-0.3, -0.2))
].sample(5, random_state=42)  # Random but reproducible
print("\nBorderline Cases (Manual Review)")
print(borderline[["body", "polarity", "sentiment"]].to_string())

