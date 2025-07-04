import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\linda\Desktop\labeled_VADER_with_polarity.csv")

# Ensure 'date' is datetime and drop missing values (no missing data from task 2)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date', 'from', 'sentiment'])

# Map sentiment to scores
sentiment_score_map = {
    'Positive': 1,
    'Neutral': 0,
    'Negative': -1
}
df['score'] = df['sentiment'].map(sentiment_score_map)

# Create a 'month' column from the date
df['month'] = df['date'].dt.to_period('M')

# Group by employee ('from') and month, then sum the scores
monthly_scores = df.groupby(['from', 'month'])['score'].sum().reset_index()

# Rename columns for clarity
monthly_scores.columns = ['employee', 'month', 'monthly_sentiment_score']

# Display result
print(monthly_scores.head())

# Save to CSV
monthly_scores.to_csv('monthly_sentiment_scores.csv', index=False)
