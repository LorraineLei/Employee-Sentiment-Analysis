import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\linda\Desktop\labeled_VADER_with_polarity.csv")  # Load your dataset

# Number of rows (records) and columns
print("Shape of the dataset:", df.shape)
print("Number of records:", len(df))

# Data types
print("\nData types:")
print(df.dtypes)

#find missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Ensure the 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ===== 1. Sentiment Label Distribution =====
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='sentiment', order=['Positive', 'Neutral', 'Negative'], palette='Set2')
plt.title('Sentiment Label Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Messages')
plt.tight_layout()
plt.show()

# ===== 2. Trend Over Time (Monthly) =====
# Convert dates to monthly period
df['month'] = df['date'].dt.to_period('M')

# Group by month and sentiment, count occurrences
sentiment_trend = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)

# ===== 3. Plot Monthly Sentiment Trend =====
plt.figure(figsize=(10, 6))
sentiment_trend.plot(kind='line', marker='o')
plt.title('Monthly Sentiment Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Messages')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

