import pandas as pd

# Sample data
df = pd.read_csv(r"C:/Users/linda/Desktop/monthly_sentiment_scores.csv")

# For each month, get top 3 positive and top 3 negative employees
months = df['month'].unique()

result_list = []

for month in months:
    df_month = df[df['month'] == month]

    # Top 3 positive sentiment scores
    top_positive = df_month.nlargest(3, 'monthly_sentiment_score').copy()
    top_positive['rank_type'] = 'top_positive'

    # Top 3 negative sentiment scores
    top_negative = df_month.nsmallest(3, 'monthly_sentiment_score').copy()
    top_negative['rank_type'] = 'top_negative'

    # Add month column explicitly (if needed)
    top_positive['month'] = month
    top_negative['month'] = month

    # Append to list
    result_list.append(top_positive)
    result_list.append(top_negative)

# Combine all results into one DataFrame
result_df = pd.concat(result_list)

# Save to CSV
result_df.to_csv('employee_sentiment_rankings.csv', index=False)

print("Saved rankings to employee_sentiment_rankings.csv")
