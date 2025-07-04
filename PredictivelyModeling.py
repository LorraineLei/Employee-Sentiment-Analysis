import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error



# Load datasets
messages_df = pd.read_csv(r"C:/Users/linda/Desktop/labeled_VADER_with_polarity.csv")
monthly_scores_df = pd.read_csv(r"C:/Users/linda/Desktop/monthly_sentiment_scores.csv")

# Preprocess date
messages_df['date'] = pd.to_datetime(messages_df['date'])
messages_df['month'] = messages_df['date'].dt.to_period('M')

# Feature Engineering
messages_df['message_length'] = messages_df['body'].apply(len)
messages_df['word_count'] = messages_df['body'].apply(lambda x: len(x.split()))
messages_df['thank_you'] = messages_df['body'].str.lower().str.count("thank you")
messages_df['sorry'] = messages_df['body'].str.lower().str.count("sorry")
messages_df['best_regards'] = messages_df['body'].str.lower().str.count("best regards")

# Aggregate features per employee per month
agg_df = messages_df.groupby(['employee', 'month']).agg({
    'body': 'count',  # message frequency
    'message_length': ['sum', 'mean'],
    'word_count': 'sum',
    'thank_you': 'sum',
    'sorry': 'sum',
    'best_regards': 'sum'
}).reset_index()

# Rename columns
agg_df.columns = ['employee', 'month', 'message_count', 'total_message_length', 'avg_message_length',
                  'total_word_count', 'thank_you_count', 'sorry_count', 'best_regards_count']

# Ensure month is same format for merging
monthly_scores_df['month'] = pd.to_datetime(monthly_scores_df['month']).dt.to_period('M')

# Merge sentiment score
merged_df = pd.merge(agg_df, monthly_scores_df, on=['employee', 'month'])

# Prepare data for modeling
features = ['message_count', 'total_message_length', 'avg_message_length',
            'total_word_count', 'thank_you_count', 'sorry_count', 'best_regards_count']
X = merged_df[features]
y = merged_df['monthly_sentiment_score']

# Compute correlation matrix
corr_matrix = merged_df[features + ['monthly_sentiment_score']].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap of Features and Sentiment Score")
plt.tight_layout()
plt.show()


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared Score (R^2)", r2)

# Printout coefficients
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
print(coef_df)

# Visualization: Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Sentiment Score")
plt.ylabel("Predicted Sentiment Score")
plt.title("Actual vs Predicted Sentiment Scores")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=20, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Sentiment Score")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()