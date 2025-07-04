import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\linda\Desktop\labeled_VADER_with_polarity.csv")

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

# Filter for only negative sentiment emails
neg_df = df[df['sentiment'] == 'Negative'].copy()

# Sort the data by sender and date
neg_df.sort_values(by=['from', 'date'], inplace=True)

# Function to check if any 30-day window has 4+ negative emails
def has_30day_window_with_4_negatives(group):
    dates = group['date'].tolist()
    for i in range(len(dates)):
        count = 1
        for j in range(i + 1, len(dates)):
            if (dates[j] - dates[i]).days <= 30:
                count += 1
                if count >= 4:
                    return True
            else:
                break
    return False

# Apply the function to each employee group
flight_risk_flags = neg_df.groupby('from').apply(has_30day_window_with_4_negatives)

# Get the list of employees who are flight risks
at_risk_employees = flight_risk_flags[flight_risk_flags].index.tolist()

# Optional – Save results to CSV
#pd.DataFrame({'employee': at_risk_employees}).to_csv('flight_risk_employees.csv', index=False)

# Print the result
print("Employees at flight risk:")
for emp in at_risk_employees:
    print(emp)
