# Employee-Sentiment-Analysis
# Brief Introduction
This project involves analyzing an unlabeled dataset of employee messages to assess sentiment and engagement. The project is divided into several distinct tasks, each focusing on a different aspect of data analysis and model development. 

# The main goal is to evaluate employee sentiment and engagement by performing the following:
·	Sentiment Labeling: Automatically label each message as Positive, Negative, or Neutral.
·	Exploratory Data Analysis (EDA): Analyze and visualize the data to understand its structure and underlying trends.
·	Employee Score Calculation: Compute a monthly sentiment score for each employee based on their messages.
·	Employee Ranking: Identify and rank employees by their sentiment scores.
·	Flight Risk Identification: A Flight risk is any employee who has sent 4 or more negative mails in a given month.
·	Predictive Modeling: Develop a linear regression model to further analyze sentiment trends.

# Coding Language
Python is used and Pytorch or sklearn library are used for AI modeling

# Summary
1. The top three positive and negative employees each month in 2010 and 2011 are listed below:
![image](https://github.com/user-attachments/assets/46cd786a-1495-4da3-b72b-a1ca77c914a3)
![image](https://github.com/user-attachments/assets/4fe04570-d6d8-46a2-b625-a5293792be16)


![image](https://github.com/user-attachments/assets/ad568766-38dd-4e7c-a8c9-67cfabe53d11)

2. The list of employees flagged as flight risks
   bobette.riner@ipgdirect.com
   johnny.palmer@enron.com
   lydia.delgado@enron.com
   patti.thompson@enron.com
   rhonda.denton@enron.com
   sally.beck@enron.com
   
4. Key insights and recommendations
· The data set contains 2191 objects, 4 columns that record the subject, from, body, and date of each employee.
· Among all of the records, there are 196 negative messages, 1019 neutral messages, and 976 positive messages. The number of positive messages increases as time goes.
· This project use linear regression model for sentiment trends analysis and prediction. Message count, average word count and keyword occurrance all positively affect the sentiment score of an employee.
· The linear regression model explains about 41% of the variance, other non-linear model may increase this R-squared value.
· Other features can be further considered to affect the sentiment score.

