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

employee	month	monthly_sentiment_score	rank_type
kayne.coulter@enron.com	2010/1/1	5	top_positive
patti.thompson@enron.com	2010/1/1	5	top_positive
don.baughman@enron.com	2010/1/1	4	top_positive
rhonda.denton@enron.com	2010/1/1	0	top_negative
bobette.riner@ipgdirect.com	2010/1/1	1	top_negative
john.arnold@enron.com	2010/1/1	2	top_negative
bobette.riner@ipgdirect.com	2010/2/1	7	top_positive
john.arnold@enron.com	2010/2/1	7	top_positive
don.baughman@enron.com	2010/2/1	6	top_positive
sally.beck@enron.com	2010/2/1	0	top_negative
kayne.coulter@enron.com	2010/2/1	1	top_negative
lydia.delgado@enron.com	2010/2/1	1	top_negative
lydia.delgado@enron.com	2010/3/1	8	top_positive
sally.beck@enron.com	2010/3/1	8	top_positive
bobette.riner@ipgdirect.com	2010/3/1	3	top_positive
rhonda.denton@enron.com	2010/3/1	0	top_negative
eric.bass@enron.com	2010/3/1	1	top_negative
kayne.coulter@enron.com	2010/3/1	1	top_negative
don.baughman@enron.com	2010/4/1	5	top_positive
johnny.palmer@enron.com	2010/4/1	4	top_positive
john.arnold@enron.com	2010/4/1	3	top_positive
eric.bass@enron.com	2010/4/1	1	top_negative
bobette.riner@ipgdirect.com	2010/4/1	2	top_negative
lydia.delgado@enron.com	2010/4/1	2	top_negative
don.baughman@enron.com	2010/5/1	8	top_positive
patti.thompson@enron.com	2010/5/1	7	top_positive
sally.beck@enron.com	2010/5/1	6	top_positive
johnny.palmer@enron.com	2010/5/1	0	top_negative
bobette.riner@ipgdirect.com	2010/5/1	1	top_negative
john.arnold@enron.com	2010/5/1	1	top_negative
don.baughman@enron.com	2010/6/1	7	top_positive
john.arnold@enron.com	2010/6/1	5	top_positive
johnny.palmer@enron.com	2010/6/1	5	top_positive
bobette.riner@ipgdirect.com	2010/6/1	0	top_negative
kayne.coulter@enron.com	2010/6/1	0	top_negative
lydia.delgado@enron.com	2010/6/1	0	top_negative
lydia.delgado@enron.com	2010/7/1	9	top_positive
bobette.riner@ipgdirect.com	2010/7/1	6	top_positive
eric.bass@enron.com	2010/7/1	6	top_positive
johnny.palmer@enron.com	2010/7/1	0	top_negative
rhonda.denton@enron.com	2010/7/1	0	top_negative
don.baughman@enron.com	2010/7/1	1	top_negative
sally.beck@enron.com	2010/8/1	13	top_positive
john.arnold@enron.com	2010/8/1	4	top_positive
kayne.coulter@enron.com	2010/8/1	4	top_positive
bobette.riner@ipgdirect.com	2010/8/1	0	top_negative
eric.bass@enron.com	2010/8/1	1	top_negative
patti.thompson@enron.com	2010/8/1	1	top_negative
patti.thompson@enron.com	2010/9/1	6	top_positive
bobette.riner@ipgdirect.com	2010/9/1	4	top_positive
johnny.palmer@enron.com	2010/9/1	4	top_positive
don.baughman@enron.com	2010/9/1	1	top_negative
kayne.coulter@enron.com	2010/9/1	1	top_negative
lydia.delgado@enron.com	2010/9/1	1	top_negative
lydia.delgado@enron.com	2010/10/1	7	top_positive
eric.bass@enron.com	2010/10/1	6	top_positive
john.arnold@enron.com	2010/10/1	6	top_positive
don.baughman@enron.com	2010/10/1	0	top_negative
sally.beck@enron.com	2010/10/1	0	top_negative
kayne.coulter@enron.com	2010/10/1	1	top_negative
lydia.delgado@enron.com	2010/11/1	5	top_positive
don.baughman@enron.com	2010/11/1	3	top_positive
kayne.coulter@enron.com	2010/11/1	3	top_positive
bobette.riner@ipgdirect.com	2010/11/1	0	top_negative
rhonda.denton@enron.com	2010/11/1	0	top_negative
john.arnold@enron.com	2010/11/1	1	top_negative
john.arnold@enron.com	2010/12/1	8	top_positive
lydia.delgado@enron.com	2010/12/1	5	top_positive
sally.beck@enron.com	2010/12/1	5	top_positive
bobette.riner@ipgdirect.com	2010/12/1	0	top_negative
johnny.palmer@enron.com	2010/12/1	0	top_negative
rhonda.denton@enron.com	2010/12/1	0	top_negative
bobette.riner@ipgdirect.com	2011/1/1	10	top_positive
sally.beck@enron.com	2011/1/1	6	top_positive
johnny.palmer@enron.com	2011/1/1	5	top_positive
john.arnold@enron.com	2011/1/1	0	top_negative
kayne.coulter@enron.com	2011/1/1	0	top_negative
patti.thompson@enron.com	2011/1/1	1	top_negative
john.arnold@enron.com	2011/2/1	10	top_positive
lydia.delgado@enron.com	2011/2/1	6	top_positive
kayne.coulter@enron.com	2011/2/1	4	top_positive
rhonda.denton@enron.com	2011/2/1	0	top_negative
bobette.riner@ipgdirect.com	2011/2/1	1	top_negative
eric.bass@enron.com	2011/2/1	1	top_negative
lydia.delgado@enron.com	2011/3/1	7	top_positive
bobette.riner@ipgdirect.com	2011/3/1	6	top_positive
john.arnold@enron.com	2011/3/1	6	top_positive
sally.beck@enron.com	2011/3/1	-2	top_negative
johnny.palmer@enron.com	2011/3/1	0	top_negative
don.baughman@enron.com	2011/3/1	2	top_negative
bobette.riner@ipgdirect.com	2011/4/1	9	top_positive
johnny.palmer@enron.com	2011/4/1	9	top_positive
eric.bass@enron.com	2011/4/1	6	top_positive
john.arnold@enron.com	2011/4/1	1	top_negative
sally.beck@enron.com	2011/4/1	1	top_negative
don.baughman@enron.com	2011/4/1	2	top_negative
lydia.delgado@enron.com	2011/5/1	9	top_positive
john.arnold@enron.com	2011/5/1	6	top_positive
eric.bass@enron.com	2011/5/1	5	top_positive
patti.thompson@enron.com	2011/5/1	-1	top_negative
bobette.riner@ipgdirect.com	2011/5/1	1	top_negative
kayne.coulter@enron.com	2011/5/1	1	top_negative
johnny.palmer@enron.com	2011/6/1	15	top_positive
eric.bass@enron.com	2011/6/1	10	top_positive
bobette.riner@ipgdirect.com	2011/6/1	5	top_positive
don.baughman@enron.com	2011/6/1	0	top_negative
kayne.coulter@enron.com	2011/6/1	0	top_negative
rhonda.denton@enron.com	2011/6/1	1	top_negative
patti.thompson@enron.com	2011/7/1	11	top_positive
sally.beck@enron.com	2011/7/1	10	top_positive
john.arnold@enron.com	2011/7/1	6	top_positive
don.baughman@enron.com	2011/7/1	0	top_negative
kayne.coulter@enron.com	2011/7/1	0	top_negative
lydia.delgado@enron.com	2011/7/1	0	top_negative
lydia.delgado@enron.com	2011/8/1	7	top_positive
john.arnold@enron.com	2011/8/1	6	top_positive
sally.beck@enron.com	2011/8/1	6	top_positive
kayne.coulter@enron.com	2011/8/1	0	top_negative
rhonda.denton@enron.com	2011/8/1	1	top_negative
bobette.riner@ipgdirect.com	2011/8/1	2	top_negative
kayne.coulter@enron.com	2011/9/1	10	top_positive
rhonda.denton@enron.com	2011/9/1	7	top_positive
don.baughman@enron.com	2011/9/1	6	top_positive
john.arnold@enron.com	2011/9/1	1	top_negative
sally.beck@enron.com	2011/9/1	1	top_negative
johnny.palmer@enron.com	2011/9/1	3	top_negative
john.arnold@enron.com	2011/10/1	6	top_positive
kayne.coulter@enron.com	2011/10/1	6	top_positive
lydia.delgado@enron.com	2011/10/1	6	top_positive
bobette.riner@ipgdirect.com	2011/10/1	0	top_negative
johnny.palmer@enron.com	2011/10/1	0	top_negative
rhonda.denton@enron.com	2011/10/1	0	top_negative
patti.thompson@enron.com	2011/11/1	9	top_positive
kayne.coulter@enron.com	2011/11/1	7	top_positive
bobette.riner@ipgdirect.com	2011/11/1	6	top_positive
eric.bass@enron.com	2011/11/1	1	top_negative
lydia.delgado@enron.com	2011/11/1	1	top_negative
rhonda.denton@enron.com	2011/11/1	1	top_negative
kayne.coulter@enron.com	2011/12/1	5	top_positive
patti.thompson@enron.com	2011/12/1	5	top_positive
don.baughman@enron.com	2011/12/1	4	top_positive
bobette.riner@ipgdirect.com	2011/12/1	0	top_negative
lydia.delgado@enron.com	2011/12/1	0	top_negative
johnny.palmer@enron.com	2011/12/1	1	top_negative
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

