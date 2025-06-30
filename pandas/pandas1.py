import pandas as pd

df = pd.read_csv('mobile_phone_price_prediction.csv')

# Display the first few rows of the DataFrame
#print(df.head())

# Display the summary information of the DataFrame, how many rows and columns it has, and the data types of each column
#print(df.info())

#print(df.describe())
# Display the summary statistics of the 'Price' column
#print(df['Price'].describe())

# Display the unique values in the 'Price' column
#print(df['Price'].value_counts())