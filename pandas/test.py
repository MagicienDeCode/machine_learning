import pandas as pd

df = pd.read_csv('mobile_phone_price_prediction.csv')

# Display the first few rows of the DataFrame
#print(df.head())

# Display the summary information of the DataFrame, how many rows and columns it has, and the data types of each column
#print(df.info())

df.drop(['Unnamed: 0', 'Name'], axis=1, inplace=True)
#print(df.info())

#print(df['company'].value_counts(dropna=False))
df['company'] = df['company'].str.lower()
#print(df['company'].value_counts(dropna=False))

# Android version
#print(df['Android_version'].value_counts(dropna=False))
#print(df['External_Memory'].value_counts(dropna=False))

# extract the android version from column 'External_Memory' and then set it to the column 'Android_version'
external_memeroy_contains_android = df['External_Memory'].str.contains('Android', na=False)
#print(external_memeroy_contains_android.value_counts(dropna=False))
android_version_in_external_memory = df['External_Memory'].where(external_memeroy_contains_android)
#print(android_version_in_external_memory.value_counts(dropna=False))
#.str.extract(r'(\d+)') Uses a regular expression to find one or more digits (\d+) in each string.
android_version_in_external_memory = android_version_in_external_memory.str.extract(r'(\d+)')[0]
#print(android_version_in_external_memory.value_counts(dropna=False))
df['Android_version'] = df['Android_version'].combine_first(android_version_in_external_memory)
#print(df['Android_version'].value_counts(dropna=False))
df['Android_version'] = df['Android_version'].str.extract(r'(\d+)')[0].astype(float)
print(df['Android_version'].value_counts(dropna=False))

