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
#print(df['Android_version'].value_counts(dropna=False))

external_memeroy_contains_harmonyos = df['External_Memory'].str.contains('HarmonyOS', na=False)
harmonyos_version_in_external_memory = df['External_Memory'].where(external_memeroy_contains_harmonyos)
harmonyos_version_in_external_memory = harmonyos_version_in_external_memory.str.extract(r'(\d+)')[0]
df['HarmonyOS_version'] = harmonyos_version_in_external_memory

#print(df['HarmonyOS_version'].value_counts(dropna=False))
#print(df['Android_version'].value_counts(dropna=False))

company_android_version = df.groupby('company')['Android_version'].mean().items()
company_android_version_mapping = {k:v for k,v in company_android_version}
#print(company_android_version_mapping)

condition = df['Android_version'].isna() & df['HarmonyOS_version'].isna()
df.loc[condition, 'Android_version'] = df.loc[condition, 'company'].map(company_android_version_mapping)

#print(df['Android_version'].value_counts(dropna=False))

df.fillna({'Android_version': 0}, inplace=True)
df['HarmonyOS_version'] = df['HarmonyOS_version'].fillna(0)

#print(df['HarmonyOS_version'].isna().sum())
#print(df['Android_version'].isna().sum())

df['HarmonyOS_version'] = df['HarmonyOS_version'].astype(int)
df['Android_version'] = df['Android_version'].round(1)

#print(df['HarmonyOS_version'].value_counts(dropna=False))
#print(df['Android_version'].value_counts(dropna=False))

#print(df.info())
#print(df['Inbuilt_memory'].value_counts(dropna=False))

# Inbuilt memory
condition = df['Ram'].str.contains('inbuilt', na=False) & df['Inbuilt_memory'].isna()
df.loc[condition, 'Inbuilt_memory'] = df.loc[condition, 'Ram']
#print(df['Inbuilt_memory'].value_counts(dropna=False))
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.replace("1 TB","1024 GB")
#print(df['Inbuilt_memory'].value_counts(dropna=False))
df = df.dropna(subset=['Inbuilt_memory'])
#print(df['Inbuilt_memory'].value_counts(dropna=False))
df = df[df['Inbuilt_memory'].str.strip() != 'Octa Core'] #df = df[df['Inbuilt_memory'].str.contains("inbuilt")]
#print(df['Inbuilt_memory'].value_counts(dropna=False))
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract(r'(\d+)')[0].astype(int)
#print(df['Inbuilt_memory'].value_counts(dropna=False))

#print(df.isnull().sum())

# Ram
#print(df['Ram'].value_counts(dropna=False))
ram_patterns = {
    '256 GB inbuilt': '12 GB RAM',
    '512 GB inbuilt': '12 GB RAM',
    '128 GB inbuilt': '8 GB RAM'
}
df['Ram'] = df['Ram'].str.strip().replace(ram_patterns)

print(df['Ram'].value_counts(dropna=False))

# Fast charging
print(df['fast_charging'].value_counts(dropna=False))
# convert fast charging to numeric values
# fill missing values with the mean of the company
# convert all values to integers