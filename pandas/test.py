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

#df['Ram'] = df['Ram'].str.split(' ', expand=True)[0].astype(float)
df['Ram'] = df['Ram'].str.extract(r'(\d+)')[0].astype(int)
#print(df['Ram'].value_counts(dropna=False))


# Fast charging
df['fast_charging'] = df['fast_charging'].str.extract(r'(\d+)').astype(float)
comany_fast_chargings = df.groupby('company')['fast_charging'].mean().items()
comany_fast_charging_mapping = {k:v for k,v in comany_fast_chargings}
condition = df['fast_charging'].isna()
df.loc[condition, 'fast_charging'] = df.loc[condition, 'company'].map(comany_fast_charging_mapping)
# df['fast_charging'] = df['fast_charging'].fillna(df.groupby('company')['fast_charging'].transform('mean'))
df['fast_charging'] = df['fast_charging'].astype(int)
# convert fast charging to numeric values
# fill missing values with the mean of the company
# convert all values to integers
#print(df['fast_charging'].value_counts(dropna=False))

#df.head().to_csv('test.csv', index=False)

# Processor

#print(df['Processor'].value_counts(dropna=False))

"""
company_process = df.groupby('company')['Processor'].apply(lambda x: x.mode()[0]).items()
company_processor_mapping = {k:v for k,v in company_process}
print(company_processor_mapping)
condition = df['Processor'].isna()
df.loc[condition, 'Processor'] = df.loc[condition, 'company'].map(company_processor_mapping)
"""
df['Processor'] = df['Processor'].fillna(' Octa Core')
#print(df['Processor'].value_counts(dropna=False))

processor_mapping = {
    'Octa Core': 'Octa Core',
    'Octa Core Processor': 'Octa Core',
    'Quad Core': 'Quad Core',
    'Deca Core': 'Deca Core',
    'Deca Core Processor': 'Deca Core',
    'Nine-Cores': 'Nine Core',
    'Nine Core': 'Nine Core',
    'Nine Cores': 'Nine Core',
    '1.6 GHz Processor': '1.6 GHz Processor',
    '2 GHz Processor': '2 GHz Processor',
    '1.8 GHz Processor': '1.8 GHz Processor',
    '1.3 GHz Processor': '1.3 GHz Processor',
    '2.3 GHz Processor': '2.3 GHz Processor'
}
df['Processor'] = df['Processor'].str.strip().replace(processor_mapping)
processor_mapping = {
    'Octa Core': 8,
    'Nine Core':'9',
    'Deca Core':'10',
    'Quad Core':'4',
    '2 GHz Processor':'1',
    '1.8 GHz Processor':'1',
    '1.6 GHz Processor':'1',
    '1.3 GHz Processor':'1',
    '2.3 GHz Processor':'1'
}
df['Processor'] = df['Processor'].replace(processor_mapping).astype(int)
#print(df['Processor'].value_counts(dropna=False))

# screen resolution
#print(df['Screen_resolution'].value_counts(dropna=False))
import re
def extract_resolution(resolution):
    match = re.search(r'(\d+) x (\d+)',resolution)
    if match:
        int1 , int2 =map(int, match.groups())
        return int1 * int2 / 1000
    return 0

df['Screen_resolution'] = df['Screen_resolution'].apply(extract_resolution)
df = df[df['Screen_resolution'] != 0]
df['Screen_resolution'] = df['Screen_resolution'].astype(int)
#print(df['Screen_resolution'].value_counts(dropna=False))

#print(df['No_of_sim'].value_counts(dropna=False))
sim_columns = ['Dual Sim', 'Single Sim', '3G', '4G', '5G', 'VoLTE', 'Vo5G']
for scol in sim_columns:
    df['Sim_'+scol.lower().replace(" ","_")] = df['No_of_sim'].apply(lambda x: 1 if scol in x else 0)
df.drop('No_of_sim',axis=1,inplace=True)

#df.head().to_csv('test.csv', index=False)


df['Camera'] = df['Camera'].str.extract(r'(\d+)').astype(str)
comany_first_camera = df.groupby('company')['Camera'].apply(lambda x: x.mode()[0]).items()
comany_first_camera_mapping = {k:v for k,v in comany_first_camera}
condition = df['Camera'] == 'nan'
df.loc[condition, 'Camera'] = df.loc[condition, 'company'].map(comany_first_camera_mapping)
#print(df['Camera'].value_counts(dropna=False))
df['Camera'] = df['Camera'].astype(int)
#print(df.info())

df.drop(['Processor_name','company','External_Memory'],axis=1,inplace=True)

df['Battery'] = df['Battery'].str.extract(r'(\d+)').astype(int)
df['Display'] = df['Display'].str.extract(r'(\d+)').astype(float)
#df.head().to_csv('test.csv', index=False)


df['Price'] = df['Price'].str.replace(',','').astype(int)
#print(df['Price'].value_counts(dropna=False))

#print(df.isnull().sum())
#df.head().to_csv('test.csv', index=False)

