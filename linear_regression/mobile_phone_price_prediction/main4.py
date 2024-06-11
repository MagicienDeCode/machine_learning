import pandas as pd

df = pd.read_csv('mobile_phone_price_prediction.csv')

df.drop(['Name','Unnamed: 0'],axis=1,inplace=True)
df['company'] = df['company'].str.lower()

# Android_version
fix_android_version_condition = df['External_Memory'].str.contains('Android', na=False)
temp_fix_android_version = df['External_Memory'].where(fix_android_version_condition)
temp_fix_android_version = temp_fix_android_version.str.extract(r'(\d+)')[0]
df['Android_version'] = df['Android_version'].combine_first(temp_fix_android_version)
df['Android_version'] = df['Android_version'].str.extract(r'(\d+)').astype(float)


fix_huawei_version_condition = df['External_Memory'].str.contains('HarmonyOS', na=False)
temp_fix_huawei_version = df['External_Memory'].where(fix_huawei_version_condition)
temp_fix_huawei_version = temp_fix_huawei_version.str.extract(r'(\d+)')[0]
df['HarmonyOS_version'] = temp_fix_huawei_version


comany_android_version = df.groupby('company')['Android_version'].mean().items()
comany_android_version_mapping = {k:v for k,v in comany_android_version}

#filtered_df = df[df['Android_version'].isna() & df['HarmonyOS_version'].isna()]
#print(filtered_df['company'].value_counts())
condition = df['Android_version'].isna() & df['HarmonyOS_version'].isna()
df.loc[condition, 'Android_version'] = df.loc[condition, 'company'].map(comany_android_version_mapping)

#filtered_df = df[df['Android_version'].isna() & df['HarmonyOS_version'].isna()]
#df = df.dropna(subset=['Android_version', 'HarmonyOS_version'], how='all')
df.fillna({'Android_version':0},inplace=True)
df.fillna({'HarmonyOS_version':0},inplace=True)

#print(df.isnull().sum())

# Inbuild_memory
#print(df['Inbuilt_memory'].value_counts(dropna=False))
condition = (df['Ram'].str.contains('inbuilt', na=False)) & (df['Inbuilt_memory'].isna())
df.loc[condition, 'Inbuilt_memory'] = df.loc[condition, 'Ram']
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.replace("1 TB","1024 GB")
df = df.dropna(subset=['Inbuilt_memory'])
df = df[df['Inbuilt_memory'].str.contains("inbuilt")]
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract(r'(\d+)').astype(int)
#print(df['Inbuilt_memory'].value_counts())

# Ram
"""
print(df['Ram'].value_counts())
filtered_df = df[df['Inbuilt_memory'] == 256]
print(filtered_df['Ram'].value_counts().idxmax()) # 12 GB RAM
filtered_df = df[df['Inbuilt_memory'] == 128]
print(filtered_df['Ram'].value_counts().idxmax()) # 8 GB RAM
filtered_df = df[df['Inbuilt_memory'] == 512]
print(filtered_df['Ram'].value_counts().idxmax()) # 12 GB RAM
"""

ram_patterns = {
    '256 GB inbuilt': '12 GB RAM',
    '512 GB inbuilt': '12 GB RAM',
    '128 GB inbuilt': '8 GB RAM'
}
df['Ram'] = df['Ram'].str.strip().replace(ram_patterns)
df['Ram'] = df['Ram'].str.split(' ', expand=True)[0].astype(float)


df['Battery'] = df['Battery'].str.split(' ', expand=True)[0].astype(int)
df['Display'] = df['Display'].str.split(' ', expand=True)[0].astype(float)


# Fast Charging

df['fast_charging'] = df['fast_charging'].str.extract(r'(\d+)').astype(float)
#print(df['fast_charging'].value_counts())
#filtered_df = df[df['fast_charging'].isna()]

comany_fast_chargings = df.groupby('company')['fast_charging'].mean().items()
comany_fast_charging_mapping = {k:v for k,v in comany_fast_chargings}

condition = df['fast_charging'].isna()  # 判断列 'B' 是否为空
df.loc[condition, 'fast_charging'] = df.loc[condition, 'company'].map(comany_fast_charging_mapping)

# Processor
#filtered_df = df[df['Processor'].isna()]
#print(filtered_df)
comany_processor = df.groupby('company')['Processor'].apply(lambda x: x.mode()[0]).items()
comany_processor_mapping = {k:v for k,v in comany_processor}
condition = df['Processor'].isna() 
df.loc[condition, 'Processor'] = df.loc[condition, 'company'].map(comany_processor_mapping)
print(df.isnull().sum())
print(df.shape)


# fix camera data
print(df.sample(10))