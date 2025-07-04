import pandas as pd

df = pd.read_csv('mobile_phone_price_prediction.csv')

df.drop(['Name','Unnamed: 0'],axis=1,inplace=True)
#print(df.shape)
#print(df.isnull().sum())

# drop empty lines
df = df.dropna(subset=['Inbuilt_memory','fast_charging','Processor'])

# Android_version
df['Android_version'] = df['Android_version'].str.extract(r'(\d+)').astype(float)
#print(df['Android_version'].value_counts())
df_android_version_without_max = df[df['Android_version'] != 13.0]
mean_value = df_android_version_without_max['Android_version'].mean()
df['Android_version'] = df['Android_version'].fillna(mean_value)
#print(df['Android_version'].value_counts())
#print(df.isnull().sum())

# No_of_sim
df = df[~df['No_of_sim'].str.contains('No Sim')]
#print(df['No_of_sim'].value_counts())

sim_columns = ['Dual Sim', 'Single Sim', '3G', '4G', '5G', 'VoLTE', 'Vo5G']

for scol in sim_columns:
    df['Sim_'+scol.lower().replace(" ","_")] = df['No_of_sim'].apply(lambda x: 1 if scol in x else 0)

df.drop('No_of_sim',axis=1,inplace=True)


df['Battery'] = df['Battery'].str.split(' ', expand=True)[0].astype(int)
df['Display'] = df['Display'].str.split(' ', expand=True)[0].astype(float)
df['Ram'] = df['Ram'].str.split(' ', expand=True)[0].astype(int)

df['Inbuilt_memory'] = df['Inbuilt_memory'].str.replace("1 TB","1024 GB")
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract(r'(\d+)').astype(int)

#print(df['Camera'].value_counts())
#print(df.sample(10))

#filtered_df.to_csv("test.csv", index=False)