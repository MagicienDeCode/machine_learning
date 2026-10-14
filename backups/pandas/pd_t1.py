import pandas as pd

# read cvs file as a dataframe
df = pd.read_csv('mobile_phone_price_prediction.csv')

# print first 5 lines
print(df.head())
# print random 7 lines
print(df.sample(7))


print(df.info())

# print(df.describe())

# drop one column, axis=1 apply column
df.drop('Unnamed: 0',axis=1,inplace=True)


# value count info
print(df['Name'].value_counts())
print(df['Processor_name'].value_counts())

# drop multi columns
df.drop(['Name','Processor_name','Android_version'],axis=1,inplace=True)
print(df.info())

print(df['Rating'].value_counts())
print(df['Spec_score'].value_counts())

print(df['No_of_sim'].value_counts())

print("======================")
# count null values
print(df.isnull().sum())
# row, col
print(df.shape)

sim_importance_mapping = {
    'No Sim Supported,': '0',
    'Single Sim, 3G, 4G,': '1',
    'Dual Sim, 3G, VoLTE,': '2',
    'Single Sim, 3G, 4G, VoLTE,': '3',
    'Dual Sim, 3G, 4G,': '4',
    'Dual Sim, 3G, 4G, VoLTE,': '5',
    'Single Sim, 3G, 4G, 5G, VoLTE,': '6',
    'Dual Sim, 3G, 4G, 5G, VoLTE,': '7',
    'Single Sim, 3G, 4G, 5G, VoLTE, Vo5G,': '8',
    'Dual Sim, 3G, 4G, 5G, VoLTE, Vo5G,': '9'
}
