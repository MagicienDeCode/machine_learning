import pandas as pd

df = pd.read_csv('test.csv')
print(df)

# drop colunm with name "Unnamed: 0", axis=1 (foreach line)
df.drop("Unnamed: 0",axis = 1,inplace=True)
print(df)

# row, col
print(df.shape)

# random pick 3 lines
print(df.sample(3))

# group by a column
print(df['Processor_name'].value_counts())

# define patterns, for one column, drop line that patterns contain its value
patterns = ['Helio P35', 'Helio G88']
df = df[~df['Processor_name'].str.contains('|'.join(patterns))]

print(df)

