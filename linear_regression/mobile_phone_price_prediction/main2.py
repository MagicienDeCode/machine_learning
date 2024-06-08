import pandas as pd

df = pd.read_csv('mobile_phone_price_prediction.csv')

df.drop('Android_version',axis=1,inplace=True)
df.drop(['Name','Processor_name','Unnamed: 0'],axis=1,inplace=True)


patterns = ['Helio G90T', '6000 mAh Battery with 22.5W Fast Charging']
df = df[~df['Ram'].str.contains('|'.join(patterns))]
df['Ram'] = df['Ram'].str.split(' ', expand=True)[0].astype(float)

df['Battery'] = df['Battery'].str.split(' ', expand=True)[0].astype(int)
df['Display'] = df['Display'].str.split(' ', expand=True)[0].astype(float)

df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract(r'(\d+)').astype(float)

df = df.dropna(subset=['Inbuilt_memory','fast_charging','Processor'])

df['fast_charging'] = df['fast_charging'].str.extract(r'(\d+)').astype(float)
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='most_frequent')
df['fast_charging']=imputer.fit_transform(df[['fast_charging']]).ravel()

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
    'Octa Core': '8',
    'Nine Core':'9',
    'Deca Core':'10',
    'Quad Core':'4',
    '2 GHz Processor':'1',
    '1.8 GHz Processor':'1',
    '2.3 GHz Processor':'1'
}
df['Processor_Encoded'] = df['Processor'].replace(processor_mapping).astype(int)

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
df['Sim_encoded'] = df['No_of_sim'].str.strip().replace(sim_importance_mapping).astype(int)
df.drop(['No_of_sim'],axis=1,inplace=True)


patterns = ['Foldable Display, Dual Display']
df = df[~df['Camera'].str.contains('|'.join(patterns))]
import re
def extract_back_camera_mp(camera_str):
    back_camera_part = camera_str.split('&')[0]
    if back_camera_part:
        back_cameras = re.findall(r'(\d+) MP', back_camera_part)
        return sum(map(int, back_cameras))
    return 0
def extract_front_camera_mp(camera_str):
    front_camera_part = re.search(r'(\d+) MP Front', camera_str)
    if front_camera_part:
        return int(front_camera_part.group(1))
    return 0

df['Front_Camera_MP'] = df['Camera'].apply(extract_front_camera_mp)
df['Back_Camera_MP'] = df['Camera'].apply(extract_back_camera_mp)


def extract_resolution_pixels(resolution_str):
    match = re.search(r'(\d+) x (\d+)', resolution_str)
    if match:
        width, height = map(int, match.groups())
        return width * height
    return 0

df['Total_Pixels'] = df['Screen_resolution'].apply(extract_resolution_pixels)
df.drop(['Screen_resolution'],axis=1,inplace=True)

df.drop(columns=['Camera','External_Memory','Processor','company'],inplace=True)

df['Price'] = df['Price'].str.replace(',', '')
df['Price'] = df['Price'].str[:-2] + '.' + df['Price'].str[-2:]
df['Price'] = df['Price'].astype(float)

print(df.info())
print(df.head())
print(df.isnull().sum())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

# 特征和标签
X = df.drop(columns=['Price'])
y = df['Price']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

hbr = HistGradientBoostingRegressor()
hbr.fit(X_train, y_train)
print(hbr.score(X_test, y_test))


from sklearn.ensemble import ExtraTreesRegressor
import seaborn as sns

extr = ExtraTreesRegressor().fit(X_train, y_train)
print(extr.score(X_test, y_test))
y_pred = extr.predict(X_test)
check = pd.DataFrame({'y_test': y_test, 'y_pred' : y_pred})
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scatterplot
sns.scatterplot(check, ax=axes[0])
axes[0].set_title('Scatterplot')

# Histplot
sns.histplot(check, kde=True, ax=axes[1])
axes[1].set_title('Histplot')

# Boxplot
sns.boxplot(check, ax=axes[2])
axes[2].set_title('Boxplot')

plt.tight_layout()
plt.show()

import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
print(data)
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)