# pip install numpy pandas scikit-learn

import pandas as pd

df = pd.read_csv('test.csv')
"""
sep: 指定分隔符，默认为逗号（,）。
header: 指定哪一行作为列名，默认为0（第一行）。
names: 指定列名列表。
index_col: 指定哪一列作为索引列。
usecols: 指定要读取的列。
dtype: 指定列的数据类型。
na_values: 指定哪些值表示缺失值。

"""

# 打印前五行数据
print("前五行数据：")
print(df.head())

# 打印数据摘要信息
print("\n数据摘要信息：")
print(df.info())

# 打印描述性统计
print("\n描述性统计：")
print(df.describe())

