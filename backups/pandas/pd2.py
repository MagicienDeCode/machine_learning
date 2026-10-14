import pandas as pd

# 创建示例DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)

print(df)
# 删除行（行索引为1）
df_row_dropped = df.drop(1, axis=0)
print("删除行后的DataFrame：")
print(df_row_dropped)

# 删除列（列标签为'B'）
df_col_dropped = df.drop('B', axis=1)
print("\n删除列后的DataFrame：")
print(df_col_dropped)