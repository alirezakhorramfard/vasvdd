import pandas as pd
df1 = pd.read_csv("DTI\\3-Data-preparation\\df_NR_ADEFG_label_1.csv")
print(df1.shape)
df2 = pd.read_csv("DTI\\2-svdd-deep\\df_NR_ADEFG\\df_NR_ADEFG_1_5.csv")

df2.insert(0, "label", 0)
df2.columns = df1.columns
print(df2.shape)
concatenated_df = pd.concat([df1, df2], axis=0)
print(concatenated_df.shape)
concatenated_df.to_csv('DTI\\3-Data-preparation\\df_NR_ADEFG\\df_NR_ADEFG_1_5.csv', index=False)
# import pandas as pd


# df1 = pd.read_csv("DTI\\2-1-preparation-svdd\\df_NR_ADEFG_sorted.csv")
# print(df1.shape)
# df1.drop(df1.columns[df1.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
# df1.drop(['drug_no', 'protein_no'], axis=1, inplace=True)
# df_first_rows = df1.head(2926)
# df_first_rows.to_csv('DTI\\3-Data-preparation\\df_NR_ADEFG_label_1.csv', index=False)