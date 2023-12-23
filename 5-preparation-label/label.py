import pandas as pd
df2 = pd.read_csv("DTI\\3-Data-preparation\\df_NR_ADEFG\\df_NR_ADEFG_1_5.csv")
print(df2.shape)
df = pd.read_csv("DTI\\4-dimension-reduction\\new\\df_NR_ADEFG_1_5.csv")
df['label'] = df2['label']
print(df.shape)
df.to_csv("DTI\\5-preparation-label\\new\\df_NR_ADEFG_1_5.csv")
