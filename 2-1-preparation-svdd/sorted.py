import pandas as pd

df_inter = pd.read_csv("dti_project\DTI\phase-05-data-balancing\df_NR_ADEFG.csv") 

# Sort dataframe by the label column
df_sorted = df_inter.sort_values(by='label', ascending=False) 
# print(df_sorted['label'])
# Save sorted dataframe
df_sorted.to_csv('dti_project\DTI\phase-05-data-balancing\df_NR_ADEFG_sorted.csv', index=False)
