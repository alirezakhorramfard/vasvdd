import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_samples = pd.read_csv("dti_project\\DTI\\phase-04-feature-combination\\NR_all_interactions.csv")
# Remove unnamed columns
df_samples.drop(df_samples.columns[df_samples.columns.str.contains('unnamed',case=False)], axis=1, inplace=True)

protein_aac = pd.read_csv("dti_project\DTI\\phase-03-feature-extraction\AAC\\df_NR_AAC.csv")
# Remove unnamed columns
protein_aac.drop(protein_aac.columns[protein_aac.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
protein_dde = pd.read_csv("dti_project\\DTI\\phase-03-feature-extraction\\DDE\\df_NR_DDE.csv")
# Remove unnamed columns
protein_dde.drop(protein_dde.columns[protein_dde.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
protein_psePSSM = pd.read_csv("dti_project\\DTI\\phase-03-feature-extraction\\PsePSSM\\df_NR_psePSSM.csv")
# Remove unnamed columns
protein_psePSSM.drop(protein_psePSSM.columns[protein_psePSSM.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# Read protein PseAACs
protein_pseAAC = pd.read_csv("dti_project\\DTI\\phase-03-feature-extraction\\PseAAC\\df_NR_pseAAC.csv")
# Remove unnamed columns
protein_pseAAC.drop(protein_pseAAC.columns[protein_pseAAC.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
#print(protein_pseAAC.head())
# Read protein CKSAAGPs
protein_cksaagp = pd.read_csv("dti_project\\DTI\\phase-03-feature-extraction\\CKSAAGP\\df_NR_CKSAAGP.csv")
# Remove unnamed columns
protein_cksaagp.drop(protein_cksaagp.columns[protein_cksaagp.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
drug_fp = pd.read_csv("dti_project\\DTI\\phase-03-feature-extraction\\FP2-fingerprints\\NR_drug_fp2.csv")
# Remove unnamed columns
drug_fp.drop(drug_fp.columns[drug_fp.columns.str.contains('unnamed', case=False)], axis=1, inplace = True)


# protein indexes
df_protein = df_samples.loc[:,'protein_no']

# drug indexes
df_drug = df_samples.loc[:,'drug_no']


# Add protein AACs to data
df_aac = []
for i in range(len(df_protein)):
    protein_no = df_protein.iloc[i]
    df_aac.append(protein_aac.iloc[protein_no])

df_aac = pd.DataFrame(df_aac)

# Add protein DDEs to data
df_dde = []

for i in range(len(df_protein)):
    protein_no = df_protein.iloc[i]
    df_dde.append(protein_dde.iloc[protein_no])

df_dde = pd.DataFrame(df_dde)


# Add protein PsePSSMs to data
df_psePSSM = []

for i in range(len(df_protein)):
    protein_no = df_protein.iloc[i]
    df_psePSSM.append(protein_psePSSM.iloc[protein_no])

df_psePSSM = pd.DataFrame(df_psePSSM)

# Add protein PseAACs to data
df_pseAAC = []

for i in range(len(df_protein)):
    protein_no = df_protein.iloc[i]
    df_pseAAC.append(protein_pseAAC.iloc[protein_no])

df_pseAAC = pd.DataFrame(df_pseAAC)



# Add protein CKSAAGPs to data
df_cksaagp = []

for i in range(len(df_protein)):
    protein_no = df_protein.iloc[i]
    df_cksaagp.append(protein_cksaagp.iloc[protein_no])

df_cksaagp = pd.DataFrame(df_cksaagp)
# Add drug fingerprints to data
df_fingers = []
for i in range(len(df_drug)):
    drug_no = df_drug.iloc[i]
    df_fingers.append(drug_fp.iloc[drug_no])


df_fingers = pd.DataFrame(df_fingers)

df_samples = pd.DataFrame(df_samples)

df_samples.reset_index(inplace=True, drop=True)
df_fingers.reset_index(inplace=True, drop=True)
df_aac.reset_index(inplace=True, drop=True)
df_dde.reset_index(inplace=True, drop=True)
df_psePSSM.reset_index(inplace=True, drop=True)
df_pseAAC.reset_index(inplace=True, drop=True)
df_cksaagp.reset_index(inplace=True, drop=True)
df_features = pd.concat([df_fingers, df_aac, df_dde,df_psePSSM, df_pseAAC,df_cksaagp], axis=1, sort=False)

# returns a numpy array
df = df_features.values
min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = df_features.columns

df_AB = pd.concat([df_samples, df_scaled], axis=1, sort=False)


print(len(df_AB))
print(df_AB.head())
df_AB.to_csv('df_NR_ADEFG.csv')