import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn import metrics
import os

warnings.filterwarnings('ignore')

root_ = "DTI\\2-1-preparation-svdd"
df_inter = pd.read_csv(os.path.join(root_, "df_NR_ADEFG_sorted.csv"))

df_inter.drop(df_inter.columns[df_inter.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
df_inter.drop(['drug_no', 'protein_no'], axis=1, inplace=True)
print(df_inter.shape)
def Speech_loader(train_batch, test_batch):
  X = df_inter.copy()
  y = X.iloc[:, 0].copy()
  X.drop(X.columns[0], axis=1, inplace=True)
  dataclass0=y[y==0].shape[0]
  dataclass1=y[y==1].shape[0]
  R = int((dataclass0 + dataclass1 )*0.13)
  X_test = X[:R]
  X_train = X[R:]
  y_test = y[:R]
  y_train = y[R:]
  train = torch.utils.data.TensorDataset(torch.Tensor(X_train.values))
  train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True,drop_last=True)
  test = torch.utils.data.TensorDataset(torch.Tensor(X_test.values),torch.Tensor(y_test.values))
  test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=False)

  return train_loader,test_loader,y_test


class DeepAE(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        
        self.input_size = 1074

        self.hidden_size = 512
        self.hidden_size_1=256
        self.hidden_size_2=128
        self.hidden_size_3=64
        self.hidden_size_4=32
        self.output_size = 1074
        self.input_layer = nn.BatchNorm1d(self.input_size)
        self.dropout = nn.Dropout(p=dropout_p)  
        # Encoder layers
        self.input_layer_1 = nn.Linear(self.input_size, self.hidden_size)
        self.input_layer_batch = nn.BatchNorm1d(self.hidden_size)
        self.dropout_1 = nn.Dropout(p=dropout_p) 
        self.enc_layer1 = nn.Linear(self.hidden_size, self.hidden_size_1) 
        self.enc_layer1_batch = nn.BatchNorm1d(self.hidden_size_1)
        self.dropout_2 = nn.Dropout(p=dropout_p) 
        self.enc_layer2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.enc_layer2_batch = nn.BatchNorm1d(self.hidden_size_2)
        self.dropout_3 = nn.Dropout(p=dropout_p) 
        self.enc_layer3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.enc_layer3_batch = nn.BatchNorm1d(self.hidden_size_3)
        self.dropout_4 = nn.Dropout(p=dropout_p) 
        self.enc_layer4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        
        # Decoder layers 
        self.dec_layer1 = nn.Linear(self.hidden_size_4, self.hidden_size_3)
        self.dec_layer1_batch = nn.BatchNorm1d(self.hidden_size_3)
        self.dropout_5 = nn.Dropout(p=dropout_p)
        self.dec_layer2 = nn.Linear(self.hidden_size_3, self.hidden_size_2)
        self.dec_layer2_batch = nn.BatchNorm1d(self.hidden_size_2) 
        self.dropout_6 = nn.Dropout(p=dropout_p)
        self.dec_layer3 = nn.Linear(self.hidden_size_2, self.hidden_size_1)
        self.dec_layer3_batch = nn.BatchNorm1d(self.hidden_size_1) 
        self.dropout_7 = nn.Dropout(p=dropout_p)
        self.dec_layer4 = nn.Linear(self.hidden_size_1, self.output_size)
        self.activation = nn.ReLU()
    def forward(self, x):

        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.activation(self.input_layer_1(x))
        x = self.input_layer_batch(x)
        x = self.dropout_1(x)
        x = self.activation(self.enc_layer1(x))
        x = self.enc_layer1_batch(x)
        x = self.dropout_2(x)
        x = self.activation(self.enc_layer2(x))
        x = self.enc_layer2_batch(x)
        x = self.dropout_3(x)
        x = self.activation(self.enc_layer3(x))
        x = self.enc_layer3_batch(x)
        x = self.dropout_4(x)
        x_1 = self.activation(self.enc_layer4(x))
        x=x_1
        # Decoder
        x = self.activation(self.dec_layer1(x))
        x = self.dec_layer1_batch(x)
        x = self.dropout_5(x)
        x = self.activation(self.dec_layer2(x))
        x = self.dec_layer2_batch(x)
        x = self.dropout_6(x)
        x = self.activation(self.dec_layer3(x))
        x = self.dec_layer3_batch(x)
        x = self.dropout_7(x)
        x = self.activation(self.dec_layer4(x))
        
        return x,x_1

def DASVDD_trainer(model, in_shape, code_size, C, train_loader, optimizer, update_center, criterion, Gamma, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=3, K=0.9):
    num_epochs = 10
    K = 0.9
    L1 = np.zeros(num_epochs)
    L2 = np.zeros(num_epochs)
    L3 = np.zeros(num_epochs)
    c_vals = np.zeros(num_epochs)
    data_scores = {}  
    for epoch in range(num_epochs):
        loss = 0
        aeloss = 0
        svddloss = 0

        for i, batch_features in enumerate(train_loader):
            if isinstance(batch_features, list):
                batch_features = batch_features[0]
        
            batch_features = batch_features.view(-1, in_shape).to(device)

            Num_batch = int(np.ceil(K * batch_features.size()[0]))
            optimizer.zero_grad()
            outputs,code = model(batch_features[:Num_batch,:])
            R = torch.sum((code.to(device) - C) ** 2, dim=1)[0]
            train_loss = criterion(outputs, batch_features[:Num_batch, :]) + Gamma * R
            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()
            aeloss += criterion(outputs, batch_features[:Num_batch, :]).item()
            svddloss += R.item()
            _,c_code = model(batch_features[Num_batch:,:])
            center = torch.mean(c_code, axis=0)
            center_loss = criterion(C, center)
            center_loss.backward()
            update_center.step()

            c_vals[epoch] += C[0]


            scores = torch.sum((code.to(device) - C) ** 2, dim=1).tolist()

            for j, score in enumerate(scores):
                data_index = i * train_loader.batch_size + j
                data_scores[data_index] = score

            
        c_vals[epoch] = c_vals[epoch] / len(train_loader)
        loss = loss / len(train_loader)
        aeloss = aeloss / len(train_loader)
        svddloss = svddloss / len(train_loader)

        L1[epoch] = loss
        L2[epoch] = aeloss
        L3[epoch] = svddloss


        sorted_data_scores = sorted(data_scores.items(), key=lambda x: x[1]) 
        X = 7381
        sorted_data_scores = sorted_data_scores[:X]
        index = [x for x, _ in sorted_data_scores]
        selected_data = [tuple(train_loader.dataset[i].numpy())
                          if isinstance(train_loader.dataset[i], torch.Tensor) else train_loader.dataset[i] for i in index]

        selected_data_scores_list = []
        for data in selected_data:
            selected_data_scores_list.append(data[0].tolist())

        
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {loss} ')

    return L1, L2, L3, c_vals, data_scores ,selected_data_scores_list,losses
def DASVDD_test(model,C,in_shape,Gamma,test_loader,labels,criterion,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

  with torch.no_grad():
    score = []
    i = 0
    for z, x_test in enumerate(test_loader):
      if isinstance(x_test, list):
          x_test = x_test[0]
      x_test =x_test.view(-1, in_shape).to(device)
      x_test_hat,code_test = model(x_test)
      loss = criterion(x_test_hat,x_test) + Gamma*torch.sum((code_test.to(device) - C) ** 2, dim=1)[0]
      score.append(loss.to("cpu").item())
      i+=1
      print(f"Step {z + 1}")
      print(loss)
    return metrics.roc_auc_score(labels,score)*100


def tune_gamma(DeepAE, in_shape, criterion, train_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), T=4):
    gamma = 0
    for k in range(T):
        R = 0
        RE = 0
        for i, batch_features in enumerate(train_loader):
            if isinstance(batch_features, list):
                batch_features = batch_features[0]
            batch_features = batch_features.view(-1, in_shape).to(device)
            outputs,code = model(batch_features)
            R += torch.sum((code.to(device)) ** 2, dim=1)[0]
            RE += criterion(outputs, batch_features)
            print(f"Step {i + 1}: R = {R.item()}, RE = {RE.item()}")
        R = R / len(train_loader)
        RE = RE / len(train_loader)
        gamma += RE / R
        print(f"Execution {k + 1}: gamma = {gamma.item()}")
    gamma = gamma / T
    gamma = gamma.detach().item()
    return gamma

train_loader,test_loader,labels = Speech_loader(train_batch=20,test_batch=1)
in_shape = 1074
code_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepAE().to(device)

params = list(model.parameters())
optimizer = torch.optim.Adam(params,lr=1e-3)
C = torch.randn(code_size,device = device,requires_grad=True)
update_center = torch.optim.Adagrad([C],lr=1,lr_decay=0.01)
criterion=nn.MSELoss()
Gamma = tune_gamma(DeepAE,in_shape,criterion,train_loader,device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),T=4)

L1, L2, L3, c_vals, data_scores, selected_data_scores_list,losses = DASVDD_trainer(model, in_shape, code_size, C, train_loader, optimizer, update_center, criterion, Gamma, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=3, K=1)

selected_data_scores_array = np.array(selected_data_scores_list)
np.savetxt('DTI\\2-svdd-deep\\df_IC_ADEFG\\df_IC_ADEFG_1_5_n.csv', selected_data_scores_array, delimiter=',')

DASVDD_test(model,C,in_shape,Gamma,test_loader,labels,criterion,C)
