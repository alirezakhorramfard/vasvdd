import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

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
        
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)  
        
        # Encoder layers
        self.enc_layer1 = nn.Linear(self.hidden_size, self.hidden_size_1) 
        self.enc_layer2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.enc_layer3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.enc_layer4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        
        # Decoder layers 
        self.dec_layer1 = nn.Linear(self.hidden_size_4, self.hidden_size_3)
        self.dec_layer2 = nn.Linear(self.hidden_size_3, self.hidden_size_2) 
        self.dec_layer3 = nn.Linear(self.hidden_size_2, self.hidden_size_1)
        self.dec_layer4 = nn.Linear(self.hidden_size_1, self.hidden_size)
        self.dec_layer5 = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, x):
        x = self.input_layer(x)
        # Encoder
        x = self.dropout(F.relu(self.enc_layer1(x)))
        x = self.dropout(F.relu(self.enc_layer2(x)))
        x = self.dropout(F.relu(self.enc_layer3(x)))
        x_1 = self.dropout(F.relu(self.enc_layer4(x)))
        x=x_1
        # Decoder
        x = self.dropout(F.relu(self.dec_layer1(x)))
        x = self.dropout(F.relu(self.dec_layer2(x)))
        x = self.dropout(F.relu(self.dec_layer3(x)))
        x = self.dropout(F.relu(self.dec_layer4(x)))
        x = self.dropout(F.relu(self.dec_layer5(x)))

        
        
        return x,x_1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
model = DeepAE() 
df2 = pd.read_csv("DTI\\3-Data-preparation\\df_EN_ADEFG\\df_EN_ADEFG_1_5.csv")

df2.drop(df2.columns[0], axis=1, inplace=True) 
tensor_data = torch.from_numpy(df2.values).float() 

tensor_data = tensor_data.to(device)
epochs = 20
batch_size = 10
learning_rate = 1e-3

# Define loss and optimizer
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(epochs):
  
  for i in range(0, tensor_data.size(0), batch_size):
  
    # Forward pass
    batch = tensor_data[i:i+batch_size]
    reconstructed,_ = model(batch)

    # Calculate loss
    loss = criterion(reconstructed, batch)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f} ')



test_df = pd.read_csv("DTI\\3-Data-preparation\\df_EN_ADEFG\\df_EN_ADEFG_test_1_5.csv")
test_df.drop(df2.columns[0], axis=1, inplace=True) 
model.eval()
test_tensor = torch.from_numpy(test_df.values).float().to(device)

with torch.no_grad():
  _,test_outputs = model(test_tensor)

last_layer_outputs = test_outputs

output = last_layer_outputs.detach().cpu().numpy()

df = pd.DataFrame(output)
df.to_csv('DTI\\5-preparation-label\\df_EN_ADEFG\\df_EN_ADEFG_1_5.csv', index=False)
