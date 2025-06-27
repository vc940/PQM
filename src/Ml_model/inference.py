import torch
from torch import nn    
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

class CNN1DModel(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN1DModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Similar to GlobalAveragePooling1D

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.15)
        self.fc3 = nn.Linear(64,num_classes)
        self.fc2 = nn.Linear(128, 64)
    
    def forward(self, x):
        # Input: (batch_size, 100, 9)
        x = x.permute(0, 2, 1)  # Convert to (batch_size, channels=9, time_steps=100)
        
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x)  # shape: (batch_size, 256, 1)
        x = x.view(x.size(0), -1)    # shape: (batch_size, 256)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # Or use raw logits and `CrossEntropyLoss`
device = torch.device("cuda")
model = CNN1DModel().to(device)
model.load_state_dict(torch.load("src/Ml_model/Model/best_model_norm (1).pth",weights_only=True))
data = pd.read_csv("src/Ml_model/Model/Harmonics_with_Sag.csv")
sample = torch.tensor(data.iloc[0]).view(1,100,1).to("cuda").float()
print(torch.softmax(model(sample),dim=1))
plt.figure(figsize=(4,2)) 
plt.plot(data.iloc[0])
plt.title("Plot")
plt.savefig('plots/swell_2025-06-21_142755.png', dpi=300, bbox_inches='tight')
plt.show()

# - Disturbance Type(s): Voltage Swell
# - Measurement Points: Main LT Panel (Incoming Feeder)
# - Sampling Frequency: 10 kHz
# - Nominal Voltage: 230V
# - Events Detected:
#   - Time: 14:27:55
#   - Type: Voltage Swell
#   - Duration: 140 ms
#   - Amplitude: Voltage Rised from 230V to 335V
#   - Recovery: Returned to nominal within 160 ms
