# #!/usr/bin/python
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out, drop=0.0):
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear1_1 = torch.nn.Linear(H, H)
#         self.linear1_2 = torch.nn.Linear(H, H)
#         self.linear1_3 = torch.nn.Linear(H, H)
#         self.linear1_4 = torch.nn.Linear(H, H)
#         self.linear2 = torch.nn.Linear(H, D_out)
#         self.relu = torch.nn.ReLU()
#         self.drop1   = torch.nn.Dropout(p=drop)
# #         self.sigmoid = torch.nn.Sigmoid()
# #         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         h_relu = self.drop1(self.linear1(x))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
#         h_relu = self.drop1(self.linear1_1(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
#         h_relu = self.drop1(self.linear1_2(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
#         h_relu = self.drop1(self.linear1_3(h_relu))#.clamp(min=0)
#         h_relu = self.relu(h_relu)
#         h_relu = self.drop1(self.linear1_4(h_relu))#.clamp(min=0)
#         h_relu = self.relu(h_relu)
#         y_pred = self.softmax(self.linear2(h_relu))
        
#         return y_pred
    
# class ConvLayerNet(torch.nn.Module):
#     def __init__(self, channels_in, channels_out, D_in, H, D_out, drop=0.0, kernel_size=1, stride =1):
#         super(ConvLayerNet, self).__init__()
#         self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size,...
#                                stride=stride)
#         W_out = (D_in - kernel_size)/(stride+1)
        
#         self.linear1_1 = torch.nn.Linear(W_out, H)
#         self.linear2 = torch.nn.Linear(H, D_out)
        
#         self.relu = torch.nn.ReLU()
#         self.drop1   = torch.nn.Dropout(p=drop)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
    
#         h_relu = self.relu(self.conv1(x))
#         h_relu, _ = h_relu.max(dim=1) # remove 3rd dimention
#         h_relu = self.relu(h_relu)
        
#         h_relu = self.linear1_1(h_relu)
        
#         y_pred = self.linear2(h_relu)
        
# #         y_pred = self.sigmoid(h_relu)
        
#         return y_pred
    
    
# class ConvLayerNet2(torch.nn.Module):
#     # input batch_size, newaxis, sequence_length
#     def __init__(self,  channels_in, channels_out, D_in, H, H2, D_out, device, drop=0.0, kernel_size=1, stride=1):
#         super(ConvLayerNet2, self).__init__()
#         self.device = device
#         self.convs = nn.ModuleList([nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size, stride=stride) for i in range(15)])

#         W_out = ((D_in - kernel_size)//stride + 1)*15*channels_out
#         W_out = np.int64(W_out)
        
#         self.linear1_1 = torch.nn.Linear(W_out, H)
#         self.linear1_2 = torch.nn.Linear(H, H2)
#         self.linear2 = torch.nn.Linear(H2, D_out)
        
#         self.relu = torch.nn.ReLU()
#         self.drop1   = torch.nn.Dropout(p=drop)
#         self.sigmoid = torch.nn.Sigmoid()
# #         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         ii = 0
#         h_relu = torch.FloatTensor([]).to(self.device)
#         for layers in self.convs:
#             x_data = layers(x[:,:,ii*(D_in):(ii+1)*D_in])
#             x_data = self.relu(x_data)
#             b, h, w = x_data.size()
#             x_data = x_data.view(b,-1)
#             h_relu = torch.cat([h_relu, x_data], axis = 1)
#             ii = ii+1

        
#         h_relu = self.drop1(self.linear1_1(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
        
#         h_relu = self.drop1(self.linear1_2(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
        
#         y_pred = (self.linear2(h_relu))
        
#         return y_pred
    
# class ConvLayerNet3(torch.nn.Module):
#     def __init__(self,  channels_in, channels_out, D_in, H, D_out, device, drop=0.0, kernel_size=1, stride=1):
#         super(ConvLayerNet3, self).__init__()
#         self.device = device
#         self.convs = nn.ModuleList([nn.Conv1d(1, channels_out, kernel_size=kernel_size, stride=stride) for i in range(15)])

#         W_out = ((D_in - kernel_size)//stride + 1)*15*channels_out
#         W_out = np.int64(W_out)
#         H3 = H//2
#         self.linear1_1 = torch.nn.Linear(W_out, H)
#         self.linear1_2 = torch.nn.Linear(H, H3)
#         self.linear1_3 = torch.nn.Linear(H3, 64)
#         self.linear2 = torch.nn.Linear(64, D_out)
        
#         self.relu = torch.nn.ReLU()
#         self.drop1   = torch.nn.Dropout(p=drop)
#         self.sigmoid = torch.nn.Sigmoid()
# #         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         ii = 0
#         h_relu = torch.FloatTensor([]).to(self.device)
#         for layers in self.convs:
#             x_data = layers(torch.unsqueeze(x[:, ii, :],1))
#             x_data = self.relu(x_data)
#             b, h, w = x_data.size()
#             x_data = x_data.view(b,-1)
#             h_relu = torch.cat([h_relu, x_data], axis = 1)
#             ii = ii+1
        
#         h_relu = self.drop1(self.linear1_1(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
        
#         h_relu = self.drop1(self.linear1_2(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
        
#         h_relu = self.drop1(self.linear1_3(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
        
#         y_pred = (self.linear2(h_relu))
        
#         return y_pred
    
# class ConvLayerNet4(torch.nn.Module):
#     def __init__(self,  channels_in, channels_out, D_in, H, D_out, device, drop=0.0, kernel_size=1, stride=1):
#         super(ConvLayerNet4, self).__init__()
#         self.device = device
#         self.convs = nn.ModuleList([nn.Sequential(
#             nn.Conv1d(1, channels_out, kernel_size=kernel_size, stride=stride),
#             nn.Conv1d(1, channels_out, kernel_size=kernel_size, stride=stride),
#             nn.Conv1d(1, channels_out, kernel_size=kernel_size, stride=stride),
#         ) for i in range(15)])
                                    
#         W_out = ((D_in - kernel_size)//stride + 1)*channels_out
#         W_out = ((W_out - kernel_size)//stride + 1)*channels_out
#         W_out = ((W_out - kernel_size)//stride + 1)*15*channels_out
#         W_out = np.int64(W_out) 
#         H3 = H//2
#         self.linear1_1 = torch.nn.Linear(W_out, H)
#         self.linear1_2 = torch.nn.Linear(H, 16)
#         self.linear2 = torch.nn.Linear(16, D_out)
        
#         self.relu = torch.nn.ReLU()
#         self.drop1   = torch.nn.Dropout(p=drop)
#         self.sigmoid = torch.nn.Sigmoid()
# #         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         ii = 0
#         h_relu = torch.FloatTensor([]).to(self.device)
#         for layers in self.convs:
#             x_data = layers(torch.unsqueeze(x[:, ii, :],1))
#             x_data = self.relu(x_data)
#             b, h, w = x_data.size()
#             x_data = x_data.view(b,-1)
#             h_relu = torch.cat([h_relu, x_data], axis = 1)
#             ii = ii+1
        
#         h_relu = self.drop1(self.linear1_1(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
        
#         h_relu = self.drop1(self.linear1_2(h_relu))#.clamp(min=0)#.clamp(min=0)
#         h_relu = self.relu(h_relu)
        
#         y_pred = self.sigmoid(self.linear2(h_relu))
        
#         return y_pred
# #     drop = 0.1
# #     H1 = [  256, 128]#, 2048,
# #     N1 = [8, 16, 32, 64]
# #     epochs = 100
# #     learning_rate = 1e-4
# #     kernel_size = 10
# #     stride  = 3