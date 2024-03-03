import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()

        # This encoder model is same as given in the PointNet paper 
        # The earlier model (commented out below) of encoder & decoder was not as  
        # per PointNet paper. The checkpoints for the earlier model is stored in cls_enc_dec_model folder. 

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        # The decoder model as given in the PointNet Paper. 
        self.fc =  nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

        # # Encoder layer.
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(3, 64, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),

        #     nn.Conv1d(64, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),

        #     nn.Conv1d(128, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     # Applying max pool on the last layer
        #     # nn.AdaptiveMaxPool1d(1),
        # )

        # # Decoder layer
        # self.decoder = nn.Sequential(
        #     nn.Linear(1024, 512), 
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),

        #     nn.Linear(512, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),

        #     nn.Linear(128, num_classes)
        # )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # points = points.permute(0, 2, 1)
        # features = self.encoder(points)
        # features = torch.max(features, 2, keepdim=True)[0] 
        # features = features.view(-1, 1024)
        # features = self.decoder(features)
        # return features

        points = points.transpose(1, 2) # Transposing input shape.

        # The encoder layers.
        # First convolutional layer
        features = self.conv1(points)
        features = self.bn1(features)
        features = F.relu(features)

        # Second convolutional layer
        features = self.conv2(features)
        features = self.bn2(features)
        features = F.relu(features)

        # Third convolutional layer
        features = self.conv3(features)
        features = self.bn3(features)
        features = F.relu(features)

        # Fourth convolutional layer
        features = self.conv4(features)
        features = self.bn4(features)
        features = F.relu(features)
        
        # Applying Max pool.
        features = torch.amax(features, dim=-1)  
        
        # Calling the decoder to give classificaiton. 
        features = self.fc(features)
        return features



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        self.point_layer = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_seg_classes, 1),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        N = points.shape[1]
        points = points.transpose(1, 2)

        # Local feature extractor
        local_out = self.conv1(points)
        local_out = self.bn1(local_out)
        local_out = F.relu(local_out)

        local_out = self.conv2(local_out)
        local_out = self.bn2(local_out)
        local_out = F.relu(local_out)

        # Global feature extractor
        global_out = self.conv3(local_out)
        global_out = self.bn3(global_out)
        global_out = F.relu(global_out)

        global_out = self.conv4(global_out)
        global_out = self.bn4(global_out)
        global_out = F.relu(global_out)

        # Global pooling
        global_out = torch.amax(global_out, dim=-1, keepdims=True).repeat(1, 1, N)

        # Concatenativng the local features with global features. 
        out = torch.cat((local_out, global_out), dim=1)
        out = self.point_layer(out).transpose(1, 2) 

        return out



