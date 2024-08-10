import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MAX_model(nn.Module):
    def __init__(self, weights = None):
        super(MAX_model, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_classes = 2
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)        
        if(weights!=None):
            self.resnet50.load_state_dict(torch.load(weights))
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.convolutional_layer = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        self.fc1 = nn.Linear(2052, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)


    def forward_once(self, view_data):
        x, boxes, _ = view_data
        x = self.convolutional_layer(x).squeeze(-1).squeeze(-1)
        x = torch.cat((x, boxes[:,:4]), axis=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.normalize(x, p=2, dim=1)


    def forward(self, view_0, view_1):
        # import pdb; pdb.set_trace()
        num_view0_props = view_0[0].shape[0]
        
        
        embedd_0 = self.forward_once(view_0)
        embedd_1 = self.forward_once(view_1)
        
        assert embedd_0.size() == embedd_1.size()

        context = torch.matmul(embedd_0, embedd_1.transpose(-1, -2))
        preds, _ = torch.max(context * view_1[1][:,4].unsqueeze(1), axis=1) 
        preds = preds + view_0[1][:,4]

        return preds
    
    