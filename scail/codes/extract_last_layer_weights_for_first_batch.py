from __future__ import division
from torchvision import models
import torch.utils.data.distributed
import os, sys


model_load_path = "D:/FR_codes/scail/model_s/vgg_faces_s10_batch1.pt"
used_model_num_classes = 100
batch_number = '1'
destination_dir = "D:/FR_codes/scail/weights_for_first_batch"


if not os.path.exists(model_load_path):
    print('No model found in the specified path')
    sys.exit(-1)



model = models.resnet18(pretrained=False, num_classes=used_model_num_classes)

print('Loading saved model from ' + model_load_path)
state = torch.load(model_load_path, map_location = lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])

model.eval()



print('Saving statistics...')
# parameters
parameters = [e.cpu() for e in list(model.fc.parameters())]


if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

torch.save(parameters, os.path.join(destination_dir, 'batch_'+batch_number))