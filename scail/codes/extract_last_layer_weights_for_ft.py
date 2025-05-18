from __future__ import division
from torchvision import models
import torch.utils.data.distributed
import os


models_load_path_prefix = 'D:/FR_codes/data/images_list_files/vgg_faces/S~10/unbalanced/train/model_ft/ift_vgg_faces_s10_5k_b'
S = 10 # number_of_states
P = 100 # number_of_classes_per_state
destination_dir = "D:/FR_codes/scail/weights_for_ft"


if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


for b in range(2, S+1):
    print('*' * 20)
    print('BATCH '+str(b))
    num_classes = b * P
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    model_load_path = models_load_path_prefix + str(b) + '.pt'
    print('Loading model from:' + model_load_path)
    state = torch.load(model_load_path, map_location = lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    model.eval()

    destination_path = os.path.join(destination_dir, 'batch_'+str(b))

    print('Saving stats in: '+ destination_path)
    # parameters
    parameters = [e.cpu() for e in list(model.fc.parameters())]

    torch.save(parameters, destination_path)