from __future__ import division
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.cuda as tc
import torch.utils.data.distributed
import sys, os, warnings, time
import numpy as np
from datetime import timedelta
import AverageMeter as AverageMeter
from MyImageFolder import ImagesListFileFolder
import socket
from Utils import DataUtils
import matplotlib.pyplot as plt


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return torch.nn.functional.kl_div(output, model_prob, reduction='sum')


# ------------- selecting loss --------------
loss_fun = 'softmax'
# loss_fun = 'hing'
# loss_fun = 'kl_div'

# ------------- reading parameters --------------
num_workers = 0
gpu = 0
num_epochs = 100
lr_decay = 0.1
lr = 0.001
momentum = 0.9
weight_decay = 0.0001
patience = 10
normalization_dataset_name = 'vgg_faces'
old_batch_size = 256
new_batch_size = 32
val_batch_size = 2
iter_size = int(old_batch_size / new_batch_size)
use_gpu = torch.cuda.is_available()

# --------------- setting path for different data ------------
algo_name = 'vgg_faces_s10_batch1'

# the path where training image list exist (this part is fix: data/images_list_files/vgg_faces/S~10/batch1/train.lst')
train_file_path = 'D:/FR_codes/data/images_list_files/vgg_faces/S~10/batch1/train.lst'

# the path where test image list exist (this part is fix: data/images_list_files/vgg_faces/S~10/batch1/val.lst')
val_file_path = 'D:/FR_codes/data/images_list_files/vgg_faces/S~10/batch1/val.lst'

# the path where the mean and std of image dataset (VGGFace2)  exist (this part is fix:
# ata/datasets_mean_std.txt')
datasets_mean_std_file_path = 'D:/FR_codes/data/datasets_mean_std.txt'

# path of saving model
models_save_dir = 'D:/FR_codes/scail/model_FR'
print('Loading train images from ' + train_file_path)
print('Loading val images from ' + val_file_path)
print('Dataset name for normalization = ' + normalization_dataset_name)

# ---------------- main part ------------------------
# catching warnings
with warnings.catch_warnings(record=True) as warn_list:
    utils = DataUtils()
    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

    print('dataset mean = ' + str(dataset_mean))
    print('dataset std = ' + str(dataset_std))

    # Data loading code
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    train_dataset = ImagesListFileFolder(
        train_file_path,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # --------- by uncommenting each of below line different data augmentation is possible-------
            # transforms.RandomRotation(degrees=(-45, 45)),
            # transforms.RandomAdjustSharpness(2, p=0.5),
            # transforms.GaussianBlur((5, 9), sigma=(0.1, 2.0)),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.2, 0.5)),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = ImagesListFileFolder(
        val_file_path, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # print different parameters
    num_classes = len(train_dataset.classes)
    print("Number of workers = " + str(num_workers))
    print("Old Batch size = " + str(old_batch_size))
    print("New Batch size = " + str(new_batch_size))
    print("Val Batch size = " + str(val_batch_size))
    print("Iter size = " + str(iter_size))
    print("Number of epochs = " + str(num_epochs))
    print("lr = " + str(lr))
    print("momentum = " + str(momentum))
    print("weight_decay = " + str(weight_decay))
    print("lr_decay = " + str(lr_decay))
    print("patience = " + str(patience))
    print("-" * 20)
    print("Number of classes = " + str(num_classes))
    print("Training-set size = " + str(len(train_dataset)))
    print("Validation-set size = " + str(len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=new_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    print("Number of batches in Training-set = " + str(len(train_loader)))
    print("Number of batches in Validation-set = " + str(len(val_loader)))

    # Creating model
    print('Creating model...')
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    # model = models.resnet34(pretrained=False, num_classes=num_classes)

    # Define Loss and Optimizer
    if loss_fun == 'softmax':
        criterion = nn.CrossEntropyLoss()
        Softmaxlayer = False
    elif loss_fun == 'hing':
        criterion = nn.MultiMarginLoss()
        Softmaxlayer = True
    else:
        criterion = LabelSmoothingLoss(0.15, num_classes)
        Softmaxlayer = True

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

    if tc.is_available():
        print("Running on " + str(socket.gethostname()) + " | gpu " + str(gpu))
        model = model.cuda(gpu)
    else:
        print("GPU not available")
        sys.exit(-1)

    # Training
    print("-" * 20)
    print("Training...")
    starting_time = time.time()
    val_acc_history = []
    train_loss_history = []
    features_extractor = nn.Sequential(*list(model.children())[:-1])

    try:
        for epoch in range(num_epochs):
            acc_val = AverageMeter.AverageMeter()

            # scheduler.step()
            model.train()
            running_loss = 0.0
            nb_batches = 0
            # zero the parameter gradients
            optimizer.zero_grad()
            for i, data in enumerate(train_loader, 0):
                nb_batches += 1
                # get the data
                inputs, labels = data
                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

                # wrap it in Variable
                inputs, labels = Variable(inputs), Variable(labels)
                # forward + backward + optimize
                if Softmaxlayer:
                    outputs = model(inputs)
                    probab = torch.nn.functional.log_softmax(outputs)
                    labels = labels.to('cpu')
                    probab = probab.to('cpu')
                    loss = criterion(probab, labels)
                    loss = loss.to('cuda')
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                loss.data /= iter_size
                loss.backward()
                running_loss += loss.data.item()
                if (i + 1) % iter_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            scheduler.step(loss.cpu().data.numpy())

            # Model evaluation
            model.eval()
            for data in val_loader:
                inputs, labels = data
                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                outputs = model(Variable(inputs))
                prec, prec2 = utils.accuracy(outputs.data, labels, topk=(1, 5))
                acc_val.update(prec.item(), inputs.size(0))

            current_elapsed_time = time.time() - starting_time
            print('{:03}/{:03} | {} | Train : loss = {:.4f} | Val : acc = {}%'.
                  format(epoch + 1, num_epochs, timedelta(seconds=round(current_elapsed_time)),
                         running_loss / nb_batches, acc_val.avg))
            val_acc_history.append(acc_val.avg)
            train_loss_history.append(running_loss / nb_batches)

            # Saving model
    except KeyboardInterrupt:
        print('Keyboard Interruption')

    finally:
        print('Finished Training, elapsed training time : {}'.format(
            timedelta(seconds=round(time.time() - starting_time))))
        models_save_dir = os.path.join(models_save_dir, algo_name)
        if not os.path.exists(models_save_dir):
            os.makedirs(models_save_dir)

        print('Saving model in ' + models_save_dir + '.pt' + '...')
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, models_save_dir + '.pt')

# printing trining loss
print("printing trining loss")
print(*train_loss_history, sep="\n")

# printing Vall
print("Validation Accuracy")
print(*val_acc_history, sep="\n")

plt.title("Resnet")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1, num_epochs + 1), val_acc_history)
plt.ylim((0, 100))
plt.xticks(np.arange(1, num_epochs + 1, 1.0))
plt.show()
