from torch.utils.data import Dataset
from torchvision import models, transforms
from configure import Config
from utlity import *
from AudiotoImg import PrecomputedAudio
from torch.optim import lr_scheduler

import torch.optim as optim
import torch.nn as nn

if __name__ == '__main__':

    # Configuration file
    cfg = Config()
    # Prepare data - you need to uncomment the following line to enable data preprocessing
    # prepare_data(cfg)

    # create new dataset and normalized images - train
    pre_train = PrecomputedAudio(cfg.train_path,
                                 img_transforms=transforms.Compose(
                                     [transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])]
                                 )
                                 )
    # create new dataset and normalized images -val
    pre_val = PrecomputedAudio(cfg.valid_path,
                               img_transforms=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
                               )
    # Load data
    train_loader = torch.utils.data.DataLoader(pre_train, cfg.bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(pre_val, cfg.bs, shuffle=True)
    train_data_size = pre_train.length
    val_data_size = pre_val.length

    # Train model
    # Using pretrained ResNet-50 model
    spec_resnet = models.resnet50(pretrained=True)
    # Start Transfer learning

    ## Transfer learning method one: Finetuning the convnet =======================================================
    # num_ftrs = spec_resnet.fc.in_features
    # # Here the size of each output sample is set to number of class.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # spec_resnet.fc = nn.Linear(num_ftrs, cfg.num_cls)

    # # send to device - cuda
    # spec_resnet = spec_resnet.to(cfg.device)

    # loss_fn = nn.CrossEntropyLoss()

    # # Observe that all parameters are being optimized
    # optimizer = optim.SGD(spec_resnet.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    # # Decay LR by a factor of 0.1 every 8 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    #
    # trained_model, history = train(trained_model, optimizer, loss_fn, scheduler, train_loader,train_data_size, val_loader, val_data_size, cfg.epochs, cfg.device)
    #
    # # plot train/val result
    # torch.save(history, 'model/SC_history.pt')
    # plt_result(history)

    # Transfer learning method two: ConvNet as fixed feature extractor============================================
    # freeze the parameters
    for param in spec_resnet.parameters():
        param.requires_grad = False

    num_ftrs = spec_resnet.fc.in_features
    # Change head of the model -> Untrained Sequential module
    spec_resnet.fc = nn.Sequential(nn.Linear(spec_resnet.fc.in_features, 500),
                                   nn.ReLU(),
                                   nn.Dropout(0.4),
                                   nn.Linear(500, cfg.num_cls),
                                   nn.LogSoftmax(dim=1))

    spec_resnet = spec_resnet.to(cfg.device)

    loss_fn = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer = optim.SGD(spec_resnet.fc.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    # Decay LR by a factor of 0.1 every 8 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    # Train the last layer parameters for several rounds
    trained_model, history = train(spec_resnet, optimizer, loss_fn, scheduler, train_loader, train_data_size,
                                   val_loader, val_data_size, cfg.pre_epochs, cfg.device)

    # unfrozen the parameters
    for param in trained_model.parameters():
        param.requires_grad = True

    # All the parameters
    optimizer = optim.SGD(trained_model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    # Decay LR by a factor of 0.1 every 8 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    # train all the parameters together
    trained_model, history = train(trained_model, optimizer, loss_fn, scheduler, train_loader, train_data_size,
                                   val_loader, val_data_size, cfg.epochs, cfg.device)

    # plot train/val result
    plt_result(history)