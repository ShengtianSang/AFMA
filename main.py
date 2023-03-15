import csv
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from models.unet.unet_model import My_unet
from models.manet.manet_model import My_MAnet
from models.deeplabv3.deeplabv3_model import My_DeepLabV3
from models.unetplusplus.unetplusplus_model import My_UnetPlusPlus
from models.fpn.fpn_model import My_FPN
from models.linknet.linknet_model import My_Linknet
from models.pan.pan_model import My_PAN
from models.pspnet.pspnet_model import My_pspnet

from models.base import get_preprocessing_fn
from data.CamVid import *
import utils
import segmentation_models_pytorch as smp

MODELS = {"0": "pspnet",
               "1": "deeplabv3",
               "2": "unet",
               "3": "unet++",
               "4": "manet",
               "5": "fpn",
               "6": "pan",
               "7": "linknet"
               }

ENCODER_WEIGHTS = 'imagenet'

if __name__ == '__main__':
    CLASSES = classes_camvid
    data_dir = "E:/data/camvid/"

    model_type_num = input(
        f"Select the models type:\n{'pspnet: 0':<15}{'deeplab: 1':<15}{'unet: 2':<15}{'unet++: 3':<15}\t{'manet: 4':<15}{'fpn: 5':<15}{'pan: 6':<15}\t{'linknet: 7':<15}\n")

    model_type = MODELS[str(model_type_num)]
    print("You have selected the %s models." %model_type)

    encoder_type = input("The resnet based encoder type is (18, 34, 50, 101, 152):")
    ENCODER = 'resnet' + encoder_type

    att_depth = input("The attention depth is (-1, 1, 2, 3, 4, 5):")
    att_depth = int(att_depth)

    the_cuda_number = input("The cuda number you choose is (0-3):")
    DEVICE = 'cuda:' + str(the_cuda_number)

    ACTIVATION = 'logsoftmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation

    best_model_location = "./checkpoints/"
    print("The best models's location is %s." %best_model_location)

    if not os.path.exists(best_model_location):
        os.makedirs(best_model_location)

    best_model_location_and_name_val = best_model_location + "My_" + model_type + "_" + ENCODER + "_" + str(
        att_depth) + "_val.pth"
    best_model_location_and_name_val_checkpoint = best_model_location + "My_" + model_type + "_" + ENCODER + "_" + str(
        att_depth) + "_val_checkpoint.pth"

    preprocessing_fn = get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    CamVidtrans = CamVidTransform()
    train_dataset = CamVidDataset(
        data_dir + "train/",
        data_dir + "trainannot/",
        augmentation=CamVidtrans.get_training_augmentation(),
        preprocessing=CamVidtrans.get_preprocessing(preprocessing_fn)
    )

    valid_dataset = CamVidDataset(
        data_dir + "val/",
        data_dir + "valannot/",
        augmentation=CamVidtrans.get_validation_augmentation(),
        preprocessing=CamVidtrans.get_preprocessing(preprocessing_fn)
    )

    # create test dataset
    test_dataset = CamVidDataset(
        data_dir + "test/",
        data_dir + "testannot/",
        augmentation=CamVidtrans.get_validation_augmentation(),
        preprocessing=CamVidtrans.get_preprocessing(preprocessing_fn)
    )

    batch_size=4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=1)

    ##
    # 2. Training
    ##
    if model_type == "deeplabv3" or model_type == "pspnet":
        if att_depth<4:
            loss = utils.losses.MyLoss_correction(weight=class_weight_camvid, att_depth=att_depth,
                                                          out_channels=len(CLASSES), patch_size=10)
        else:
            loss = utils.losses.MyLoss_correction(weight=class_weight_camvid, att_depth=3, out_channels=len(CLASSES), patch_size=10)

    else:
        loss = utils.losses.MyLoss_correction(weight=class_weight_camvid, att_depth=att_depth,
                                                      out_channels=len(CLASSES), patch_size=10)

    metrics = ["mean_iou_score", "global_iou_score", "accuracy", "my_loss"]

    # create segmentation models with pretrained encoder
    if model_type == "unet":
        model = My_unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "manet":
        model = My_MAnet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "deeplabv3":
        model = My_DeepLabV3(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "unet++":
        model = My_UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "fpn":
        model = My_FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "linknet":
        model = My_Linknet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "pan":
        model = My_PAN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "pspnet":
        model = My_pspnet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )

    model.to(DEVICE)
    lr = 0.0001
    optimizer =torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0e-5)
    scheduler = MultiStepLR(optimizer, milestones=[400, 500, 700, 800, 900], gamma=0.5)

    best_iou_validation=0
    best_iou_test=0

    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
        baseline_method=False
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
        baseline_method=False
    )

    test_epoch = utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        baseline_method=False
    )

    for epoch in range(1000):
        print("Epoch %d ..." % epoch )
        # train models
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if best_iou_validation < valid_logs['mean_iou_score']:
            best_iou_validation = valid_logs['mean_iou_score']

            torch.save(model, best_model_location_and_name_val_checkpoint)
            torch.save(model.state_dict(), best_model_location_and_name_val)

        print("%-10s\t%-10s\t%-10s\t%-10s\t%-10s" % ("Dataset", "mIoU score", "IoU score", "accuracy", "loss"))
        print("%-10s\t %-10s\t %-10s\t %-10s\t %-10s\t" % ("train",round(train_logs["mean_iou_score"],3), round(train_logs["global_iou_score"],3), round(train_logs["accuracy"],3), round(train_logs["my_loss"],3)))
        print("%-10s\t %-10s\t %-10s\t %-10s\t %-10s\t" % ("val",round(valid_logs["mean_iou_score"],3), round(valid_logs["global_iou_score"],3), round(valid_logs["accuracy"],3), round(valid_logs["my_loss"],3)))

    if model_type == "unet":
        best_model = My_unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "manet":
        best_model = My_MAnet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "deeplabv3":
        best_model = My_DeepLabV3(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "unet++":
        best_model = My_UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "fpn":
        best_model = My_FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "linknet":
        best_model = My_Linknet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "pan":
        best_model = My_PAN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )

    best_model.to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_location_and_name_val))

    test_epoch = utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        baseline_method=False
    )
    log_from_val_model = test_epoch.run(test_dataloader)

    print("-------------------------Performance-------------------------")
    print("%-10s\t%-10s\t%-10s\t%-10s\t%-10s" % ("Dataset", "mIoU", "global IoU", "accuracy", "loss"))
    print("%-10s\t %-10s\t %-10s\t %-10s\t %-10s\t" % (
    "val->test", round(log_from_val_model["mean_iou_score"], 3), round(log_from_val_model["global_iou_score"], 3),
    round(log_from_val_model["accuracy"], 3), round(log_from_val_model["my_loss"], 3)))
