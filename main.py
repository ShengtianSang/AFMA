# import the baseline models with AFMA
from models.unet.unet_model import My_unet
from models.manet.manet_model import My_MAnet
from models.deeplabv3.deeplabv3_model import My_DeepLabV3
from models.unetplusplus.unetplusplus_model import My_UnetPlusPlus
from models.fpn.fpn_model import My_FPN
from models.linknet.linknet_model import My_Linknet
from models.pan.pan_model import My_PAN
from models.pspnet.pspnet_model import My_pspnet
from models.base import get_preprocessing_fn

# import the datasets
from data.CityScapes import *
from data.CamVid import *
from data.LiTS import *
from data.SkinLesion import *
from data.Birds import *

import csv
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

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

DATASETS={
    "0": "camvid",
    "1": "cityscapes",
    "2": "skin lesion",
    "3": "CUB",
    "4": "liver"
}

if __name__ == '__main__':

    data_type = input(
        f"Select the dataset:\n{'camvid: 0':<15}{'cityscape: 1':<15}{'skin lesion: 2':<15}{'CUB: 3':<15}{'liver: 4':<15}\n")
    data_type_selected=DATASETS[data_type]

    if data_type_selected == "camvid":
        CLASSES = classes_camvid
        data_dir = "./dataset/camvid/"
    elif data_type_selected == "cityscapes":
        CLASSES =class_cityscape
        data_dir = "./dataset/cityscapes/"
    elif data_type_selected == "skin lesion":
        CLASSES =classes_skin_lesion
        data_dir = "./dataset/skinlesion/"
    elif data_type_selected == "CUB":
        CLASSES =classes_bird
        data_dir = "./dataset/CUB/"
    elif data_type_selected == "liver":
        CLASSES =classes_LiTS
        data_dir = "./dataset/liver/"

    print("You have selected the %s." %DATASETS[data_type])

    model_type_num = input(
        f"Select the model type:\n{'pspnet: 0':<15}{'deeplab: 1':<15}{'unet: 2':<15}{'unet++: 3':<15}\t{'manet: 4':<15}{'fpn: 5':<15}{'pan: 6':<15}\t{'linknet: 7':<15}\n")

    model_type = MODELS[str(model_type_num)]
    print("You have selected the %s model." %model_type)

    encoder_type = input("The resnet based encoder type is (18, 34, 50, 101, 152):")
    ENCODER = 'resnet' + encoder_type

    att_depth = input("The attention depth is (-1, 1, 2, 3, 4, 5):")
    att_depth = int(att_depth)

    the_cuda_number = input("The cuda number you choose is (0-3):")
    DEVICE = 'cuda:' + str(the_cuda_number)

    ACTIVATION = 'logsoftmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation

    best_model_location = "./checkpoints/"+data_type_selected+"/"
    print("The best model's location is %s." %best_model_location)

    if not os.path.exists(best_model_location):
        os.makedirs(best_model_location)

    best_model_location_and_name_val = best_model_location + "My_" + model_type + "_" + ENCODER + "_" + str(
        att_depth) + "_val.pth"
    best_model_location_and_name_val_checkpoint = best_model_location + "My_" + model_type + "_" + ENCODER + "_" + str(
        att_depth) + "_val_checkpoint.pth"

    results_writer=csv.writer(open(best_model_location + "My_" + model_type + "_" + ENCODER + "_" + str(
        att_depth) + "_results.csv", 'w+', newline=''))
    #results_writer.writerow(['train_loss', 'val_loss', 'test_loss','test_loss_from_val', 'train_iou','val_iou','test_iou','test_iou_from_val','train_acc','val_acc','test_acc','test_acc_from_val','train_soft_iou','val_soft_iou','test_soft_iou','test_soft_iou_from_val'])
    results_writer.writerow(
        ['train_loss', 'val_loss', 'test_loss', 'train_iou', 'val_iou', 'test_iou', 'train_acc', 'val_acc', 'test_acc', 'train_soft_iou', 'val_soft_iou',
         'test_soft_iou'])

    # get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)返回 被偏函数(partial)后的preprocess_input(x,...)函数,其作用是对输入的x进行归一化和(x-mean)/std
    preprocessing_fn = get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    if data_type_selected == "camvid":
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

        test_dataset = CamVidDataset(
            data_dir + "test/",
            data_dir + "testannot/",
            augmentation=CamVidtrans.get_validation_augmentation(),
            preprocessing=CamVidtrans.get_preprocessing(preprocessing_fn)
        )
    elif data_type_selected == "cityscapes":
        Cityscapestrans = CityscapesTransform()
        train_dataset = Cityscapes_Dataset(
            data_dir + "leftImg8bit/train/",
            data_dir + "gtFine/train/",
            augmentation=Cityscapestrans.get_training_augmentation(),
            preprocessing=Cityscapestrans.get_preprocessing(preprocessing_fn)
        )
        valid_dataset = Cityscapes_Dataset(
            data_dir + "leftImg8bit/val/",
            data_dir + "gtFine/val/",
            augmentation=Cityscapestrans.get_validation_augmentation(),
            preprocessing=Cityscapestrans.get_preprocessing(preprocessing_fn)
        )
        test_dataset = Cityscapes_Dataset(
            data_dir + "leftImg8bit/val/",
            data_dir + "gtFine/val/",
            augmentation=Cityscapestrans.get_validation_augmentation(),
            preprocessing=Cityscapestrans.get_preprocessing(preprocessing_fn)
        )

    batch_size=2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=1)

    ##
    # model training
    ##
    if model_type == "deeplabv3" or model_type == "pspnet":
        if att_depth<4:
            if data_type_selected == "camvid":
                loss = smp.utils.losses.MyLoss_correction(weight=class_weight_camvid, att_depth=att_depth,
                                                          out_channels=len(CLASSES), patch_size=10)
            elif data_type_selected == "cityscapes":
                loss = smp.utils.losses.MyLoss_correction(weight=class_weight_cityscape, att_depth=att_depth,
                                                          out_channels=len(CLASSES), patch_size=10)
        else:
            if data_type_selected == "camvid":
                loss = smp.utils.losses.MyLoss_correction(weight=class_weight_camvid, att_depth=3,
                                                          out_channels=len(CLASSES), patch_size=10)
            elif data_type_selected == "cityscapes":
                loss = smp.utils.losses.MyLoss_correction(weight=class_weight_cityscape, att_depth=3,
                                                          out_channels=len(CLASSES), patch_size=10)
    else:
        if data_type_selected == "camvid":
            loss = smp.utils.losses.MyLoss_correction(weight=class_weight_camvid, att_depth=att_depth,
                                                      out_channels=len(CLASSES), patch_size=10)
        elif data_type_selected == "cityscapes":
            loss = smp.utils.losses.MyLoss_correction(weight=class_weight_cityscape, att_depth=att_depth,
                                                      out_channels=len(CLASSES), patch_size=10)

    metrics = ["mean_iou_score", "global_iou_score", "accuracy", "my_loss"]

    # create segmentation model with pretrained encoder
    if model_type == "unet":
        model = My_unet(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "manet":
        model = My_MAnet(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "deeplabv3":
        model = My_DeepLabV3(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "unet++":
        model = My_UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "fpn":
        model = My_FPN(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "linknet":
        model = My_Linknet(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "pan":
        model = My_PAN(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )
    elif model_type == "pspnet":
        model = My_pspnet(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
            att_depth=att_depth
        )

    #model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    lr = 0.0001
    optimizer =torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0e-5)
    scheduler = MultiStepLR(optimizer, milestones=[300,400, 500], gamma=0.5)

    best_iou_validation=0
    best_iou_test=0

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
        baseline_method=False
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
        baseline_method=False
    )

    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        baseline_method=False
    )


    for epoch in range(500):
        print("Epoch %d ..." % epoch )
        # train model
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        test_logs = test_epoch.run(test_dataloader)

        if best_iou_validation < valid_logs['mean_iou_score']:
            best_iou_validation = valid_logs['mean_iou_score']

            # save model checkpoints
            torch.save(model, best_model_location_and_name_val_checkpoint)
            torch.save(model.state_dict(), best_model_location_and_name_val)

    best_model = torch.load(best_model_location_and_name_val_checkpoint)
    best_model.to(DEVICE)
    test_epoch_with_testmodel = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        baseline_method=False
    )
    test_logs_with_testmodel = test_epoch_with_testmodel.run(test_dataloader)

    results_writer.writerow([train_logs['my_loss'],valid_logs['my_loss'],test_logs_with_testmodel['my_loss'],
                             train_logs['mean_iou_score'],valid_logs['mean_iou_score'],test_logs_with_testmodel['mean_iou_score'],
                             train_logs['accuracy'],valid_logs['accuracy'],test_logs_with_testmodel['accuracy'],
                             train_logs['global_iou_score'],valid_logs['global_iou_score'],test_logs_with_testmodel['global_iou_score']
                             ])

    print("%-10s\t%-10s\t%-10s\t%-10s\t%-10s" % ("Dataset", "mIoU score", "IoU score", "accuracy", "loss"))
    print("%-10s\t %-10s\t %-10s\t %-10s\t %-10s\t" % ("train",round(train_logs["mean_iou_score"],3), round(train_logs["global_iou_score"],3), round(train_logs["accuracy"],3), round(train_logs["my_loss"],3)))
    print("%-10s\t %-10s\t %-10s\t %-10s\t %-10s\t" % ("val",round(valid_logs["mean_iou_score"],3), round(valid_logs["global_iou_score"],3), round(valid_logs["accuracy"],3), round(valid_logs["my_loss"],3)))
    print("%-10s\t %-10s\t %-10s\t %-10s\t %-10s\t" % ("test",round(test_logs_with_testmodel["mean_iou_score"],3), round(test_logs_with_testmodel["global_iou_score"],3), round(test_logs_with_testmodel["accuracy"],3), round(test_logs_with_testmodel["my_loss"],3)))


