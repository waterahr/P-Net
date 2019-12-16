import os
import numpy as np
import glob
import argparse
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras.optimizers import SGD

from utils.prepare_data import *
from inception_v3.build_models import *
from resnet50.build_models import *
from utils.train_utils import *

def parse_arg():
    model_nms = ["Inception", "InceptionV3", "ResNet50", "DenseNet201", "InteResNet50", "PartPoolResNet50", "PartPoolInceptionV3", "PartPoolInception"]
    data_nms = ["PETA", "RAP", "PA100K"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name in ' + str(model_nms) + '.')
    parser.add_argument('-s', '--save', type=str, default="",
                        help='The model savename.')
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-d', '--data', type=str, default="",
                        help='The dataset need to be trained.')
    parser.add_argument('-w', '--weight', type=str, default="",
                        help='The initial weight.')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The epochs need to be trained')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='The batch size in the training progress.')
    parser.add_argument('-c', '--classes', type=int, default=51,
                        help='The class number.')
    parser.add_argument('-i', '--iteration', type=int, default=0,
                        help='The iteration number.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    if args.data == "" or args.data not in data_nms:
        raise RuntimeError('NO DATABASE FOUND IN ' + str(data_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

if __name__ == "__main__":
    print("-----------------training begining---------------------")
    args = parse_arg()
    model_prefix = "../models/" + args.data + "/" + args.model + "/"
    os.makedirs(model_prefix, exist_ok=True)
    if args.save != "":
        model_prefix = model_prefix + args.save + "_"
    nb_epoch = args.epochs
    nb_class = args.classes
    batch_size = args.batch
    monitor = 'val_mA'
    
    if args.data == "PETA":
        X_train, y_train, X_test, y_test, _ = generate_peta()
    elif args.data == "RAP":
        X_train, y_train, X_test, y_test, _ = generate_rap()
    elif args.data == "PA100K":
        X_train, y_train, X_test, y_test, _, _, _ = generate_pa100k()
      
    if args.data == "PETA":
        if args.save.startswith("coarse"):
            whole = [1,2,3,4,5,16,34]
            up = [0,8,20,21,25,28,36,37,44,54,15,19,23,27,30,39,40,46,50,51,55,56,58,59,60,6,7,11,12,13,17,33,35,38,41,52]
            lb = [10,14,18,22,24,29,31,32,45,47,53,57,9,26,42,43,48,49]
            parts = [len(whole), len(up), len(lb)]
            idx_indices = list(np.hstack((whole, up, lb))[:nb_class])
        else:
            whole = [1,2,3,4,5,16,34]
            hs = [0,8,20,21,25,28,36,37,44,54]
            ub = [15,19,23,27,30,39,40,46,50,51,55,56,58,59,60]
            lb = [10,14,18,22,24,29,31,32,45,47,53,57]
            sh = [9,26,42,43,48,49]
            at = [6,7,11,12,13,17,33,35,38,41,52]
            parts = [len(whole), len(hs), len(ub), len(lb), len(sh), len(at)]
            idx_indices = list(np.hstack((whole, hs, ub, lb, sh, at))[:nb_class])
    elif args.data == "RAP":
        if args.save.startswith("coarse"):
            whole = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50]
            up = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,35,36,37,38,39,40,41,42]
            lb = [24,25,26,27,28,29,30,31,32,33,34]
            parts = [len(whole), len(up), len(lb)]
            idx_indices = list(np.hstack((whole, up, lb))[:nb_class])
        else:
            whole = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50]
            hs = [9,10,11,12,13,14]
            ub = [15,16,17,18,19,20,21,22,23]
            lb = [24,25,26,27,28,29]
            sh = [30,31,32,33,34]
            at = [35,36,37,38,39,40,41,42]
            parts = [len(whole), len(hs), len(ub), len(lb), len(sh), len(at)]
            idx_indices = list(np.hstack((whole, hs, ub, lb, sh, at))[:nb_class])
    elif args.data == "PA100K":
        if args.save.startswith("coarse"):
            whole = [0,1,2,3,4,5,6]
            up = [7,8,13,14,15,16,17,18,21,9,10,11,12]
            lb = [19,20,22,23,24,25]
            parts = [len(whole), len(up), len(lb)]
            idx_indices = list(np.hstack((whole, up, lb))[:nb_class])
        else:
            whole = [0,1,2,3,4,5,6]
            hs = [7,8]
            ub = [13,14,15,16,17,18,21]
            lb = [19,20,22,23,24]
            sh = [25]
            at = [9,10,11,12]
            parts = [len(whole), len(hs), len(ub), len(lb), len(sh), len(at)]
            idx_indices = list(np.hstack((whole, hs, ub, lb, sh, at))[:nb_class])
    if args.model in ["PartPoolResNet50", "PartPoolInception", "PartPoolInceptionV3"]:
        y_train = y_train[:, idx_indices]
        y_test = y_test[:, idx_indices]
    
    #loss_func = ""
    #loss_weights = None
    #metrics=[]
    center_loss = False
    alpha = np.sum(y_train, axis=0)#(len(data_y[0]), )
    alpha = alpha / len(y_train)
    print(alpha)
    image_shape=(299, 299)
    
    opt_sgd = "adam"
    #opt_sgd=SGD(lr=0.0001, momentum=0.9,decay=0.0001,nesterov=True)
    #model_prefix = model_prefix + "sgd_"
    
    #version=args.save, 
    if args.model == "ResNet50":
        model = build_orig_resnet50(nb_class, loss=weighted_binary_crossentropy(alpha[:nb_class]), metrics=["accuracy", mA], optimizer=opt_sgd)
    elif args.model == "DenseNet201":
        model = build_orig_densenet201(nb_class, loss=weighted_binary_crossentropy(alpha[:nb_class]), metrics=["accuracy", mA], optimizer=opt_sgd)
    elif args.model == "InteResNet50":
        model = build_inte_resnet50(nb_class, y_train, loss=weighted_binary_crossentropy(alpha[:nb_class]), metrics=["accuracy", mA], optimizer=opt_sgd)
    elif args.model == "Inception":
        model = build_orig_inception(nb_class, loss=weighted_binary_crossentropy(alpha[:nb_class]), metrics=["accuracy", mA], optimizer=opt_sgd)
    elif args.model == "InceptionV3":
        model = build_orig_inceptionv3(nb_class, loss=weighted_binary_crossentropy(alpha[:nb_class]), metrics=["accuracy", mA], optimizer=opt_sgd)
    elif args.model == "PartPoolInception":
        if args.save.startswith("v"):
            loss_func = weighted_binary_crossentropy(alpha[:nb_class])
            metric_lis = ["accuracy", mA]
            tri_loss = False
            center_loss = False
        elif args.save.startswith("triplet"):
            loss_func = tri_weighted_binary_crossentropy(alpha[:nb_class], nb_class)
            metric_lis = [tri_mA(nb_class)]
            tri_loss = True
            center_loss = False
        elif args.save.startswith("center"):
            loss_func = center_weighted_binary_crossentropy(alpha[:nb_class], nb_class)
            metric_lis = [tri_mA(nb_class)]
            tri_loss = False
            center_loss = True
        model = build_partpool_inception(parts, version=args.save[args.save.index("v"):], 
            width=image_shape[1], height=image_shape[0],
            tri_loss = tri_loss, center_loss = center_loss, loss=loss_func, metrics=metric_lis, optimizer=opt_sgd)
    elif args.model == "PartPoolResNet50":
        model = build_partpool_resnet50(parts, version=args.save[args.save.index("v"):],
            loss=weighted_binary_crossentropy(alpha[:nb_class]), metrics=["accuracy", mA], optimizer=opt_sgd)
    elif args.model == "PartPoolInceptionV3":
        model = build_partpool_inceptionv3(parts, version=args.save[args.save.index("v"):],
            loss=weighted_binary_crossentropy(alpha[:nb_class]), metrics=["accuracy", mA], optimizer=opt_sgd)
    if args.weight != "":
        model.load_weights(args.weight, by_name=True)
        
    train_generator = generate_image_from_nmlist(X_train, y_train[:, :nb_class], batch_size, image_shape)
    val_generator = generate_image_from_nmlist(X_test, y_test[:, :nb_class], batch_size, image_shape)
    if center_loss:
        train_generator = generate_imageandtarget_from_nmlist(X_train, y_train[:, :nb_class], batch_size, image_shape)
        val_generator = generate_imageandtarget_from_nmlist(X_test, y_test[:, :nb_class], batch_size, image_shape)
    checkpointer = ModelCheckpoint(filepath = model_prefix + 'epoch{epoch:03d}_valloss{'+ monitor + ':.6f}.hdf5',
                        monitor = monitor,
                        verbose=1, 
                        save_best_only=True, 
                        save_weights_only=True,
                        mode='max',#'auto', 
                        period=1)
    csvlog = CSVLogger(model_prefix + str(args.epochs) + 'iter' + '_log.csv', append=True)
    def step_decay(epoch):
        initial_lrate = 0.001
        gamma = 0.75
        step_size = 200
        lrate = initial_lrate * math.pow(gamma, math.floor((1+epoch) / step_size))
        return lrate
    lrate = LearningRateScheduler(step_decay)
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train.shape[0] / batch_size),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test.shape[0] / batch_size),
            callbacks = [checkpointer, csvlog], #, lrate
            workers = 32,
            initial_epoch = args.iteration)#
    model.save_weights(model_prefix + 'final' + str(args.epochs) + 'iter_model.h5')
    print("-----------------training endding---------------------")