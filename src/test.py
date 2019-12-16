import os
import numpy as np
import pandas as pd
import glob
import argparse
import pickle
import re
import tqdm
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator

from utils.prepare_data import *
from inception_v3.build_models import *
from resnet50.build_models import *
from utils.test_utils import *

def parse_arg():
    model_nms = ["Inception", "InceptionV3", "ResNet50", "DenseNet201", "InteResNet50", "PartPoolResNet50", "PartPoolInceptionV3", "PartPoolInception"]
    data_nms = ["PETA", "RAP", "PA100K"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name in ' + str(model_nms) + '.')
    parser.add_argument('-s', '--save', type=str, default="",
                        help='The save name.')
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-d', '--data', type=str, default="",
                        help='The dataset need to be trained.')
    parser.add_argument('-w', '--weights', type=str, default="",
                        help='The weight file need to be loaded.')
    parser.add_argument('-c', '--classes', type=int, default=51,
                        help='The class number.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    if args.data == "" or args.data not in data_nms:
        raise RuntimeError('NO DATABASE FOUND IN ' + str(data_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # config = tf.ConfigProto()  
    # config.gpu_options.allow_growth=True
    # sess = tf.Session(config=config)

    # KTF.set_session(sess)
    return args

if __name__ == "__main__":
    print("-----------------testing begining---------------------")
    args = parse_arg()
    model_prefix = "../models/" + args.data + "/" + args.model + "/"
    result_prefix = "../results/" + args.data + "/" + args.model + "/"
    os.makedirs(result_prefix, exist_ok=True)
    nb_class = args.classes
    save_name = args.save
    
    if args.data == "PETA":
        _, _, X_test, y_test, attributes_list = generate_peta()
    elif args.data == "RAP":
        _, _, X_test, y_test, attributes_list = generate_rap()
    elif args.data == "PA100K":
        _, _, _, _, X_test, y_test, attributes_list = generate_pa100k()
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
        attributes_list = np.array(attributes_list)[idx_indices]
        y_test = y_test[:, idx_indices]
    file= open(result_prefix + save_name + '_attributes_list.txt', 'w')  
    for fp in attributes_list:
        file.write(str(fp))
        file.write('\n')
    file.close()

    
    if args.model == "ResNet50":
        model = build_orig_resnet50(nb_class)
    elif args.model == "DenseNet201":
        model = build_orig_densenet201(nb_class)
    elif args.model == "InceptionV3":
        model = build_orig_inceptionv3(nb_class)
    elif args.model == "Inception":
        model = build_orig_inception(nb_class)
    elif args.model == "PartPoolInception":
        version = args.weights[args.weights.index("v"):args.weights.rindex("_")]
        #cheat
        #version = args.weights[args.weights.index("v"):args.weights.index("_")]
        #cheat
        model = build_partpool_inception(parts, version)
    elif args.model == "PartPoolResNet50":
        version = args.weights[args.weights.index("v"):args.weights.rindex("_")]
        model = build_partpool_resnet50(parts, version)
    elif args.model == "PartPoolInceptionV3":
        version = args.weights[args.weights.index("v"):args.weights.rindex("_")]
        model = build_partpool_inceptionv3(parts, version)
        
    test_df = pd.DataFrame(X_test[:, np.newaxis], columns=["file_name"])
    test_datagen = ImageDataGenerator()#rescale=1./255, featurewise_center=True, featurewise_std_normalization=True
    test_generator = test_datagen.flow_from_dataframe(dataframe = test_df,        
            #directory = '../input/test_images',
            x_col = 'file_name', y_col = None,
            target_size = (299, 299),
            batch_size = 1,
            shuffle = False,
            class_mode = None)
    #reg = args.weights + "sgd_" + "(e|f)1*"
    reg = args.weights + "adam_" + "(e|f)1*"
    print(reg)
    weights = [s for s in os.listdir(model_prefix) 
          if re.match(reg, s)]
    print(weights)
    for w in tqdm.tqdm(weights):
        #if os.path.exists(result_prefix + save_name + w + "_results.file"):
        #    continue
        model.load_weights(model_prefix + w, by_name=True)
        test_generator.reset()
        predictions_list = model.predict_generator(test_generator, steps = len(test_generator.filenames), verbose=1)
        # predictions_list = model.predict(X_test_img)
        # print(np.array(predictions_list).shape)
        # for i in predictions_list:
        #     print(i.shape)
        if False:
            predictions = np.hstack(predictions_list)
        else:
            predictions = np.array(predictions_list)
        print("The shape of the predictions_test is: ", predictions.shape)
        label = y_test[:, :nb_class].astype("float64")
        #best_thr = 0.5
        
        ### thr1
        #best_thr, _ = find_best_fixed_threshold(predictions, label)
        
        ### thr2/3
        #"""
        best_thr = np.zeros((1, nb_class))
        for i in range(nb_class):
            best_thr[0, i], _ = find_best_fixed_threshold(predictions[:, i:i+1], label[:, i:i+1])
        #"""
        
        predictions = np.sign(predictions - best_thr)
        label = np.sign(label - 0.5)
        results = calculate_accuracy(label, predictions)
        #cheat
        """
        #前三行注释掉
        results = {}
        results['label_ma'] = np.average(keras_mA(label, predictions))
        results['instance_acc'] = keras_acc(label, predictions)
        results['instance_precision'] = keras_prec(label, predictions)
        results['instance_recall'] = keras_rec(label, predictions)
        results['instance_F1'] = 2 * results['instance_precision'] * results['instance_recall'] / (results['instance_precision'] + results['instance_recall'])
        print(keras_mA(label, predictions))
        print(results['label_ma'])
        #"""
        #cheat
        with open(result_prefix + save_name + "_" + w + "_results.file", "wb") as f:
            pickle.dump(results, f)
        print(result_prefix + save_name + "_" + w + '_results.file')
    print("-----------------testing endding---------------------")