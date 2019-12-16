import os
import numpy as np
import sys
sys.path.append("../")
import keras
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, concatenate, concatenate, Activation, Lambda, Dropout, BatchNormalization, Embedding, Reshape
#from utils.mask_operation import tf_mask

from utils.mask_layer import Masking
from utils.MDA import MDA
from utils.FC import FC
from utils.channel_pool import max_out, ave_out, sum_out


class GoogLeNet:
    @staticmethod
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1,1), name=None, trainable=True):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name
        else:
            bn_name = None
            conv_name = None
     
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name, trainable=trainable)(x)
        x = BatchNormalization(axis=3, name=bn_name, trainable=trainable)(x)
        return x
 
    @staticmethod
    def Inception(x, nb_filter, name=None, trainable=True):
        """
        branch1x1 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
     
        branch3x3 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        branch3x3 = GoogLeNet.Conv2d_BN(branch3x3, nb_filter,(3,3), padding='same', strides=(1,1), name=name)
     
        branch5x5 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1),name=name)
        branch5x5 = GoogLeNet.Conv2d_BN(branch5x5, nb_filter, (5,5), padding='same', strides=(1,1), name=name)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = GoogLeNet.Conv2d_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        """
        branch1x1 = GoogLeNet.Conv2d_BN(x, nb_filter[0], (1,1), padding='same', strides=(1,1), name=name+'_1x1', trainable=trainable)
     
        branch3x3 = GoogLeNet.Conv2d_BN(x, nb_filter[1], (1,1), padding='same', strides=(1,1), name=name+'_3x3_reduce', trainable=trainable)
        branch3x3 = GoogLeNet.Conv2d_BN(branch3x3, nb_filter[2],(3,3), padding='same', strides=(1,1), name=name+'_3x3', trainable=trainable)
     
        branch5x5 = GoogLeNet.Conv2d_BN(x, nb_filter[3], (1,1), padding='same', strides=(1,1),name=name+'5x5_reduce', trainable=trainable)
        branch5x5 = GoogLeNet.Conv2d_BN(branch5x5, nb_filter[4], (5,5), padding='same', strides=(1,1), name=name+'_5x5', trainable=trainable)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', trainable=trainable)(x)
        branchpool = GoogLeNet.Conv2d_BN(branchpool, nb_filter[5], (1,1), padding='same', strides=(1,1), name=name+'_pool_proj', trainable=trainable)
     
        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
     
        return x
    

    @staticmethod
    def orig_build(width, height, depth, classes, weights="imagenet"):
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = GoogLeNet.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same', name="conv1_7x7_s2", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        x = GoogLeNet.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same', name="conv2_3x3", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 64, name="inception_3a")#256
        x = GoogLeNet.Inception(x, 120, name="inception_3b")#480
        """
        x = GoogLeNet.Inception(x, [64,96,128,16,32,32], name="inception_3a", trainable=True)#256
        x = GoogLeNet.Inception(x, [128,128,192,32,96,64], name="inception_3b", trainable=True)#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 128, name="inception_4a")#512
        x = GoogLeNet.Inception(x, 128, name="inception_4b")
        x = GoogLeNet.Inception(x, 128, name="inception_4c")
        x = GoogLeNet.Inception(x, 132, name="inception_4d")#528
        x = GoogLeNet.Inception(x, 208, name="inception_4e")#832
        """
        x = GoogLeNet.Inception(x, [192,96,208,16,48,64], name="inception_4a", trainable=True)#512
        x = GoogLeNet.Inception(x, [160,112,224,24,64,64], name="inception_4b", trainable=True)
        x = GoogLeNet.Inception(x, [128,128,256,24,64,64], name="inception_4c", trainable=True)
        x = GoogLeNet.Inception(x, [112,144,288,32,64,64], name="inception_4d", trainable=True)#528
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_4e", trainable=True)#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 208, name="inception_5a")
        x = GoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_5a", trainable=True)
        x = GoogLeNet.Inception(x, [384,192,384,48,128,128], name="inception_5b", trainable=True)#1024
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(classes, activation='sigmoid')(x)
        # create the model
        model = Model(inpt, x, name='Inception')
        # return the constructed network architecture
        if weights == "imagenet":
            print("ImageNet...")
            weights = np.load("../results/googlenet_weights.npy", encoding='latin1').item()
            for layer in model.layers:
                if layer.get_weights() == []:
                    continue
                #weight = layer.get_weights()
                if layer.name in weights:
                    #print(layer.name, end=':')
                    #print(layer.get_weights()[0].shape == weights[layer.name]['weights'].shape, end=' ')
                    #print(layer.get_weights()[1].shape == weights[layer.name]['biases'].shape)
                    layer.set_weights([weights[layer.name]['weights'], weights[layer.name]['biases']])

        return model
    
    @staticmethod
    def partpool_build(width, height, depth, classes, version="v1", tri_loss=False, center_loss=False, weights="imagenet"):
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = GoogLeNet.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same', name="conv1_7x7_s2", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        x = GoogLeNet.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same', name="conv2_3x3", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 64, name="inception_3a")#256
        x = GoogLeNet.Inception(x, 120, name="inception_3b")#480
        """
        x = GoogLeNet.Inception(x, [64,96,128,16,32,32], name="inception_3a", trainable=True)#256
        x = GoogLeNet.Inception(x, [128,128,192,32,96,64], name="inception_3b", trainable=True)#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        if version == "v6":
            #"""
            attention_low = Conv2D(8, (1, 1), activation="relu", name="conv_attention_low")(x)
            _, h_dim, w_dim, feature_channel = K.int_shape(x)
            attention_channel = K.int_shape(attention_low)[-1]
            refined_x_low = MDA(attention_channel, feature_channel)([attention_low, x])
            refined_x_low = GlobalAveragePooling2D()(refined_x_low)
            #"""
        """
        x = GoogLeNet.Inception(x, 128, name="inception_4a")#512
        x = GoogLeNet.Inception(x, 128, name="inception_4b")
        x = GoogLeNet.Inception(x, 128, name="inception_4c")
        x = GoogLeNet.Inception(x, 132, name="inception_4d")#528
        x = GoogLeNet.Inception(x, 208, name="inception_4e")#832
        """
        x = GoogLeNet.Inception(x, [192,96,208,16,48,64], name="inception_4a", trainable=True)#512
        x = GoogLeNet.Inception(x, [160,112,224,24,64,64], name="inception_4b", trainable=True)
        x = GoogLeNet.Inception(x, [128,128,256,24,64,64], name="inception_4c", trainable=True)
        x = GoogLeNet.Inception(x, [112,144,288,32,64,64], name="inception_4d", trainable=True)#528
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_4e", trainable=True)#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        if version == "v6":
            #"""
            attention_mid = Conv2D(8, (1, 1), activation="relu", name="conv_attention_mid")(x)
            _, h_dim, w_dim, feature_channel = K.int_shape(x)
            attention_channel = K.int_shape(attention_mid)[-1]
            refined_x_mid = MDA(attention_channel, feature_channel)([attention_mid, x])
            refined_x_mid = GlobalAveragePooling2D()(refined_x_mid)
            #"""
        """
        x = GoogLeNet.Inception(x, 208, name="inception_5a")
        x = GoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_5a", trainable=True)
        x = GoogLeNet.Inception(x, [384,192,384,48,128,128], name="inception_5b", trainable=True)#1024
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        if version == "v6":
            #"""
            attention_hig = Conv2D(8, (1, 1), activation="relu", name="conv_attention_hig")(x)
            _, h_dim, w_dim, feature_channel = K.int_shape(x)
            attention_channel = K.int_shape(attention_hig)[-1]
            refined_x_hig = MDA(attention_channel, feature_channel)([attention_hig, x])
            refined_x_hig = GlobalAveragePooling2D()(refined_x_hig)
            #"""
        if version == "v1":
            ###########################################Part Max&Ave Pool Softmax###########################################
            #"""
            max_pool = Lambda(lambda x:max_out(x, 1))(x)
            ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            con_pool = concatenate([max_pool, ave_pool], axis=-1)
            w = Conv2D(len(classes), (1, 1))(con_pool)
            w = Lambda(lambda x:keras.activations.softmax(x, axis=-1))(w)
            predictions = []
            for i in range(len(classes)):
                w_ = Lambda(lambda x:K.tile(x[..., i:i+1], [1,1,1,1024]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w_])
                # print(refined_x)
                refined_x = GlobalAveragePooling2D()(refined_x)
                # print(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
                # print(y)
                predictions.append(y)
                #"""
            ###########################################Part Max&Ave Pool Softmax###########################################
        elif version == "v2":
            ###########################################Part Max&Ave Pool Sigmoid###########################################
            #"""
            max_pool = Lambda(lambda x:max_out(x, 1))(x)
            ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            con_pool = concatenate([max_pool, ave_pool], axis=-1)
            predictions = []
            for i in range(len(classes)):
                w = Conv2D(1, (1, 1), activation="sigmoid", name="conv_part_"+str(i+1))(con_pool)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
                # print(refined_x)
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(512, activation='relu')(refined_x)
                # print(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
                # print(y)
                predictions.append(y)
                #"""
            ###########################################Part Max&Ave Pool Sigmoid###########################################
        elif version == "v3":
            ###########################################Part Conv Sigmoid###########################################
            #"""
            predictions = []
            for i in range(len(classes)):
                w = Conv2D(1, (1, 1), activation="sigmoid", name="conv_part_"+str(i+1))(x)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
                # print(refined_x)
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(512, activation='relu')(refined_x)
                # print(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
                # print(y)
                predictions.append(y)
                #"""
            ###########################################Part Conv Sigmoid###########################################
        elif version == "v4":
            ###########################################Part Coarse Split###########################################  
            #"""
            predictions = []
            idx = [0, 2, 5, 8, 10]
            for i in range(len(classes)):
                if i == 0:
                    refined_x = x
                elif i+1 == len(classes):
                    refined_x = Lambda(lambda x:x[:, idx[1]:idx[3], :, :])(x)
                else:
                    refined_x = Lambda(lambda x:x[:, idx[i-1]:idx[i], :, :])(x)
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(1024, activation='relu')(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
                predictions.append(y)
                #"""
            ###########################################Part Coarse Split###########################################
        elif version == "v5":
            ###########################################Part MDA###########################################
            #"""
            attention_hig = Conv2D(8, (1, 1), activation="relu", name="conv_attention_hig")(x)
            #print(K.int_shape(x))
            _, h_dim, w_dim, feature_channel = K.int_shape(x)
            attention_channel = K.int_shape(attention_hig)[-1]
            refined_x_hig = MDA(attention_channel, feature_channel)([attention_hig, x])
            c_dim = K.int_shape(refined_x_hig)[-1]
            predictions = []
            for i in range(len(classes)):
                w = GlobalAveragePooling2D()(refined_x_hig)
                w = Dense(c_dim // 16, activation="relu")(w)
                w = Dense(c_dim, activation="sigmoid")(w)
                w = Lambda(lambda x:K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([refined_x_hig, w])
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dense(1024, activation="relu")(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
                predictions.append(y)
                #"""
            ###########################################Part MDA###########################################
        elif version == "v6":
            ###########################################Part MDA three level###########################################
            #"""
            refined_x = concatenate([refined_x_low, refined_x_mid, refined_x_hig], axis=-1)
            c_dim = K.int_shape(refined_x)[-1]
            predictions = []
            for i in range(len(classes)):
                w = Dense(c_dim // 16, activation="relu")(refined_x)
                w = Dense(c_dim, activation="sigmoid")(w)
                refined_x_ = Lambda(lambda x:x[0] * x[1])([refined_x, w])
                refined_x_ = Dense(1024, activation="relu")(refined_x_)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x_)
                predictions.append(y)
                #"""
            ###########################################Part MDA three level###########################################
        elif version == "v7":
            ###########################################Part MDA three level###########################################
            #"""
            _, h_dim, w_dim, c_dim = K.int_shape(x)
            if center_loss:
                input_target = Input(shape=(len(classes),))
            #max_pool = Lambda(lambda x:max_out(x, 1))(x)
            #ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            #con_pool = concatenate([max_pool, ave_pool], axis=-1)
            #c_dim = K.int_shape(con_pool)[-1]
            predictions = []
            ws = []
            for i in range(len(classes)):
                w = GlobalAveragePooling2D()(x)
                w = Dense(c_dim // 16, activation="relu")(w)
                w = Dense(c_dim, activation="sigmoid")(w)
                #print(w)
                #print(w)
                ws.append(w)
                
                if center_loss:
                    centers = Embedding(len(classes), c_dim)(Lambda(lambda x:K.expand_dims(x[:, i], axis=1))(input_target)) #Embedding层用来存放中心
                    #print(centers)
                    if i == 0:
                        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='l2_loss'+str(i))([w, centers])
                    else:
                        new_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='new_loss'+str(i))([w, centers])
                        l2_loss = Lambda(lambda x:x[0] + x[1], name = 'l2_loss' + str(i))([l2_loss, new_loss])
                        #print(l2_loss)
                        
                w = Lambda(lambda x:K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(1024, activation='relu')(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
                predictions.append(y)
                #"""
            ###########################################Part MDA three level###########################################
        elif version == "v8":
            ###########################################Part Max&Ave Pool Softmax###########################################
            #"""
            max_pool = Lambda(lambda x:max_out(x, 1))(x)
            ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            con_pool = concatenate([max_pool, ave_pool], axis=-1)
            w = Conv2D(len(classes), (1, 1))(con_pool)
            w = Lambda(lambda x:keras.activations.softmax(x, axis=-1))(w)
            preds = []
            for i in range(len(classes)):
                w_ = Lambda(lambda x:K.tile(x[..., i:i+1], [1,1,1,1024]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w_])
                # print(refined_x)
                refined_x = GlobalAveragePooling2D()(refined_x)
                # print(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
                # print(y)
                preds.append(y)
                
            _, h_dim, w_dim, c_dim = K.int_shape(x)
            if center_loss:
                input_target = Input(shape=(len(classes),))
            #max_pool = Lambda(lambda x:max_out(x, 1))(x)
            #ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            #con_pool = concatenate([max_pool, ave_pool], axis=-1)
            #c_dim = K.int_shape(con_pool)[-1]
            predictions = []
            ws = []
            for i in range(len(classes)):
                w = GlobalAveragePooling2D()(x)
                w = Dense(c_dim // 16, activation="relu")(w)
                w = Dense(c_dim, activation="sigmoid")(w)
                #print(w)
                #print(w)
                ws.append(w)
                
                if center_loss:
                    centers = Embedding(len(classes), c_dim)(Lambda(lambda x:K.expand_dims(x[:, i], axis=1))(input_target)) #Embedding层用来存放中心
                    #print(centers)
                    if i == 0:
                        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='l2_loss'+str(i))([w, centers])
                    else:
                        new_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='new_loss'+str(i))([w, centers])
                        l2_loss = Lambda(lambda x:x[0] + x[1], name = 'l2_loss' + str(i))([l2_loss, new_loss])
                        #print(l2_loss)
                        
                w = Lambda(lambda x:K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(1024, activation='relu')(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc2_part_"+str(i+1))(refined_x)
                predictions.append(Lambda(lambda x:(x[0] + x[1]) / 2)([preds[i], y]))
                """
                yy = []
                for j in range(classes[i]):
                    yy.append(Dense(1, activation='linear', name="dense_part_"+str(i+1)+"_"+str(j+1))(concatenate([preds[i, :, j:j+1], y[:, j:j+1]], axis=-1)))
                predictions.append(concatenate(yy, axis=-1))
                #"""
        elif version == "v9":
            ###########################################Part Max&Ave Pool Softmax###########################################
            #"""
            max_pool = Lambda(lambda x:max_out(x, 1))(x)
            ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            con_pool = concatenate([max_pool, ave_pool], axis=-1)
            w = Conv2D(len(classes), (1, 1))(con_pool)
            w = Lambda(lambda x:keras.activations.softmax(x, axis=-1))(w)
            preds = []
            for i in range(len(classes)):
                w_ = Lambda(lambda x:K.tile(x[..., i:i+1], [1,1,1,1024]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w_])
                # print(refined_x)
                refined_x = GlobalAveragePooling2D()(refined_x)
                # print(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
                # print(y)
                preds.append(y)
                
            _, h_dim, w_dim, c_dim = K.int_shape(x)
            if center_loss:
                input_target = Input(shape=(len(classes),))
            #max_pool = Lambda(lambda x:max_out(x, 1))(x)
            #ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            #con_pool = concatenate([max_pool, ave_pool], axis=-1)
            #c_dim = K.int_shape(con_pool)[-1]
            predictions = []
            ws = []
            for i in range(len(classes)):
                w = GlobalAveragePooling2D()(x)
                w = Dense(c_dim // 16, activation="relu")(w)
                w = Dense(c_dim, activation="sigmoid")(w)
                #print(w)
                #print(w)
                ws.append(w)
                
                if center_loss:
                    centers = Embedding(len(classes), c_dim)(Lambda(lambda x:K.expand_dims(x[:, i], axis=1))(input_target)) #Embedding层用来存放中心
                    #print(centers)
                    if i == 0:
                        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='l2_loss'+str(i))([w, centers])
                    else:
                        new_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='new_loss'+str(i))([w, centers])
                        l2_loss = Lambda(lambda x:x[0] + x[1], name = 'l2_loss' + str(i))([l2_loss, new_loss])
                        #print(l2_loss)
                        
                w = Lambda(lambda x:K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(1024, activation='relu')(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc2_part_"+str(i+1))(refined_x)
                #predictions.append(Lambda(lambda x:(x[0] + x[1]) / 2)([preds[i], y]))
                #"""
                xx = concatenate([Lambda(lambda x:K.expand_dims(x, axis=2))(preds[i]), Lambda(lambda x:K.expand_dims(x, axis=2))(y)], axis=-1)
                yy = Dense(1, activation='sigmoid', use_bias=False, name="fc3_part_"+str(i+1))(xx)
                yy = Lambda(lambda x : K.squeeze(x, 2))(yy)
                #yy = Lambda(lambda x : sum_out(x))(yy)
                #print(yy)
                #yy = Reshape(target_shape=(-1, ), input_shape=(classes[i], 1))(xx)
                #yy = FC(str(i+1))(xx)
                print(yy)
                """
                #print(xx)
                for j in range(classes[i]):
                    #print(preds[i])
                    #print(y)
                    #yy.append(Dense(1, activation='linear', use_bias=False, name="fc3_part_"+str(i+1)+"_"+str(j+1))(concatenate([preds[i, :, j:j+1], y[:, j:j+1]], axis=-1)))
                    xxx = Lambda(lambda x:x[:, :, j])(xx)
                    #print(xx)
                    xxx = Dense(1, activation='linear', use_bias=False, name="fc3_part_"+str(i+1)+"_"+str(j+1))(xxx)
                    print(xxx)
                    yy.append(xxx)
         
                predictions.append(concatenate(yy, axis=-1))
                #"""
                predictions.append(yy)
        elif version == "v10":
            ###########################################Part Max&Ave Pool Softmax###########################################
            #"""
            max_pool = Lambda(lambda x:max_out(x, 1))(x)
            ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            con_pool = concatenate([max_pool, ave_pool], axis=-1)
            w = Conv2D(len(classes), (1, 1))(con_pool)
            spatial_w = Lambda(lambda x:keras.activations.softmax(x, axis=-1))(w)
            preds = []
            
                
            _, h_dim, w_dim, c_dim = K.int_shape(x)
            if center_loss:
                input_target = Input(shape=(len(classes),))
            #max_pool = Lambda(lambda x:max_out(x, 1))(x)
            #ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            #con_pool = concatenate([max_pool, ave_pool], axis=-1)
            #c_dim = K.int_shape(con_pool)[-1]
            predictions = []
            ws = []
            for i in range(len(classes)):
                w = GlobalAveragePooling2D()(x)
                w = Dense(c_dim // 16, activation="relu")(w)
                w = Dense(c_dim, activation="sigmoid")(w)
                #print(w)
                #print(w)
                ws.append(w)
                
                if center_loss:
                    centers = Embedding(len(classes), c_dim)(Lambda(lambda x:K.expand_dims(x[:, i], axis=1))(input_target)) #Embedding层用来存放中心
                    #print(centers)
                    if i == 0:
                        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='l2_loss'+str(i))([w, centers])
                    else:
                        new_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='new_loss'+str(i))([w, centers])
                        l2_loss = Lambda(lambda x:x[0] + x[1], name = 'l2_loss' + str(i))([l2_loss, new_loss])
                        #print(l2_loss)
                        
                w = Lambda(lambda x:K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
                w_ = Lambda(lambda x:K.tile(x[..., i:i+1], [1,1,1,1024]))(spatial_w)
                refined_x = Lambda(lambda x:x[0] * x[1])([refined_x, w_])
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(1024, activation='relu')(refined_x)
                y = Dense(classes[i], activation='sigmoid', name="fc2_part_"+str(i+1))(refined_x)
                predictions.append(y)
        
        output = concatenate(predictions, axis=-1)
        # create the model
        if tri_loss:
            outputs = [output]
            for i in ws:
                #print(i)
                outputs.append(i)
            oupt = concatenate(outputs, axis=-1)
            model = Model(inpt, oupt, name='Inception')
        elif center_loss:
            outputs = [output, l2_loss]
            oupt = concatenate(outputs, axis=-1)
            model = Model([inpt, input_target], oupt, name='Inception')
        else:
            model = Model(inpt, output, name='Inception')
        # return the constructed network architecture
        if weights == "imagenet":
            print("ImageNet...")
            weights = np.load("/home/anhaoran/codes/spatial_attribute/results/googlenet_weights.npy", encoding='latin1', allow_pickle=True).item()
            for layer in model.layers:
                if layer.get_weights() == []:
                    continue
                #weight = layer.get_weights()
                if layer.name in weights:
                    #print(layer.name, end=':')
                    #print(layer.get_weights()[0].shape == weights[layer.name]['weights'].shape, end=' ')
                    #print(layer.get_weights()[1].shape == weights[layer.name]['biases'].shape)
                    layer.set_weights([weights[layer.name]['weights'], weights[layer.name]['biases']])

        return model
    
def build_orig_inception(nb_classes, width=299, height=299, depth=3, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    model = GoogLeNet.orig_build(width, height, depth, nb_classes)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_partpool_inception(nb_classes, version="v1", width=299, height=299, depth=3, tri_loss=False, center_loss=False, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    model = GoogLeNet.partpool_build(width, height, depth, nb_classes, version, tri_loss, center_loss)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_orig_inceptionv3(nb_classes, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully connected layer
    x = Dense(1024, activation='relu')(x)
    # add a logistic layer
    predictions = Dense(nb_classes, activation='sigmoid')(x)
    # the model returned
    model = Model(inputs=base_model.input, outputs=predictions, name="InceptionV3")
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_partpool_inceptionv3(nb_classes, version="v1", optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    base_model = InceptionV3(input_shape=(299, 299, 3), weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    
    if version == "v1":
        ###########################################Part Max&Ave Pool Softmax###########################################
        #"""
        max_pool = Lambda(lambda x:max_out(x, 1))(x)
        ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
        con_pool = concatenate([max_pool, ave_pool], axis=-1)
        w = Conv2D(len(nb_classes), (1, 1))(con_pool)
        w = Lambda(lambda x:keras.activations.softmax(x, axis=-1))(w)
        predictions = []
        for i in range(len(nb_classes)):
            w_ = Lambda(lambda x:K.tile(x[..., i:i+1], [1,1,1,2048]))(w)
            refined_x = Lambda(lambda x:x[0] * x[1])([x, w_])
            # print(refined_x)
            refined_x = GlobalAveragePooling2D()(refined_x)
            # print(refined_x)
            y = Dense(nb_classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
            # print(y)
            predictions.append(y)
            #"""
        ###########################################Part Max&Ave Pool Softmax###########################################
    elif version == "v2":
        ###########################################Part Max&Ave Pool Sigmoid###########################################
        #"""
        max_pool = Lambda(lambda x:max_out(x, 1))(x)
        ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
        con_pool = concatenate([max_pool, ave_pool], axis=-1)
        predictions = []
        for i in range(len(nb_classes)):
            w = Conv2D(1, (1, 1), activation="sigmoid", name="conv_part_"+str(i+1))(con_pool)
            refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
            refined_x = GlobalAveragePooling2D()(refined_x)
            refined_x = Dropout(0.4)(refined_x)
            refined_x = Dense(1024, activation='relu')(refined_x)
            y = Dense(nb_classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
            predictions.append(y)
            #"""
        ###########################################Part Max&Ave Pool Sigmoid########################################### 
    elif version == "v3":
        ###########################################Part Conv Sigmoid###########################################  
        #"""
        predictions = []
        for i in range(len(nb_classes)):
            w = Conv2D(1, (1, 1), activation="sigmoid", name="conv_part_"+str(i+1))(x)
            refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
            refined_x = GlobalAveragePooling2D()(refined_x)
            refined_x = Dropout(0.4)(refined_x)
            refined_x = Dense(1024, activation='relu')(refined_x)
            y = Dense(nb_classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
            predictions.append(y)
            #"""
        ###########################################Part Conv Sigmoid###########################################
    elif version == "v4":
        ###########################################Part Coarse Split###########################################  
        #"""
        predictions = []
        idx = [0, 1, 4, 7, 8]
        for i in range(len(nb_classes)):
            if i == 0:
                refined_x = x
            elif i+1 == len(nb_classes):
                refined_x = Lambda(lambda x:x[:, idx[1]:idx[3], :, :])(x)
            else:
                refined_x = Lambda(lambda x:x[:, idx[i-1]:idx[i], :, :])(x)
            refined_x = GlobalAveragePooling2D()(refined_x)
            refined_x = Dropout(0.4)(refined_x)
            refined_x = Dense(1024, activation='relu')(refined_x)
            y = Dense(nb_classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
            predictions.append(y)
            #"""
        ###########################################Part Coarse Split###########################################
    elif version == "v5":
        ###########################################Part MDA###########################################
        #"""
        attention_hig = Conv2D(8, (1, 1), activation="relu", name="conv_attention_hig")(x)
        #print(K.int_shape(x))
        _, h_dim, w_dim, feature_channel = K.int_shape(x)
        attention_channel = K.int_shape(attention_hig)[-1]
        refined_x_hig = MDA(attention_channel, feature_channel)([attention_hig, x])
        c_dim = K.int_shape(refined_x_hig)[-1]
        predictions = []
        for i in range(len(nb_classes)):
            w = GlobalAveragePooling2D()(refined_x_hig)
            w = Dense(c_dim // 16, activation="relu")(w)
            w = Dense(c_dim, activation="sigmoid")(w)
            w = Lambda(lambda x:K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
            refined_x = Lambda(lambda x:x[0] * x[1])([refined_x_hig, w])
            refined_x = GlobalAveragePooling2D()(refined_x)
            refined_x = Dense(1024, activation="relu")(refined_x)
            y = Dense(nb_classes[i], activation='sigmoid', name="fc_part_"+str(i+1))(refined_x)
            predictions.append(y)
            #"""
        ###########################################Part MDA###########################################
    
    output = concatenate(predictions, axis=-1)
    # print(output)
    # the model returned
    model = Model(inputs=base_model.input, outputs=output, name="InceptionV3")
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    build_partpool_inception([1,2,3,4,5,6], "v6")