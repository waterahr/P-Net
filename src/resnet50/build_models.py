import os
import sys
sys.path.append("../")
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, concatenate, Activation, Lambda, Dropout
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121

from utils.mask_layer import Masking
from utils.MDA import MDA
from utils.channel_pool import max_out, ave_out

def build_orig_resnet50(nb_classes, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    base_model = ResNet50(weights='imagenet', include_top=False)
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

def build_orig_densenet201(nb_classes, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    base_model = DenseNet121(weights='imagenet', include_top=False)
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

#"""
def build_inte_resnet50(nb_classes, labels, M=512, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    base_model = ResNet50(input_shape=(299, 299, 3), weights="imagenet", include_top=False)
    # add a interpretable mask layer
    x = base_model.output
    # print(x)### Tensor("activation_49/Relu:0", shape=(?, ?, ?, 2048), dtype=float32)
    # x = tf_mask(x, labels, 1, 0.1)
    x = Masking(labels)(x)
    print(x)### Tensor("Mask:0", shape=(?, ?, ?, 2048), dtype=float32)
    # add a new interetable convolutional layer
    x = Conv2D(M, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    # x = tf_mask(x)
    x = Masking(labels)(x)
    print(x)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    # add a logistic layer
    predictions = Dense(nb_classes, activation='sigmoid')(x)
    # the model returned
    model = Model(inputs=base_model.input, outputs=predictions, name="InceptionV3")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model
    #"""

def build_partpool_resnet50(nb_classes, version="v1", optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    base_model = ResNet50(input_shape=(299, 299, 3), weights='imagenet', include_top=False)
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
            w_ = Lambda(lambda x:keras.backend.tile(x[..., i:i+1], [1,1,1,2048]))(w)
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
        idx = [0, 2, 5, 8, 10]
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
            w = Lambda(lambda x:keras.backend.tile(keras.backend.expand_dims(keras.backend.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
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
    build_partpool_resnet50([1, 2, 3, 4, 5, 6], "v3")