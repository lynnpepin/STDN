import pickle
import numpy as np
import json
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM, Conv3D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import ipdb
import attention
from capsule_3D_layers import CapsuleLayer, PrimaryCap, CapsuleLayer2D, PrimaryCap2D

class baselines:
    def __init__(self):
        pass


class models:
    def __init__(self):
        pass

    def dual_capsnet(self,
            input_shape = (24, 10, 20, 2), # 24 = Window Size; must not be too small
            routings    = 3,
            optimizer   = 'adagrad',
            loss        = 'mse',
            metrics     = [],
            verbose     = True):
        
        # There are two x-channel inputs to this NN; same input shape.
        x1 = Input(shape=input_shape)
        x2 = Input(shape=input_shape)
        
        # Track 1
        conv1 = Conv3D(filters     = 128,
                       kernel_size = (5,5,5),
                       strides     = 1,
                       padding     = 'valid',
                       activation  = 'relu', name = 'conv1' )(x1)               
        pcap1 = PrimaryCap(conv1,
                           dim_capsule = 12,
                           n_channels  = 8,
                           kernel_size = (5,3,3),
                           strides = 1,
                           padding = 'valid',
                           name='pcap1')
        dcap1 = CapsuleLayer(num_capsule = 8,
                             dim_capsule = 16,
                             routings    = routings,
                             name = 'dcap1')(pcap1)
        # Track 2
        conv2 = Conv3D(filters     = 128,
                       kernel_size = (5,5,5),
                       strides     = 1,
                       padding     = 'valid',
                       activation  = 'relu', name = 'conv2' )(x2)               
        pcap2 = PrimaryCap(conv2,
                           dim_capsule = 12,
                           n_channels  = 8,
                           kernel_size = (5,3,3),
                           strides = 1,
                           padding = 'valid',
                           name    = 'pcap2')
        dcap2 = CapsuleLayer(num_capsule = 8,
                             dim_capsule = 16,
                             routings    = routings,
                             name = 'dcap2')(pcap2)

        # Concat
        dcap_merged = Concatenate()([dcap1, dcap2])
        # Extra dcap layer
        dcap_merged = CapsuleLayer(num_capsule = 8,
                                     dim_capsule = 16,
                                     routings    = routings,
                                     name = 'dcap_m')(dcap_merged)
        # Fully connected part
        flatten = Flatten()(dcap_merged)
        dense1 = Dense(512, activation='relu', name='dense1')(flatten)
        dense2 = Dense(1024, activation='relu', name='dense2')(dense1)
        dense3 = Dense(512,  activation='relu', name='dense3')(dense2)

        dense4 = Dense(512,  activation='relu', name='dense4')(dense3)
        d_out  = Dense(np.prod(input_shape[1:]), activation='relu', name='dout')(dense4)
        y_out  = Reshape(target_shape = input_shape[1:])(d_out)
        
        model = Model([x1, x2], y_out)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model



    def single_capsnet(self,
            input_shape = (24, 10, 20, 2), # 24 = Window Size; must not be too small
            routings    = 3,
            optimizer   = 'adagrad',
            loss        = 'mse',
            metrics     = [],
            verbose     = True):
        
        # Input layer
        x = Input(shape=input_shape)
        # Layer 1: Just conventional Conv2D layers
        conv1 = Conv3D(filters     = 128,
                       kernel_size = (7,5,5),
                       strides     = 1,
                       padding     = 'valid',
                       activation  = 'relu', name = 'conv1' )(x)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(conv1,
                                 dim_capsule = 8,
                                 n_channels  = 8,
                                 kernel_size = (7,3,3),
                                 strides = 1,
                                 padding = 'valid')

        # Layer 3: Capsule layer. Routing algorithm works here.
        #   num_capsule and dim_capsule are a choice you need to make, by intuition.
        digitcaps1 = CapsuleLayer(num_capsule = 8,
                                  dim_capsule = 16,
                                  routings    = routings,
                                  name = 'dcaps1')(primarycaps)
        
        # Prediction layers:
        # Should have shape input_shape[1:]. e.g. (7, 10, 20, 2) --> (10, 20, 2)
        flatten = Flatten()(digitcaps1)
        dense1 = Dense(512, activation='relu', name='dense1')(flatten)
        dense2 = Dense(1024, activation='relu', name='dense2')(dense1)
        dense3 = Dense(512,  activation='relu', name='dense3')(dense2)
        dense4 = Dense(512,  activation='relu', name='dense4')(dense3)
        d_out  = Dense(np.prod(input_shape[1:]), activation='relu', name='dout')(dense4)
        y_out  = Reshape(target_shape = input_shape[1:])(d_out)
        
        model = Model(x, y_out)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model


    def stdcn(self,
             att_lstm_num, #3
             att_lstm_seq_len, #3
             lstm_seq_len, #7
             feature_vec_len, #160
             cnn_flat_size  = 128,
             lstm_out_size  = 128,
             nbhd_size      = 3,
             nbhd_type      = 2,
             map_x_num      = 10,
             map_y_num      = 20,
             flow_type      = 4,
             output_shape   = 2,
             routings    = 3,
             optimizer      = 'adagrad',
             loss           = 'mse',
             metrics        = [],
             verbose        = True):
        
        """
        Returns a Keras model (STDCN - Spatial-temporal dynamics capsule network).
        **Heavily** derivative of stdn() (below)
        Usage:
            >>> import models
            >>> modeler = models.models()
            >>> my_model = modeler.stdcn(...)
        """
        if verbose:
            print("  Model: Creating STD-Capsule-N with parameters:",
                      att_lstm_num,
                      att_lstm_seq_len,
                      lstm_seq_len,
                      feature_vec_len,
                      cnn_flat_size,
                      lstm_out_size,
                      nbhd_size,
                      nbhd_type,
                      map_x_num,
                      map_y_num,
                      flow_type,
                      output_shape,
                      optimizer,
                      loss,
                      metrics,"\n")
        
        flatten_att_nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "att_nbhd_volume_input_time_{0}_{1}".format(att+1, ts+1))
                                   for ts in range(att_lstm_seq_len)
                                   for att in range(att_lstm_num)]
        flatten_att_flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "att_flow_volume_input_time_{0}_{1}".format(att+1, ts+1))
                                   for ts in range(att_lstm_seq_len)
                                   for att in range(att_lstm_num)]
        
        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])

        att_lstm_inputs = [Input(shape = (att_lstm_seq_len, feature_vec_len,), name = "att_lstm_input_{0}".format(att+1))
                           for att in range(att_lstm_num)]
        nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "nbhd_volume_input_time_{0}".format(ts+1))
                       for ts in range(lstm_seq_len)]
        flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "flow_volume_input_time_{0}".format(ts+1))
                       for ts in range(lstm_seq_len)]
        lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")
        
        #short-term part
        #1st level gate
        #nbhd cnn
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time0_{0}".format(ts+1))(nbhd_inputs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu",name = "nbhd_convs_activation_time0_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        #flow cnn
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time0_{0}".format(ts+1))(flow_inputs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time0_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        #flow gate
        flow_gates = [Activation("sigmoid",

                                 name = "flow_gate0_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]])
                      for ts in range(lstm_seq_len)]
        
        
        #2nd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time1_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time1_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time1_{0}".format(ts+1))(flow_inputs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time1_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate1_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]])
                      for ts in range(lstm_seq_len)]

        #3rd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time2_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time2_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time2_{0}".format(ts+1))(flow_inputs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time2_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate2_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        # Study the methods used here!
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]])
                      for ts in range(lstm_seq_len)]
        # TODO: TWEAK GROUP[1] CAPSULE LAYERS HERE
        # Rename?
        nbhd_convs = [PrimaryCap2D(flow_convs[ts], dim_capsule=8, n_channels=16, kernel_size=3, strides=2, padding='valid', name='pcap_{0}'.format(ts+1))
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [CapsuleLayer2D(num_capsule=8, dim_capsule=12, routings=routings, name='dcaps_{0}'.format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        # TODO: Replace flatten?


        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_vecs = [Dense(units = cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1))(nbhd_vecs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts])
                      for ts in range(lstm_seq_len)]

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (lstm_seq_len, cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])

        #lstm
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        #attention part
        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_inputs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time0_{0}_{1}".format(att+1,ts+1))(att_flow_inputs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate0_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate1_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate2_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        # TODO: TWEAK GROUP[2] CAPSULE LAYERS HERE
        att_nbhd_convs = [[PrimaryCap2D(att_nbhd_convs[att][ts], dim_capsule=8, n_channels=16, kernel_size=3, strides=2, padding='valid', name='pcap_{0}_{1}'.format(att+1, ts+1))
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[CapsuleLayer2D(num_capsule=8, dim_capsule=12, routings=routings, name='dcaps_{0}_{1}'.format(att+1, ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        # TODO: Replace flatten?

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                          for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Dense(units = cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts])
                          for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts])
                          for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]


        att_nbhd_vec    = [Concatenate(axis=-1)(att_nbhd_vecs[att])
                           for att in range(att_lstm_num)]
        att_nbhd_vec    = [Reshape(target_shape = (att_lstm_seq_len, cnn_flat_size))(att_nbhd_vec[att])
                           for att in range(att_lstm_num)]
        att_lstm_input  = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]])
                           for att in range(att_lstm_num)]

        att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att])
                     for att in range(att_lstm_num)]

        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm])
                       for att in range(att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(att_lstm_num, lstm_out_size))(att_low_level)


        att_high_level = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = Dense(units = output_shape)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        inputs = flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs,]
        # print("Model input length: {0}".format(len(inputs)))

        # ipdb.set_trace()
        model = Model(inputs = inputs, outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model

    
    def stdn(self,
             att_lstm_num, #3
             att_lstm_seq_len, #3
             lstm_seq_len, #7
             feature_vec_len, #160
             cnn_flat_size  = 128,
             lstm_out_size  = 128,
             nbhd_size      = 3,
             nbhd_type      = 2,
             map_x_num      = 10,
             map_y_num      = 20,
             flow_type      = 4,
             output_shape   = 2,
             optimizer      = 'adagrad',
             loss           = 'mse',
             metrics        = [],
             verbose        = True):
        """
        Returns a Keras model (STDN).
        Usage:
            >>> import models
            >>> modeler = models.models()
            >>> my_model = modeler.stdn(...)
        """
        if verbose:
            print("  Model: Creating STDN with parameters:",
                      att_lstm_num,
                      att_lstm_seq_len,
                      lstm_seq_len,
                      feature_vec_len,
                      cnn_flat_size,
                      lstm_out_size,
                      nbhd_size,
                      nbhd_type,
                      map_x_num,
                      map_y_num,
                      flow_type,
                      output_shape,
                      optimizer,
                      loss,
                      metrics,"\n")
        
        flatten_att_nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "att_nbhd_volume_input_time_{0}_{1}".format(att+1, ts+1))
                                   for ts in range(att_lstm_seq_len)
                                   for att in range(att_lstm_num)]
        flatten_att_flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "att_flow_volume_input_time_{0}_{1}".format(att+1, ts+1))
                                   for ts in range(att_lstm_seq_len)
                                   for att in range(att_lstm_num)]
        
        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])

        att_lstm_inputs = [Input(shape = (att_lstm_seq_len, feature_vec_len,), name = "att_lstm_input_{0}".format(att+1))
                           for att in range(att_lstm_num)]
        nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "nbhd_volume_input_time_{0}".format(ts+1))
                       for ts in range(lstm_seq_len)]
        flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "flow_volume_input_time_{0}".format(ts+1))
                       for ts in range(lstm_seq_len)]
        lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")
        
        #short-term part
        #1st level gate
        #nbhd cnn
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time0_{0}".format(ts+1))(nbhd_inputs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu",name = "nbhd_convs_activation_time0_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        #flow cnn
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time0_{0}".format(ts+1))(flow_inputs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time0_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        #flow gate
        flow_gates = [Activation("sigmoid",
                                 name = "flow_gate0_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]])
                      for ts in range(lstm_seq_len)]
        
        
        #2nd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time1_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time1_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time1_{0}".format(ts+1))(flow_inputs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time1_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate1_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]])
                      for ts in range(lstm_seq_len)]

        #3rd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time2_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time2_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time2_{0}".format(ts+1))(flow_inputs[ts])
                      for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time2_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate2_{0}".format(ts+1))(flow_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]])
                      for ts in range(lstm_seq_len)]


        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_vecs = [Dense(units = cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1))(nbhd_vecs[ts])
                      for ts in range(lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts])
                      for ts in range(lstm_seq_len)]

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (lstm_seq_len, cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])

        #lstm
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        #attention part
        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_inputs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time0_{0}_{1}".format(att+1,ts+1))(att_flow_inputs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate0_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate1_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate2_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts])
                          for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Dense(units = cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts])
                          for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts])
                          for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]


        att_nbhd_vec    = [Concatenate(axis=-1)(att_nbhd_vecs[att])
                           for att in range(att_lstm_num)]
        att_nbhd_vec    = [Reshape(target_shape = (att_lstm_seq_len, cnn_flat_size))(att_nbhd_vec[att])
                           for att in range(att_lstm_num)]
        att_lstm_input  = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]])
                           for att in range(att_lstm_num)]

        att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att])
                     for att in range(att_lstm_num)]

        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm])
                       for att in range(att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(att_lstm_num, lstm_out_size))(att_low_level)


        att_high_level = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = Dense(units = output_shape)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        inputs = flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs,]
        # print("Model input length: {0}".format(len(inputs)))
        # ipdb.set_trace()
        model = Model(inputs = inputs, outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model
