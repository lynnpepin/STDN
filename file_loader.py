import numpy as np
import pickle
import json
from math import ceil

class file_loader:
    def __init__(self, n = 2, config_path = "data.json"):
        self.config             = json.load(open(config_path, "r"))
        # timeslot_sec = 1800; that is the amount of seconds in 30 minutes;
        # TODO: Add arg n, replace timeslot_sec with 3600/n
        #self.timeslot_daynum    = int(86400 / self.config["timeslot_sec"]) # = number of time slots per day (24*n);
        self.timeslot_daynum    = 24*n
        # Note: I'm using 'n' to mean the number of time slots per hour, and assuming it is 1 or an even integer
        self.threshold          = int(self.config["threshold"]) # Threshhold for filtering, used in main.py as sampler.threshhold
        self.isVolumeLoaded     = False
        self.isFlowLoaded       = False
        # self.volume_max and self.flow_max are normalizing constants, and are set during the first phase of sample_stdn()

    def base_sample(self, datatype):
        # Base level sampler
        # Returns data and flow_data,
        # sets self.isFlowLoaded, self.isVolumeLoader, self.volume_max, selfflow_max

        ## Loading the preprocessed datasets
        self.isFlowLoaded   = True
        self.isVolumeLoaded = True
        
        if datatype == "train":
            self.volume_max = self.config["volume_train_max"]
            self.flow_max = self.config["flow_train_max"]
            
            data = np.load(open(self.config["volume_train"], "rb"))["volume"] / self.volume_max

            flow_data = np.load(open(self.config["flow_train"], "rb"))["flow"] / self.flow_max
            

        elif datatype == "test":
            self.volume_max = self.config["volume_train_max"]
            self.flow_max = self.config["flow_train_max"]
            
            data = np.load(open(self.config["volume_test"], "rb"))["volume"] / self.volume_max
            flow_data = np.load(open(self.config["flow_test"], "rb"))["flow"] / self.flow_max

        elif datatype == "tiny":
            # TODO: Rework tiny/tiny2 to draw from the already-existing train and test datasets
            self.volume_max = 1289.0
            self.flow_max = 173.0
            
            data = np.load("data/volume_tiny.npz")['arr_0'] / self.volume_max# np.max(), as the above
            flow_data = np.load("data/flow_tiny.npz")['arr_0'] / self.flow_max # np.max(), as the above

        elif datatype == "tiny2":
            self.volume_max = 1283.0
            self.flow_max = 110.0
            
            data = np.load("data/volume_tiny2.npz")['arr_0'] / self.volume_max # np.max(), as the above
            flow_data = np.load("data/flow_tiny2.npz")['arr_0'] / self.flow_max # np.max(), as the above
        
        elif datatype[0:3] == 'man':
            # e.g. man-train-1, man-tiny-4, etc.
            data = np.load("data/man_volume.npz")['arr_0']
            flow_data = np.load("data/man_flow.npz")['arr_0'] 
            
            # Reshape from 4 slots per hour, if n == 2
            datasetn = int(datatype[-1]) # 4, 2, or 1
            
            assert(datasetn in (4, 2, 1))
            
            if datasetn == 2 or datasetn == 1:
                divisor = int(4 / datasetn)
                # The original dataset has 4.
                setsize = data.shape[0]
                newsetsize = int(setsize/divisor)
                
                data = data.reshape(newsetsize, divisor, 10, 20, 2).sum(axis=1)
                flow_data = flow_data.reshape(2, newsetsize, divisor, 10, 20, 10, 20).sum(axis=2)
            
            
            # Cut our dataset up depending on what set we're using
            dataset = datatype[4:-2] #train, test, tiny, or tiny2
            setsize = data.shape[0] # should be same as flow_data.shape[1]
            
            assert(dataset in ("train", "test", "tiny", "tiny2"))
            
            if dataset == 'train':
                subsetsize = int(setsize*2/3)
            elif dataset == 'test':
                subsetsize = int(setsize*2/3)
            elif dataset == 'tiny' or dataset == 'tiny2':
                subsetsize = 250*datasetn
                # Inspection: It seems the number of samples increases by 200
                # each time this number increases, for n=1.
                # e.g. 250 --> 1400 samples
                #      251 --> 1600 samples
                #      252 --> 1800 samples
                # Presumably, at 243 (where training starts) there is 0 samples.
            
            if dataset == 'train' or dataset == 'tiny':
                data = data[:subsetsize, :, :, :]
                flow_data = flow_data[:,:subsetsize,:,:,:,:]
            if dataset == 'test':
                data = data[subsetsize:, :, :, :]

                flow_data = flow_data[:,subsetsize:,:,:,:,:]
            elif dataset == 'test' or dataset == 'tiny2':
                data = data[-subsetsize:, :, :, :]
                flow_data = flow_data[:,-subsetsize:,:,:,:,:]
            
            # Train dataset values:
            #           vdata   fdata
            # n = 4:    693     59/117
            # n = 2:    1331    98/218
            # n = 1:    2604    166/433
            
            if datasetn == 4:
                self.volume_max = 693
                self.flow_max = 117
            elif datasetn == 2:
                self.volume_max = 1331
                self.flow_max = 218
            elif datasetn == 1:
                self.volume_max = 2604
                self.flow_max = 433
            
            data = data / self.volume_max
            flow_data = flow_data / self.flow_max
                
        else:
            self.isFlowLoaded = False
            self.isVolumeLoaded = False
            print("Please select valid data!")
            raise Exception
        
        return data, flow_data
    
    # TODO: sample_3DConv that produces a full list, X and y.
    def sample_3DConv(self,
                      datatype,
                      window_size = 7):
        # TODO: Test
        # Usage: Returns ((X tuple), (y tuple))
        # A simple sampler for 3DConv based architectures. Works over vdata.]
        
        data, flow_data = self.base_sample(datatype)
        
        def slider(AA, wsize):
            # Generates one batch of inputs
            end = len(AA)
            for ii in range(wsize,end):
                yield AA[ii-wsize:ii]

        def targets(AA, wsize):
            end = len(AA)
            return AA[wsize:end]
        
        X_in = tuple([x for x in slider(data, window_size)])
        y_out = tuple([x for x in targets(data, window_size)])

        return X_in, y_out 
    
    def sample_3DConv_generator(self,
                      datatype,
                      window_size = 7,
                      batch_size = 128):
        # TODO: Test
        # Usage:
        # >>> sampler = file_loader.file_loader(n=n)
        # >>> steps_per_epoch, data_generator = sampler.sample_3DConv(window_size = 7, batch_size = 64)
        # A simple sampler for 3DConv based architectures. Works over vdata.
        # Returns int number_of_batches and generator that yields of format (inputs, targets)
        
        data, flow_data = self.base_sample(datatype)
        
        def input_batch(AA, wsize, bsize, start):
            # Generates one batch of inputs
            end = min(len(AA), wsize+bsize+start)
            for ii in range(start+wsize, end):
                yield AA[ii-wsize:ii]

        def target_batch(AA, wsize, bsize, start):
            # Generates one batch of targets
            end = min(len(AA), wsize+bsize+start)
            return tuple(AA[wsize+start: end])

        def data_generator(AA, wsize, bsize):
            # Generates batches
            for start in range(0, len(AA), bsize): # Iterate from 0..len(AA) with steps of bsize
                inputs  = tuple(input_batch (AA, wsize, bsize, start))
                targets =       target_batch(AA, wsize, bsize, start)
                yield inputs, targets

        total_number_of_samples = len(data) - window_size
        number_of_batches = ceil(total_number_of_samples/batch_size)
        return number_of_batches, data_generator(AA = data, wsize = window_size, bsize = batch_size)
        # e.g:
        # d = [x for x in data_generator], wsize = 3, bsize = 128
        # d[ batch_number ][ target/input ][ number_in_batch ]
    
    #this function nbhd for cnn, and features for lstm, based on attention model
    def sample_stdn(self,
                    datatype,
                    att_lstm_num            = 3,  # In terms of days; leave unchanged
                    long_term_lstm_seq_len  = 3,  # In terms of number of time slots
                    short_term_lstm_seq_len = 7,  # In terms of number of time slots
                    hist_feature_daynum     = 7,  # In terms of days; leave unchanged
                    last_feature_num        = 48, # In terms of timeslots, should be the number of timeslots in a day (24*n)
                    nbhd_size               = 1,  # I'm guessing this is 3x3? 
                    cnn_nbhd_size           = 3): # e.g. convolutions are 7x7
                    # nbhd_size and cnn_nbhd_size might have to do with local-conv-net, implemented using Conv2D in the model.
                    # I'm not entirely sure, but (presumably) these are spatial.
        # TODO: Probably better to do this as a generator
        
        if long_term_lstm_seq_len % 2 != 1:
            print("Att-lstm seq_len must be odd!")
            raise Exception

        data, flow_data = self.base_sample(datatype)

        # Sampling begins here
        cnn_att_features  = []
        lstm_att_features = []
        flow_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features.append([])
            cnn_att_features.append([])
            flow_att_features.append([])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i].append([])
                flow_att_features[i].append([])
        
        cnn_features = []
        flow_features = []
        for i in range(short_term_lstm_seq_len):
            cnn_features.append([])
            flow_features.append([])
        
        short_term_lstm_features = []
        labels = []

        time_start = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        time_end = data.shape[0]
        volume_type = data.shape[-1]
        
        #import code
        #code.interact(local=locals())
        
        print("  Sampling starting at timeslot",time_start)
        for t in range(time_start, time_end):
            if t%100 == 0:
                print("  Now sampling at {0} timeslots.".format(t))
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    #sample common (short-term) lstm
                    short_term_lstm_samples = []
                    for seqn in range(short_term_lstm_seq_len):
                        # real_t from (t - short_term_lstm_seq_len) to (t-1)
                        real_t = t - (short_term_lstm_seq_len - seqn)
                        
                        #cnn features, zero_padding
                        cnn_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, volume_type))
                        #actual idx in data
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                #boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size),
                                            cnn_nbhd_y - (y - cnn_nbhd_size), :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                        cnn_features[seqn].append(cnn_feature)
                        
                        #flow features, 4 types
                        flow_feature_curr_out          = flow_data[0, real_t,     x, y, :, :]
                        flow_feature_curr_in           = flow_data[0, real_t,     :, :, x, y]
                        flow_feature_last_out_to_curr  = flow_data[1, real_t - 1, x, y, :, :]
                        #real_t - 1 is the time for in flow in longflow1
                        flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]
                        
                        flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))
                        
                        flow_feature[:, :, 0] = flow_feature_curr_out
                        flow_feature[:, :, 1] = flow_feature_curr_in
                        flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                        #calculate local flow, same shape cnn
                        local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4))
                        #actual idx in data
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                #boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                local_flow_feature[cnn_nbhd_x - (x - cnn_nbhd_size),
                                                   cnn_nbhd_y - (y - cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                        flow_features[seqn].append(local_flow_feature)

                        #lstm features
                        # nbhd feature, zero_padding
                        nbhd_feature = np.zeros((2*nbhd_size+1, 2*nbhd_size+1, volume_type))
                        #actual idx in data
                        for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                            for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                #boundary check
                                if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = data[real_t, nbhd_x, nbhd_y, :]
                        nbhd_feature = nbhd_feature.flatten()

                        #last feature
                        last_feature = data[real_t - last_feature_num: real_t, x, y, :].flatten()

                        #hist feature
                        hist_feature = data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, x, y, :].flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))

                        short_term_lstm_samples.append(feature_vec)
                    short_term_lstm_features.append(np.array(short_term_lstm_samples))

                    #sample att-lstms
                    for att_lstm_cnt in range(att_lstm_num):
                        
                        #sample lstm at att loc att_lstm_cnt
                        long_term_lstm_samples = []
                        # get time att_t, move forward for (att_lstm_num - att_lstm_cnt) day, then move back for ([long_term_lstm_seq_len / 2] + 1)
                        # notice that att_t-th timeslot will not be sampled in lstm
                        # e.g., **** (att_t - 3) **** (att_t - 2) (yesterday's t) **** (att_t - 1) **** (att_t) (this one will not be sampled)
                        # sample att-lstm with seq_len = 3
                        att_t = t - (att_lstm_num - att_lstm_cnt) * self.timeslot_daynum + (long_term_lstm_seq_len - 1) / 2 + 1
                        att_t = int(att_t)
                        #att-lstm seq len
                        for seqn in range(long_term_lstm_seq_len):
                            # real_t from (att_t - long_term_lstm_seq_len) to (att_t - 1)
                            real_t = att_t - (long_term_lstm_seq_len - seqn)

                            #cnn features, zero_padding
                            cnn_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, volume_type))
                            #actual idx in data
                            for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                    #boundary check
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    #get features
                                    # import ipdb; ipdb.set_trace()
                                    cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            #flow features, 4 type
                            flow_feature_curr_out = flow_data[0, real_t, x, y, :, :]
                            flow_feature_curr_in = flow_data[0, real_t, :, :, x, y]
                            flow_feature_last_out_to_curr = flow_data[1, real_t - 1, x, y, :, :]
                            #real_t - 1 is the time for in flow in longflow1
                            flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]

                            flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))
                            
                            flow_feature[:, :, 0] = flow_feature_curr_out
                            flow_feature[:, :, 1] = flow_feature_curr_in
                            flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                            flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                            #calculate local flow, same shape cnn
                            local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4))
                            #actual idx in data
                            for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                    #boundary check
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    #get features
                                    local_flow_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                            flow_att_features[att_lstm_cnt][seqn].append(local_flow_feature)

                            #att-lstm features
                            # nbhd feature, zero_padding
                            nbhd_feature = np.zeros((2*nbhd_size+1, 2*nbhd_size+1, volume_type))
                            #actual idx in data
                            for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                                for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                    #boundary check
                                    if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                        continue
                                    #get features
                                    nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = data[real_t, nbhd_x, nbhd_y, :]
                            nbhd_feature = nbhd_feature.flatten()

                            #last feature
                            last_feature = data[real_t - last_feature_num: real_t, x, y, :].flatten()

                            #hist feature
                            hist_feature = data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, x, y, :].flatten()

                            feature_vec = np.concatenate((hist_feature, last_feature))
                            feature_vec = np.concatenate((feature_vec, nbhd_feature))

                            long_term_lstm_samples.append(feature_vec)
                        lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))

                    #label
                    labels.append(data[t, x , y, :].flatten())


        output_cnn_att_features = []
        output_flow_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features[i] = np.array(lstm_att_features[i])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                flow_att_features[i][j] = np.array(flow_att_features[i][j])
                output_cnn_att_features.append(cnn_att_features[i][j])
                output_flow_att_features.append(flow_att_features[i][j])
        
        for i in range(short_term_lstm_seq_len):
            cnn_features[i] = np.array(cnn_features[i])
            flow_features[i] = np.array(flow_features[i])
        short_term_lstm_features = np.array(short_term_lstm_features)
        labels = np.array(labels)
        print("  Finished sampling from data.")
        
        # for n = 1 on the tiny dataset (243 to 250)
        # output_cnn_att_features:  List of Numpy array, length 9
        #     Each element is a (1400, 7, 7, 2) float64 array
        # output_cnn_att_features:  List of Numpy array, length 9
        #     Each element is a (1400, 7, 7, 4) float64 array
        # lstm_att_features:        List of Numpy array, length 3
        #     Each element is a (1400, 3, 112) float64 array
        # cnn_features:             List of Numpy array, length 3
        #     Each element is a (1400, 7, 7, 2) float64 array
        # flow_features:            List of Numpy array, length 3
        #     Each element is a (1400, 7, 7, 4) float64 array
        # short_term_lstm_features: A float64 array, shape (1400, 3, 12)
        # labels:                   A float64 array, shape (1400, 2)
        return output_cnn_att_features, output_flow_att_features, lstm_att_features, cnn_features, \
               flow_features, short_term_lstm_features, labels
        
        # in main.py, these are passed to the models as inputs by list concatenation.
        
        
        
