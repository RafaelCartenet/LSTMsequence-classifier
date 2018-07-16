# Libraries
import tensorflow as tf
from logger import Logger
import numpy as np
import os
import sys
import time


class LSTMSequenceClassifier:
    def __init__(self,
                 models_path,   # path where will be save/restore models
                 model_name,    # name of the folder of models
                 VERBOSE=0,     # if 1, prints additionnal informations
                 SAVE=1):       # if 1, save model after each epoch
        self.SAVE= SAVE
        self.MODELS_PATH= models_path
        self.MODEL_NAME= model_name
        self.logger= Logger(VERBOSE)

        self.max_length= 100    # Sequence max length for training

    def load_data(self, X_train, Y_train, X_test, Y_test):
        self.X_train= np.asarray(X_train)
        self.Y_train= np.asarray(Y_train)
        self.X_test= np.asarray(X_test)
        self.Y_test= np.asarray(Y_test)

        # Determine the set of labels we are working on
        labels= set(np.unique(self.Y_train))
        labels= labels.union(set(np.unique(self.Y_test)))
        self.labels= sorted(labels)

        # Define the input and output size
        self.input_size= self.X_train[0].shape[1]
        self.output_size= len(self.labels)

    def arrange_data(self, batchsize):
        logstring=  '\nDATA SHAPES BEFORE ARRANGEMENT\n'
        logstring+= 'X_train shape %s\n'% (self.X_train.shape,)
        logstring+= 'Y_train shape %s\n'% (self.Y_train.shape,)
        logstring+= 'X_test shape %s\n'% (self.X_test.shape,)
        logstring+= 'Y_test shape %s\n'% (self.Y_test.shape,)
        self.logger.write_log(logstring)

        nb_trainsamples= len(self.X_train)
        nb_testsamples= len(self.X_test)
        self.batchsize= batchsize

        # TRANSFORM LABELS TO ONE HOT VECTORS.
        # Training data labels
        onehots= np.zeros((nb_trainsamples, self.output_size), dtype='f')
        for i in range(nb_trainsamples):
            ind= self.labels.index(self.Y_train[i])
            onehots[i, ind]= 1
        self.Y_train= onehots

        # Testing data labels
        onehots= np.zeros((nb_testsamples, self.output_size), dtype='f')
        for i in range(nb_testsamples):
            ind= self.labels.index(self.Y_test[i])
            onehots[i, ind]= 1
        self.Y_test= onehots

        # # Compute maximum length among samples
        # self.max_length= max([len(sample) for sample in np.concatenate((self.X_train, self.X_test), axis=0)])

        # Create np arrays for X and lengths
        X_train= np.zeros((nb_trainsamples, self.max_length, self.input_size), dtype='f')
        self.lengths_train= np.zeros((nb_trainsamples), dtype=np.uint8)
        X_test= np.zeros((nb_testsamples, self.max_length, self.input_size), dtype='f')
        self.lengths_test= np.zeros((nb_testsamples), dtype=np.uint8)

        # Fill the created arrays with the data
        # training data
        for i in range(nb_trainsamples):
            self.lengths_train[i]= int(len(self.X_train[i]))
            X_train[i,:int(self.lengths_train[i])]= self.X_train[i]
        self.X_train= X_train
        # testing data
        for i in range(nb_testsamples):
            self.lengths_test[i]= int(len(self.X_test[i]))
            X_test[i,:int(self.lengths_test[i])]= self.X_test[i]
        self.X_test= X_test

        # Compute number of batches for each set
        self.nb_batch_train= self.X_train.shape[0]//self.batchsize
        self.nb_batch_test= self.X_test.shape[0]//self.batchsize

        # Shuffle the training set according to a random permutation
        train_perm= np.random.permutation(self.nb_batch_train*self.batchsize)
        self.X_train= self.X_train[:self.nb_batch_train*self.batchsize][train_perm]
        self.Y_train= self.Y_train[:self.nb_batch_train*self.batchsize][train_perm]
        self.lengths_train= self.lengths_train[:self.nb_batch_train*self.batchsize][train_perm]

        # Shuffle the testing set according to a random permutation
        test_perm= np.random.permutation(self.nb_batch_test*self.batchsize)
        self.X_test= self.X_test[:self.nb_batch_test*self.batchsize][test_perm]
        self.Y_test= self.Y_test[:self.nb_batch_test*self.batchsize][test_perm]
        self.lengths_test= self.lengths_test[:self.nb_batch_test*self.batchsize][test_perm]

        # Reshape training data to batches
        self.X_train= self.X_train.reshape(self.nb_batch_train, -1, self.max_length, self.input_size)
        self.Y_train= self.Y_train.reshape(self.nb_batch_train, -1, self.output_size)
        self.lengths_train= self.lengths_train.reshape(self.nb_batch_train, -1)

        # Reshape testing data to batches
        self.X_test= self.X_test.reshape(self.nb_batch_test, -1, self.max_length, self.input_size)
        self.Y_test= self.Y_test.reshape(self.nb_batch_test, -1, self.output_size)
        self.lengths_test= self.lengths_test.reshape(self.nb_batch_test, -1)


        logstring=  '\nDATA SHAPES AFTER ARRANGEMENT\n'
        logstring+= 'X_train shape %s\n'% (self.X_train.shape,)
        logstring+= 'Y_train shape %s\n'% (self.Y_train.shape,)
        logstring+= 'X_test shape %s\n'% (self.X_test.shape,)
        logstring+= 'Y_test shape %s\n'% (self.Y_test.shape,)
        logstring+= 'lengths_train %s\n'% (self.lengths_train.shape,)
        logstring+= 'lengths_test %s\n'% (self.lengths_test.shape,)
        self.logger.write_log(logstring)


    def extract_lasts(self, data, ind):
        """ Extract the last element of a numpy matrix
        :param data: numpy matrix (shape ())
        """
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        return tf.gather_nd(data, indices)

    #### Building Graph ####
    def build_graph(self,
                   nb_hiddenunits=100,
                   n_RNNlayers=2,
                   learning_rate=1e-4,
                   name="",
                   dropout=-1,
                   dropoutwrapper=-1):
        self.build_graph_variables(nb_hiddenunits=nb_hiddenunits,
                                   n_RNNlayers=n_RNNlayers,
                                   dropout=dropout,
                                   dropoutwrapper=dropoutwrapper)
        self.build_graph_optimizer(learning_rate)
        self.build_graph_summaries(name)

        logstring=  '\nGRAPH VARIABLES SHAPES:\n'
        logstring+= 'INPUT:\t\t%s\n'% (self.X_.get_shape(),)
        logstring+= 'RNN outputs:\t%s\n'% (self.outputs.get_shape(),)
        logstring+= 'last output:\t%s\n'% (self.lasts.get_shape(),)
        logstring+= 'W shape:\t%s\n'% (self.W.get_shape(),)
        logstring+= 'b shape:\t%s\n'% (self.b.get_shape(),)
        logstring+= 'OUTPUT:\t\t%s\n'% (self.predictions.get_shape(),)
        self.logger.write_log(logstring)


    def build_graph_variables(self,
                    nb_hiddenunits=100,
                    n_RNNlayers=2,
                    dropout=-1,
                    dropoutwrapper=-1):

        tf.reset_default_graph()

        with tf.variable_scope(self.group):
            # Input placeholders
            self.X_= tf.placeholder(tf.float32, [None, None, self.input_size], name='input')
            self.Y_= tf.placeholder(tf.float32, [None, self.output_size], name='output')
            self.seq_lengths= tf.placeholder(tf.int32, [None], name='seq_length')
            self.keep_prob_inter= tf.placeholder_with_default(1., shape=(), name="keep_prob_inter")
            self.keep_prob_warp= tf.placeholder_with_default(1., shape=(), name="keep_prob_warp")


            # RNN
            ## RNN Cell and Dynamic RNN, memory size= nb_hiddenunits
            stacked_cells= []
            for i in range(n_RNNlayers):
                # Define reccurent cell
                cell= tf.contrib.rnn.LSTMCell(nb_hiddenunits)

                # Dropout warper on reccurent cell
                cell= tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob= self.keep_prob_warp)

                # Stack the cell
                stacked_cells.append(cell)


            self.stacked_cells= tf.contrib.rnn.MultiRNNCell(stacked_cells, state_is_tuple=True)
            self.outputs,_= tf.nn.dynamic_rnn(
                self.stacked_cells,
                self.X_,
                sequence_length= self.seq_lengths,
                time_major=False,
                dtype=tf.float32
            )

            # Extract the last outputs for each sequence, according to their length.
            self.lasts= self.extract_lasts(self.outputs, self.seq_lengths - 1)

            # Inter Dropout layer
            self.lasts= tf.nn.dropout(self.lasts, keep_prob= self.keep_prob_inter)

            # Final fully connected layer
            self.W= tf.Variable(tf.truncated_normal([nb_hiddenunits, self.output_size], stddev= 0.1), name= "FC_weights")
            self.b= tf.Variable(tf.truncated_normal([self.output_size], stddev=0.1), name= "FC_bias")

            # Predictions
            self.predictions= tf.nn.softmax(tf.matmul(self.lasts, self.W) + self.b, name='predictions')

        # saver to store and restore variables
        self.saver= tf.train.Saver()


    def build_graph_optimizer(self, learning_rate=1e-4,):
        # Cost function
        self.cross_entropy= -tf.reduce_sum(self.Y_ * tf.log(self.predictions))

        regularizer= tf.nn.l2_loss(self.W)
        self.loss= tf.reduce_mean(self.cross_entropy + .01*regularizer)

        # Optimizer
        self.learning_rate= learning_rate
        self.optimizer= tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # Accuracy
        self.acc= tf.equal(tf.argmax(self.Y_, 1), tf.argmax(self.predictions, 1))
        self.acc= tf.reduce_mean(tf.cast(self.acc, tf.float32))

    def build_graph_summaries(self, name= ""):
        self.name= name
        tf.summary.scalar(self.name+'acc', self.acc)
        tf.summary.scalar(self.name+'loss', self.loss)
        self.summary_op= tf.summary.merge_all()

    def train_model(self,
        keep_prob_inter,
        keep_prob_warp,
        nb_epoch=200
    ):
        """ Train the model for nb_epoch iterations.
            Model is restored from models_path/model_name.cpkt
            If model could not be loaded, random initialisation for parameters
            After each epoch the model parameters are saved and model
            is tested on the whole testing set
        """
        # Initialize the variables
        sess= tf.Session()
        sess.run(tf.global_variables_initializer())

        # Load variables if needed
        if self.SAVE:
            try:
                modelname= self.MODELS_PATH + self.name
                saver= tf.train.import_meta_graph(modelname + ".meta")
                self.saver.restore(sess, modelname)
            except:
                print "Failed to restore. Starting model from scratch."

        # Define log writers for Tensorboard
        self.file_writer_train= tf.summary.FileWriter("tflogs/training_set/"+self.name, sess.graph)
        self.file_writer_test= tf.summary.FileWriter("tflogs/testing_set/"+self.name, sess.graph)

        for epoch_id in range(nb_epoch):
            #### TRAINING ####
            epoch_time= time.time()
            epoch_acc= 0.
            epoch_loss= 0.
            self.logger.write_log("epo#%s\t|batch#\t|accu\t|loss\t|time\t|"% (epoch_id+1))
            for batch_id in range(self.nb_batch_train):
                batch_time= time.time()

              # input
                batch_X, batch_Y= self.X_train[batch_id], self.Y_train[batch_id]
                lengths= self.lengths_train[batch_id]
              # optimize and compute loss + accuracy
                _, loss, acc, summary= sess.run(
                    fetches= [
                        self.optimizer,
                        self.loss,
                        self.acc,
                        self.summary_op
                    ],
                    feed_dict= {
                        self.X_: batch_X,
                        self.Y_: batch_Y,
                        self.seq_lengths: lengths,
                        self.keep_prob_inter: keep_prob_inter,
                        self.keep_prob_warp: keep_prob_warp
                    }
                )
              # update infos
                epoch_acc+= acc;epoch_loss+= loss
                batch_time= time.time()-batch_time

              # display results
                if batch_id % 50 == 0:
                    self.logger.write_log("\t|%s\t|%.2f%%\t|%.2f\t|%.2fs\t|"% (batch_id, 100*acc, loss, batch_time))

              # write log to tensorboard
                self.file_writer_train.add_summary(summary, epoch_id*self.nb_batch_train+ batch_id)

            # update epoch infos
            epoch_acc/= self.nb_batch_train
            epoch_loss/= self.nb_batch_train
            epoch_time= time.time()- epoch_time
            self.logger.write_log("\t|ALL\t|%.2f%%\t|%.2f\t|%.2fs\t|"% (100*epoch_acc, epoch_loss, epoch_time))


            #### SAVING MODEL ####
            if self.SAVE:
                self.saver.save(sess, self.MODELS_PATH + self.name)


            #### TESTING ####
            testing_time= time.time()
            testing_acc= 0
            testing_loss= 0
            for batch_id in range(self.nb_batch_test):
              # input
                batch_X, batch_Y= self.X_test[batch_id], self.Y_test[batch_id]
                lengths= self.lengths_test[batch_id]
              # compute loss + accuracy
                loss, acc, summary= sess.run(
                    fetches= [
                        self.loss,
                        self.acc,
                        self.summary_op
                    ],
                    feed_dict= {
                        self.X_: batch_X,
                        self.Y_: batch_Y,
                        self.seq_lengths: lengths
                    }
                )
              # update infos
                testing_acc+= acc
                testing_loss+= loss

              # write log to tensorboard
                self.file_writer_test.add_summary(summary, epoch_id*self.nb_batch_test+ batch_id)

            testing_time= time.time()- testing_time
            testing_acc/= self.nb_batch_test
            testing_loss/= self.nb_batch_test
            self.logger.write_log("\t|TEST\t|%.2f%%\t|%.2f\t|%.2fs\t|\n"% (100*testing_acc, testing_loss, testing_time))

    def evaluate_model(self):
        # Initialize the variables
        sess= tf.Session()
        sess.run(tf.global_variables_initializer())

        # Load variables
        if self.SAVE:
            try:
                modelname= self.MODELS_PATH + self.name
                saver= tf.train.import_meta_graph(modelname + ".meta")
                self.saver.restore(sess, modelname)
            except:
                print "Failed to restore model. Exiting."
                exit()

        #### TESTING ####
        Y_true= np.zeros((self.nb_batch_test*self.batchsize, self.output_size))
        Y_pred= np.zeros((self.nb_batch_test*self.batchsize, self.output_size))
        testing_time= time.time()
        testing_acc= 0
        testing_loss= 0
        for batch_id in range(self.nb_batch_test):
            batch_time= time.time()

          # input
            batch_X, batch_Y= self.X_test[batch_id], self.Y_test[batch_id]
            Y_true[batch_id*self.batchsize:(batch_id+1)*self.batchsize]= batch_Y
            lengths= self.lengths_test[batch_id]
          # compute loss + accuracy
            loss, acc, summary= sess.run(
                fetches= [
                    self.loss,
                    self.acc,
                    self.summary_op
                ],
                feed_dict= {
                    self.X_: batch_X,
                    self.Y_: batch_Y,
                    self.seq_lengths: lengths
                }
            )
          # update infos
            testing_acc+= acc
            testing_loss+= loss
            Y_pred[batch_id*self.batchsize:(batch_id+1)*self.batchsize]= predictions

        testing_time= time.time()- testing_time
        testing_acc/= self.nb_batch_test
        testing_loss/= self.nb_batch_test
        self.logger.write_log("\n\nAccuracy:\t%.2f%%\nLoss:\t\t%s\nTime:\t\t%.2fs\n"% (100*testing_acc, testing_loss, testing_time))
        Y_true= np.argmax(Y_true, axis= 1)
        Y_pred= np.argmax(Y_pred, axis=1)
        tophonetic= np.vectorize(lambda t: sorted(self.labels)[t])
        Y_true= tophonetic(Y_true)
        Y_pred= tophonetic(Y_pred)

        self.logger.write_log(CR(Y_true, Y_pred))
        mat= CM(Y_true, Y_pred)
        CONFMAT= "\t" + "\t".join(sorted(self.labels)) + "\n"
        for i,phonetic in enumerate(sorted(self.labels)):
            CONFMAT+= phonetic + "\t"+"\t".join(map(str, mat[i].tolist()+[np.sum(mat[i])])) + "\n\n"
        CONFMAT+= "\t" + "\t".join(map(str, np.sum(mat, axis=0).tolist()))
        self.logger.write_log(CONFMAT)

    def init_session(self):
        """ Method to inialize a new tensorflow session.
            From the model path and the model name given,
                - Load the meta graph
                - Restore saved variables
            No need to build graph from scratch.
            Returns the session so that it can be saved in cache.
        """
        # Initialize a new tensorflow session
        tf.reset_default_graph()
        session= tf.Session()
        session.run(tf.global_variables_initializer())

        # Define full path of the model
        full_path_model= self.MODELS_PATH + self.MODEL_NAME
        try:
            # Restore graph from the .meta file
            saver= tf.train.import_meta_graph(full_path_model + ".meta")

            # Restore parameters
            saver.restore(session, full_path_model)
        except:
            print "Failed to restore RNN model: %s.\nEXIT."% (full_path_model)
            exit()
        return session


    def predict(self, session, data):
        """ Method to predict output from new samples, data.
            It is using a preloaded tensorflow session.
            Format of data should be [batchsize, X, self.inputsize]
        """
        # Determine batchsize and input size according to data
        batchsize= data.shape[0]
        max_length= data.shape[1]
        input_size= data[0][1].shape[0]

        # Compute the length of each sample and apply zero padding
        # Load datastructures to store length and new data
        lengths= np.zeros((batchsize))
        new_data= np.zeros((batchsize, max_length, input_size), dtype='f')

        # Fill the structures
        for i in range(batchsize):
            lengths[i]= len(data[i])
            new_data[i,:int(lengths[i])]= data[i]

        # Compute the prediction using pre-loaded session
        predictions= session.run(
            fetches= [
                '%s/predictions:0'% (self.group)
            ],
            feed_dict= {
                '%s/input:0'% (self.group): new_data,
                '%s/seq_length:0'% (self.group): lengths
            }
        )

        return predictions



def forward_test():
    Clfr= PhonemeClassifier('vowels')
    Clfr.load_data()
    Clfr.arrange_data()
    Clfr.build_graph()

    sess= tf.Session()
    sess.run(tf.global_variables_initializer())

    X= Clfr.X_train[0]
    Y= Clfr.Y_train[0]
    lengths= Clfr.lengths_train[0]

    outputs, lasts, predictions, acc, loss= sess.run(
        [Clfr.X_, Clfr.lasts, Clfr.predictions, Clfr.acc, Clfr.cross_entropy],
        feed_dict={
            Clfr.X_: X,
            Clfr.Y_: Y,
            Clfr.seq_lengths: lengths,
        }

    )

    print "\nOUTPUTS\n", outputs
    print "\nLASTS\n", lasts
    print "\nPREDICTIONS\n", predictions
    print "\nREAL LABELS\n", Y
    print "\nACCURACY\n%.2f%%"% acc
    print "\nLOSS\n%s"% loss


def predict_test():
    Clfr= PhonemeClassifier('fricatives')
    Clfr.build_graph()

    sess= tf.Session()
    sess.run(tf.global_variables_initializer())

    A= np.zeros((6, 13), dtype='f')
    yA= np.zeros(8)
    B= np.zeros((7, 13), dtype='f')
    yB= np.zeros(8)

    outputs= sess.run([Clfr.predictions], feed_dict= {
        Clfr.X_:A,
        Clfr.Y_:yA,
        Clfr.lengths:[6]
    })

    print outputs

if __name__ == "__main__":
    # forward_test()
    start= time.time()
    predict_test()
    print 'Running time: %.2f' % (time.time() - start)
