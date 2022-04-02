import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN
from models import GAT
from tensorflow.contrib import rnn

class dense_layer():
    def __init__(self,hidden_graph):
        self.n_hidden_gru = 224
        self.n_hidden_dense1 = 64
        self.n_hidden_dense2 = 32
        self.n_hidden_dense3 = 16
        self.initializer = tf.random_normal_initializer(stddev=0.01)
        self.hidden_graph = hidden_graph
        self.activation = tf.nn.relu
        self.build_var()
        self.pred = self.build_model()
        
    def build_var(self):
        with tf.variable_scope('dense'):
            self.weights = {
                        'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                        self.n_hidden_dense1])),
                       'dense2': tf.get_variable('dense2_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                       self.n_hidden_dense2])),
                       'dense3': tf.get_variable('dense3_weight', initializer=self.initializer([self.n_hidden_dense2, self.n_hidden_dense3])),
                        'out': tf.get_variable('out_weight', initializer=self.initializer([self.n_hidden_dense3, 1]))
                }
            self.biases = {
                    'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([self.n_hidden_dense1])),
                   'dense2': tf.get_variable('dense2_bias', initializer=self.initializer([self.n_hidden_dense2])),
                   'dense3': tf.get_variable('dense3_bias', initializer=self.initializer([self.n_hidden_dense3])),
                    'out': tf.get_variable('out_bias', initializer=self.initializer([1]))
                }
            
    def build_model(self):
        with tf.variable_scope('dense'):
            dense1 = self.activation(tf.add(tf.matmul(self.hidden_graph, self.weights['dense1']), self.biases['dense1']))
            dense2 = self.activation(tf.add(tf.matmul(dense1, self.weights['dense2']), self.biases['dense2']))
            dense3 = self.activation(tf.add(tf.matmul(dense2, self.weights['dense3']), self.biases['dense3']))
            pred = self.activation(tf.add(tf.matmul(dense3, self.weights['out']), self.biases['out']))
        return pred

class lstm_layer():
    def __init__(self,x):
        self.n_gru = 32
        self.x = x
        self.build_var()
        self.results = self.build_model()
        
    def build_var(self):
        with tf.variable_scope('Bigru'):
            self.gru_fw_cell = tf.nn.rnn_cell.LSTMCell(self.n_gru)
            self.gru_bw_cell = tf.nn.rnn_cell.LSTMCell(self.n_gru)
            
    def build_model(self):
        with tf.variable_scope('Bigru'):
            outputs , _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = self.gru_fw_cell, cell_bw = self.gru_bw_cell, inputs = self.x, dtype = tf.float32)
            hidden_states = tf.concat(outputs,2)
        return hidden_states

class UCGGAT(BaseGAttN):
    def inference(inputs, nb_nodes, training, attn_drop, ffd_drop,bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        print("ok")
        lstm_input = []
        for i in range(7):
            
            model_gat = GAT
            results = model_gat.inference_gat(inputs[:,i,:,:], nb_nodes, training, attn_drop, ffd_drop, bias_mat[:,i,:,:], hid_units, n_heads, activation=tf.nn.elu, residual=False)
            results = tf.reduce_sum(results,1)
            lstm_input.append(results)
        lstm_input = tf.convert_to_tensor(lstm_input, dtype = tf.float32)
        
        lstm_input = tf.transpose(lstm_input,[1,0,2])
        model_lstm = lstm_layer(lstm_input)
        lstm_input = model_lstm.results
        
        lstm_input = tf.reshape(lstm_input, [2,-1])
        model_dense = dense_layer(lstm_input)
        lstm_output = model_dense.pred
        #lstm_output = tf.reshape(lstm_input, [2,-1])
        print("lstm_output",lstm_output.get_shape())
        out = tf.reshape(lstm_output,[2,1])
    
        return out
    
    