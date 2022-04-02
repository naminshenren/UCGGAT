import numpy as np
import tensorflow as tf
from models import GAT
from models import UCGGAT

data = np.random.rand(100,7,132,64)
y = np.random.rand(100,1)
bias = np.random.rand(100,7,132,132)
avg = np.mean(bias)
data = data.astype("float32")
y = y.astype("float32")
bias = bias.astype("float32")
bias[bias>avg] = 1
bias[bias<=avg] = 0

data_train = data[0:70]
data_val = data[70:85]
data_test = data[85:100]
bias_train = bias[0:70]
bias_val = bias[70:85]
bias_test = bias[85:100]
y_train = y[0:70]
y_val = y[70:85]
y_test = y[85:100]

batch_size = 2
num_graphs = 7
nb_nodes = 132
ft_size = 64
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = UCGGAT

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_graphs, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_graphs, nb_nodes, nb_nodes))
        y = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())
        

    logits = model.inference(inputs=ftr_in, nb_nodes=nb_nodes, training=is_train,attn_drop=attn_drop, ffd_drop=ffd_drop,bias_mat=bias_in, hid_units=hid_units, n_heads=n_heads, residual=residual, activation=nonlinearity)
    
    loss = model.mse(logits, y)
   

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = 100
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = data_train.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr= sess.run([train_op, loss],
                    feed_dict={
                        ftr_in: data_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: bias_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        y: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                #print("loss_value_tr",loss_value_tr)
                train_loss_avg += loss_value_tr
                tr_step += 1

            vl_step = 0
            vl_size = data_val.shape[0]

            while (vl_step+1) * batch_size < vl_size:
                loss_value_vl= sess.run([loss],
                    feed_dict={
                        ftr_in: data_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: bias_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        y: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                #print(loss_value_vl[0])
                val_loss_avg += loss_value_vl[0]
                vl_step += 1

            print('Training: loss = %.5f,| Val: loss = %.5f' %
                    (train_loss_avg/tr_step, val_loss_avg/vl_step))

            if val_loss_avg/vl_step <= vlss_mn:
                if val_loss_avg/vl_step <= vlss_mn:
                    vlss_early_model = val_loss_avg/vl_step
                    #saver.save(sess, checkpt_file)
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn)
                    print('Early stop model validation loss: ', vlss_early_model)
                    break

            train_loss_avg = 0
            
            val_loss_avg = 0
           

        #saver.restore(sess, checkpt_file)

        ts_size = data_test.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while (ts_step+1) * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: data_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: bias_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step)

        sess.close()
