from train_model_py3 import *
from test_model_py3 import *
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import numpy as np

OUTPUT_PATH = '../create_memory_graphs/graph/'
input_size = 64
hidden_size = 128
model_path = 'model/_ETHREAD/20240413_214248/struct2vec_edge_type.HOP3._ETHREAD-1000'

def load_data_G(model, target_obj_type='_ETHREAD'):
    log('Object type: ' + target_obj_type)
    
    # load the threshold of the target object type
    obj_type_threshold = load_threshold(target_obj_type)

    dict_key_node_to_weight = load_key_node_weight(target_obj_type)
    log(str(dict_key_node_to_weight))
    output_vector_size = get_outout_vector_size(dict_key_node_to_weight)
    log('output_vector_size:\t%d' %output_vector_size)

    _, _, list_file_test = get_file_list(target_obj_type)
    log(list_file_test)
    i = 0
    for file_graph in list_file_test:
        log(file_graph)
        list_vector, _, _, _, _, _, _, _ = read_dataset(file_graph, output_vector_size, dict_key_node_to_weight, target_obj_type)
        list_vector = torch.tensor(list_vector)
        outputs = []

        dataset = tf.data.Dataset.from_tensor_slices(list_vector)
        dataset = dataset.shuffle(buffer_size=1024).batch(1)

        for _, row in enumerate(dataset):
            predictions = generator(row)
            outputs.append(predictions)

        outputs = tf.concat(outputs, axis=0)
        
        with open(file_graph, 'r') as f:
            lines = [x for x in f]

        for idx, line in enumerate(lines):
            s = line.strip().split('\t')
            obj_type = None
            if len(s) >= 8:
                obj_type = s[7].split('@')[0]
            if obj_type == target_obj_type:
                n_bit = int(s[-3])
                for idx_bit in range(n_bit, 32):
                    outputs[idx][idx_bit] = 0
            else:
                outputs[idx] = list_vector[idx]

        outputs = outputs.detach().numpy().astype(np.uint8)
        np.save(f'g_model/output.{i}.npy', outputs)
        i += 1
    return outputs

class Generator(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super(Generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(input_size, activation='relu')
        self.final_activation = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.final_activation(x) * 255
        return tf.math.round(x)

def train_G(generator, inputs, optimizer):
    for epoch in range(10):
        with tf.GradientTape() as tape:
            loss = -np.log(loss_D(inputs, model_path))
        grads = tape.gradient(loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        print(f'Epoch {epoch+1}, Step {step}, Loss: {loss.numpy()}')
    generator.save('g_model/model.h5')

def loss_D(inputs, model_path):
    target_obj_type = '_ETHREAD'
    # load the threshold of the target object type
    obj_type_threshold = load_threshold(target_obj_type)
    dict_key_node_to_weight = load_key_node_weight(target_obj_type)
    output_vector_size = get_outout_vector_size(dict_key_node_to_weight)
    # tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, [None, VECTOR_SIZE], name='x')
    ln = tf.compat.v1.sparse_placeholder(tf.float32, name='input_ln')
    rn = tf.compat.v1.sparse_placeholder(tf.float32, name='input_rn')
    lp = tf.compat.v1.sparse_placeholder(tf.float32, name='input_lp')
    rp = tf.compat.v1.sparse_placeholder(tf.float32, name='input_rp')
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    W_1 = tf.compat.v1.get_variable('W_1', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    W_2 = tf.compat.v1.get_variable('W_2', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    W_3 = tf.compat.v1.get_variable('W_3', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    W_4 = tf.compat.v1.get_variable('W_4', dtype=tf.float32, shape=[VECTOR_SIZE, output_vector_size], trainable=False)
    P_1 = tf.compat.v1.get_variable('P_1', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_2 = tf.compat.v1.get_variable('P_2', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_3 = tf.compat.v1.get_variable('P_3', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_4 = tf.compat.v1.get_variable('P_4', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_5 = tf.compat.v1.get_variable('P_5', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_6 = tf.compat.v1.get_variable('P_6', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_7 = tf.compat.v1.get_variable('P_7', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_8 = tf.compat.v1.get_variable('P_8', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_9 = tf.compat.v1.get_variable('P_9', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_10 = tf.compat.v1.get_variable('P_10', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_11 = tf.compat.v1.get_variable('P_11', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_12 = tf.compat.v1.get_variable('P_12', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_path)
        log('Model restored.')
        mu = tf.zeros_like(x)
        for t in range(HOP):
            l1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.matmul(tf.compat.v1.sparse_tensor_dense_matmul(ln,mu),P_1),P_2),P_3)),keep_prob)
            l2 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.matmul(tf.compat.v1.sparse_tensor_dense_matmul(rn,mu),P_4),P_5),P_6)),keep_prob)
            l3 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.matmul(tf.compat.v1.sparse_tensor_dense_matmul(lp,mu),P_7),P_8),P_9)),keep_prob)
            l4 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.matmul(tf.compat.v1.sparse_tensor_dense_matmul(rp,mu),P_10),P_11),P_12)),keep_prob)
            mu = tf.nn.tanh(tf.matmul(x, W_1) + l1 + l2 + l3 + l4)
        y = tf.nn.dropout(tf.matmul(tf.matmul(tf.matmul(mu, W_2), W_3), W_4), keep_prob)

        _, _, list_file_test = get_file_list(target_obj_type)
        log(list_file_test)
        for file_graph in list_file_test:
            log(file_graph)
            list_vector, list_label, ln_matrix, rn_matrix, lp_matrix, rp_matrix, dict_idx_to_addr, set_obj_addr = read_dataset(file_graph, output_vector_size, dict_key_node_to_weight, target_obj_type)
            ye = sess.run(y, feed_dict={x:list_vector, ln:ln_matrix, rn:rn_matrix, lp:lp_matrix, rp:rp_matrix, keep_prob:1})
            dict_addr_to_weight = get_addr_to_weight(ye, dict_idx_to_addr, dict_key_node_to_weight)
                
    TP_amount = FP_amount = FN_amount = TN_amount = 0
    for addr, weight in iter(dict_addr_to_weight.items()):
        if weight > threshold:
            if addr in set_obj_addr:
                TP_amount += 1
            else:
                FP_amount += 1
        else:
            if addr in set_obj_addr:
                FN_amount += 1
            else:
                TN_amount += 1
    FN_amount = len(set_obj_addr) - TP_amount
    precision_p = precision_t = eecall_p = recall_t = F1 = 0.0

    return float(TP_amount) / (TP_amount + FN_amount)
    
def log(message):
    print('%s\t%s' %(strftime("%Y-%m-%d %H:%M:%S", gmtime()), message))
    sys.stdout.flush()

def main():
    generator = Generator(input_size=input_size, hidden_size=hidden_size)
    optimizer = tf.keras.optimizers.Adam(0.01)
    inputs = load_data_G(generator)
    train_G(generator, inputs, optimizer)


if __name__ == '__main__':
    main()
