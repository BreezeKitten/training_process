import tensorflow as tf


tf.reset_default_graph()

'''
Parameter
'''
number_of_state = 18
layer1_output_number = 150
layer2_output_number = 100
layer3_output_number = 100
layer4_output_number = 50



def add_layer(inputs, in_size, out_size, W_name, B_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name=W_name)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name=B_name)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, biases

if __name__ == '__main__':

    state = tf.placeholder(tf.float32, [None, number_of_state])
    value = tf.placeholder(tf.float32, [None, 1])
    
    H1, W1, B1 = add_layer(state, number_of_state, layer1_output_number, 'W1', 'B1', activation_function=tf.nn.relu)
    H2, W2, B2 = add_layer(H1, layer1_output_number, layer2_output_number, 'W2', 'B2', activation_function=tf.nn.relu)
    H3, W3, B3 = add_layer(H2, layer2_output_number, layer3_output_number, 'W3', 'B3', activation_function=tf.nn.relu)
    H4, W4, B4 = add_layer(H3, layer3_output_number, layer4_output_number, 'W4', 'B4', activation_function=tf.nn.sigmoid)
    predict_value, Wf, Bf = add_layer(H4, layer4_output_number, 1, 'Wf', 'Bf', activation_function=tf.nn.sigmoid)
    
    cost = tf.losses.mean_squared_error(predict_value, value)
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)
    loss = cost + 0.0001* regularizers
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)

    
