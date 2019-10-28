import tensorflow as tf
import json
import random
import numpy as np
import math

tf.reset_default_graph()
'''
Common parameter
'''
PI = math.pi


'''
DL Parameter
'''
number_of_state = 18
layer1_output_number = 150
layer2_output_number = 100
layer3_output_number = 100
layer4_output_number = 50 
training_eposide_num = 500
training_num = 1000
test_num = 1

'''
Motion Parameter
'''
deltaT = 0.1            #unit:s
V_max = 3               #m/s
W_max = 2               #rad/s
linear_acc_max = 10     #m/s^2
angular_acc_max = 7     #rad/s^2
size_min = 0.1          #unit:m
x_upper_bound = 5       #unit:m
x_lower_bound = -5      #unit:m
y_upper_bound = 5       #unit:m
y_lower_bound = -5      #unit:m
TIME_OUR_FACTOR = 10

RL_epsilon = 0.1
gamma = 0.8

class State:
    def __init__(self, Px, Py, Pth, V, W, r, gx, gy, gth, rank):
        self.Px = Px
        self.Py = Py
        self.Pth = Pth
        self.V = V
        self.W = W
        self.r = r
        self.gx = gx
        self.gy = gy
        self.gth = gth
        self.rank = rank
    def Set_priority(self, m11, m12, m13):
        self.m11 = m11
        self.m12 = m12
        self.m13 = m13
        
def add_layer(inputs, in_size, out_size, W_name, B_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name=W_name)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name=B_name)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, biases
        

def Random_state():
    Px = random.random()*(x_upper_bound - x_lower_bound) + x_lower_bound
    Py = random.random()*(y_upper_bound - y_lower_bound) + y_lower_bound
    Pth = random.random()*2*PI 
    V = (random.random() - 0.5) * V_max
    W = (random.random() - 0.5) * W_max
    r = random.random() + size_min
    gx = random.random()*(x_upper_bound - x_lower_bound) + x_lower_bound
    gy = random.random()*(y_upper_bound - y_lower_bound) + y_lower_bound
    gth = random.random()*2*PI 
    rank = random.randint(1,3)    
    
    agent = State(Px, Py, Pth, V, W, r, gx, gy, gth, rank)
    return agent


def Calculate_distance(x1, y1, x2, y2):
    return np.sqrt(math.pow( (x1-x2) , 2) + math.pow( (y1-y2) , 2))

def angle_correct(angle):
    angle = math.fmod(angle, 2*PI)
    if angle < 0:
        angle = angle + 2*PI
    return angle

def Motion_model(Px, Py, Pth, V, W):
    TH = Pth + W * deltaT
    TH = angle_correct(TH)    
    X = Px + V * deltaT * math.cos((Pth+TH)/2)
    Y = Py + V * deltaT * math.sin((Pth+TH)/2)
    
    return X, Y, TH

def Check_collussion(agent1, agent2):
    distance = Calculate_distance(agent1.Px, agent1.Py, agent2.Px, agent2.Py)
    if (distance <= (agent1.r + agent2.r)):
        return True
    else:
        return False
    
    
def Check_Goal(agent, position_tolerance, orientation_tolerance):
    position_error = Calculate_distance(agent.Px, agent.Py, agent.gx, agent.gy)
    orientation_error = abs(agent.Pth - agent.gth)
    if (position_error < position_tolerance) and (orientation_error < orientation_tolerance):
        return True
    else:
        return False
 
def Calculate_value(Path, reward, reward_time):
    for item in Path:
        remain_time_step = (reward_time - Path[item]['time_tag'])/deltaT
        Path[item]['Value'] = reward * math.pow(gamma, remain_time_step)
    return Path

def Observe_state(agent):
    

def Predict_action_value(agent1, agent2, V_pred, W_pred):
    X_pred,  Y_pred, TH_pred = Motion_model(agent1.Px, agent1.Py, agent1.Pth, V_pred, W_pred)
    Px2, Py2, Vx2, Vy2, r2 = Observe_state(agent2)
    state_pred = [[X_pred, Y_pred, TH_pred, V_pred, W_pred, agent1.gx, agent1.gy, agent1.gth, V_max, agent1.m11, agent1.m12, agent1.m13, Px2, Py2, Vx2, Vy2, r2]]
    value_matrix = sess.run(predict_value, feed_dict={state: state_pred})
    action_value = value_matrix[0][0]

    return action_value    
    
def Choose_action(agent1, agent2, epsilon):
    dice = random.random()
    if dice < epsilon:
        linear_acc = -linear_acc_max + random.random() * 2 * linear_acc_max
        angular_acc = -angular_acc_max + random.random() * 2 * angular_acc_max
        V_pred = np.clip(agent1.V + linear_acc * deltaT, -V_max, V_max)
        W_pred = np.clip(agent1.W + angular_acc * deltaT, -W_max, W_max)
        return V_pred, W_pred
    else:
        linear_acc_set = np.arange(-linear_acc_max, linear_acc_max, 1)
        angular_acc_set = np.arange(-angular_acc_max, angular_acc_max, 1)
        for linear_acc in linear_acc_set:
            V_pred = np.clip(agent1.V + linear_acc * deltaT, -V_max, V_max)
            for angular_acc in angular_acc_set:
                W_pred = np.clip(agent1.W + angular_acc * deltaT, -W_max, W_max)
                action_value = Predict_action_value(agent1, agent2, V_pred, W_pred)
            
            
    
    

def Read_data(file_name):
    data = {}
    file = open(file_name,'r')
    data_line = file.readline()
    count = 0
    while(data_line):
        data[count] = json.loads(data_line)
        data_line = file.readline()
        count = count + 1
    file.close()
    return data

def Record_data(data, file_name):
    file = open(file_name, 'a+')
    for item in data:
        json.dump(data[item],file)
        file.writelines('\n')   
    file.close()
       
       
def Sample_data(data_base, sample_number):
    sampled_data = {}
    sample_array = random.sample(range(0,len(data_base)), sample_number)
    for index in sample_array:
        sampled_data[index] = data_base[index]
    return sampled_data

   

def Divide_state_value(data):
    Start_flag = 1
    for item in data:
        temp_state = [[data[item]['Px'],data[item]['Py'],data[item]['Pth'],data[item]['V'],data[item]['W'],data[item]['r1'],data[item]['gx'],data[item]['gy'],data[item]['gth'],data[item]['Vmax'],data[item]['m11'],data[item]['m12'],data[item]['m13'],data[item]['Px2'],data[item]['Py2'],data[item]['Vx2'],data[item]['Vy2'],data[item]['r2']]]
        temp_value = [[data[item]['Value']]]
        if Start_flag:
            state = temp_state
            value = temp_value
            Start_flag = 0
        else:
            state = np.concatenate((state, temp_state), axis=0)
            value = np.concatenate((value, temp_value), axis=0)
    return state, value

def DL_process():
    data = Read_data('record.json')
    #test_data = Sample_data(data, test_num)
    #test_state, test_value = Divide_state_value(test_data)
    #test_predict = []
    
    for training_eposide in range(training_eposide_num):
        training_data = Sample_data(data, training_num)
        training_state, training_value = Divide_state_value(training_data)
        sess.run(train_step, feed_dict={state: training_state, value: training_value})
        #test_predict.append(sess.run(predict_value, feed_dict={state: test_state}))
        if training_eposide%10 == 0:
            rs = sess.run(loss_record, feed_dict = {state: training_state, value: training_value})
            writer.add_summary(rs, training_eposide)
            print('record', training_eposide)
        #print('eposide: ',training_eposide, 'test error: ', test_value-test_predict[-1][0][0])
    

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
    
    loss_record = tf.summary.scalar('loss',loss)
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', sess.graph)
    
    
    init = tf.global_variables_initializer()
    sess.run(init)       
    
    saver.restore(sess,'test/test.ckpt')
    
    DL_process()
    
    print('Finish')
