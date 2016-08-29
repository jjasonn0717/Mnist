import tensorflow as tf
import numpy as np
import time
import sys
import argparse
import json
import mnist_DNN_pb2
import tensorflow.contrib.learn as skflow
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data


SERVER_TIMEOUT = 10*60

def modelINFO(model_params, out):
	out.write("----------MODEL INFO----------"+'\n')
	out.write(("  Input Size:").ljust(25))
	out.write(str(model_params["in_size"])+'\n')
	out.write(("  Output Size:").ljust(25))
	out.write(str(model_params["out_size"])+'\n')
	out.write(("  Hidden Layers:").ljust(25))
	out.write(str(model_params["layer_num"])+'\n')
	out.write(("  Number of Nodes:").ljust(25))
	out.write(str(model_params["layer_size"])+'\n')
	out.write(("  Learning Rate:").ljust(25))
	out.write(str(model_params["L_rate"])+'\n')
	out.write(("  Dropout Prob:").ljust(25))
	out.write(str(model_params["prob"])+'\n')
	 

class DNN_classifier:
    def __init__(self, in_size=784, out_size=10, layer_num=1, layer_size=[200], L_rate=0.1, drop=0.8):
        ## input ##
        self.in_size = in_size
        self.out_size = out_size
        self.x_ = tf.placeholder(tf.float32, [None, in_size], name="x_")
        self.y_ = tf.placeholder(tf.float32, [None, out_size], name="y_")
        self.prob_ = tf.placeholder(tf.float32, name="prob")
        
        ## result ##
        self.y = None
        
        ## model ##
        self.layer_num = layer_num
        self.layer_size = layer_size
        self.FCLs = []
        self.FCLs_d = []
        self.L_rate = L_rate
        self.drop = drop
                
        ## creat saver ##
        self.saver = None # for model restoring

        ## creat session ##
        self.sess = tf.Session()

    def init_w(self, size, name):
        return tf.Variable(tf.truncated_normal(size, stddev=0.01), name=name)

    def init_b(self, size, name):
        return tf.Variable(tf.zeros(size), name=name)

    def load_model(self, model_path):
        ## build DNN graph ##
        fully_connected = layers.fully_connected(self.x_,
                                                 self.layer_size[0],
                                                 weights_regularizer=layers.l2_regularizer(0.1), 
                                                 biases_regularizer=layers.l2_regularizer(0.1), 
                                                 scope='FCL'+str(1))
        fully_connected_d = layers.dropout(fully_connected, keep_prob=self.prob_)
        self.FCLs.append(fully_connected)
        self.FCLs_d.append(fully_connected_d)
        for layer in xrange(1, self.layer_num):    
            fully_connected = layers.fully_connected(self.FCLs_d[-1],
                                                     self.layer_size[layer],
                                                     weights_regularizer=layers.l2_regularizer(0.1), 
                                                     biases_regularizer=layers.l2_regularizer(0.1), 
                                                     scope='FCL'+str(layer+1))
            fully_connected_d = layers.dropout(fully_connected, keep_prob=self.prob_)
            self.FCLs.append(fully_connected)
            self.FCLs_d.append(fully_connected_d)
        self.y, loss = skflow.models.logistic_regression(self.FCLs[-1], self.y_, init_stddev=0.01)
                
        ## initialize var ##
        self.sess.run(tf.initialize_all_variables())
        ## saver should be created after graph be constructed ##        
        self.saver = tf.train.Saver(max_to_keep=5) 
        self.saver.restore(self.sess, model_path)

    ## calc accuracy ##
    def test(self, x_test, y_test):
        if x_test.shape[1] != self.in_size:
            raise "Incorrect input size!"
        if y_test.shape[1] != self.out_size:
            raise "Incorrect output size!"
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.sess.run(accuracy, feed_dict={self.x_:x_test, self.y_:y_test, self.prob_:1})

    ## predict ##
    def Validate(self, in_data):
        if in_data.shape[1] != self.in_size:
            raise "Incorrect input size!"
        prediction = tf.argmax(self.y, 1) ## return (num,) one dimension array
        result = self.sess.run(prediction, feed_dict={self.x_:in_data, self.prob_:1})
        return result


class Servicer(mnist_DNN_pb2.Betamnist_InferenceServicer):
    def __init__(self, model_params, model_path):
        ## Load DNN Model ##
        self.Model = DNN_classifier(in_size=model_params["in_size"],
                                    out_size=model_params["out_size"],
                                    layer_num=model_params["layer_num"],
                                    layer_size=model_params["layer_size"],
                                    L_rate=model_params["L_rate"],
                                    drop=model_params["prob"])
        self.Model.load_model(model_path)
        ## model testing ##
        print "\n### model testing start  ###"
        Mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
        print "testing acc:", self.Model.test(Mnist_data.test.images, Mnist_data.test.labels)
        print "### model testing finish ###\n"
    
    def GetInput(self, request, context):
        data = np.asarray(request.image)
        if len(data.shape) != 1:
            print "Test data can only be one dimension array!"
            return mnist_DNN_pb2.Digit(digits=[-1])
        elif data.shape[0] != self.Model.in_size:
            print "Wrong data shape!"
            return mnist_DNN_pb2.Digit(digits=[-1])

        data = data.reshape(1, self.Model.in_size)
        predict = self.Model.Validate(data).tolist()
        if request.in_type == "train":
            print "validation type:", request.in_type, "data"
        else:
            print "validation type:", request.in_type, "data"
        reply = mnist_DNN_pb2.Digit(digits=predict)
        return reply

    def GetMultiInput(self, request_iter, context):
        data_batch = []
        for image_array in request_iter:
            data = np.asarray(image_array.image)
            if len(data.shape) != 1:
                print "Test data can only be one dimension array!"
                return mnist_DNN_pb2.Digit(digits=[-1])
            elif data.shape[0] != self.Model.in_size:
                print "Wrong data shape!"
                return mnist_DNN_pb2.Digit(digits=[-1])
            data_batch.append(data)
        data_batch = np.asarray(data_batch)
        assert data_batch.shape[1] == self.Model.in_size
        print "Get", data_batch.shape[0], "test data"
        print "validating..."
        predict_list = self.Model.Validate(data_batch).tolist()
        reply = mnist_DNN_pb2.Digit(digits=predict_list)
        print "...finished"
        return reply


class Mnist_DNN_Server:
    def __init__(self, address, model_params, model_path):
        
        ### Constructing Server ###
        servicer = Servicer(model_params, model_path)
        self.server = mnist_DNN_pb2.beta_create_mnist_Inference_server(servicer)
        self.server.add_insecure_port(address)
        self.addr = address
	
    def server_run(self):
        print "Mnist DNN Server Running at", (self.addr + "...")
        self.server.start()
        try:
            while True:
                time.sleep(SERVER_TIMEOUT)
        except KeyboardInterrupt:
            print "\b\b  \nserver stop..."
            self.server.stop(0)


if __name__ == '__main__':
	
	### Argument Parse ###
	parser = argparse.ArgumentParser(description="mnist DNN Server!")	
	parser.add_argument("struc", help="absolute path for the structure of trained model")
	parser.add_argument("model", help="absolute path for trained model")
    parser.add_argument("-a", "--addr", help="address for the server, default: 0.0.0.0:50051")
	args = parser.parse_args()

	model_file = open(args.struc, 'r')
	model_params = json.loads("".join(model_file.readlines()))
	model_file.close()
    if not model_params['layer_num'] == len(model_params['layer_size']):
        raise "layer_num does not match size of layer_size!!"
	
	outfile = sys.stdout
	modelINFO(model_params, outfile)

    ### Mnist DNN Server ###
    if args.addr == None:
        address = '0.0.0.0:50051'
    else:
        address = args.addr
    mnist_dnn_server = Mnist_DNN_Server(address, model_params, args.model)
    mnist_dnn_server.server_run()

	
