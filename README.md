###DNN Training Usage
1. launch mnist DNN trainer by ```python mnist_DNN_train.py MODEL_STRUC ITER BATCH [-d DIR]```
2. also see ```python mnist_DNN_train.py -h```

###Server-Client Usage
1. execute ```./run_codegen.sh``` to produce the file mnist_DNN_pb2.py
2. execute server by ```python mnist_serving.py TRAINED_MODEL```, also see ```python mnist_serving.py -h``` for more help
3. launch client by ```python mnist_client.py IMAGE [-l LABEL]```, also see ```python mnist_client.py -h``` for more help

###Dependency
- [TensorFlow](https://www.tensorflow.org/)
- [gRPC](http://www.grpc.io/)
- [Protocol Buffers(proto3)](https://developers.google.com/protocol-buffers/)
