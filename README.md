# VFPS-SM
VFPS-SM is a framework for Vertical Federated Participant Selection via Submodular Maximization, which chooses KNN classifier as the proxy model and enables the data consortium  to select more informative participants, protect data privacy via homomorphic encryption.

## Requirements
Our code is based on Python version 3.8 and PyTorch version 2.0.1. Please make sure you have installed Python and PyTorch correctly. 

Please refer to the source code to install the required packages that have not been installed in your environment such as following:
* `scikit-learn==1.3.2`
* `tenseal==0.3.14`
* `grpcio==1.59.0`
* `grpcio-tools==1.59.0`

You can install these packages in a shell as:
```
pip install grpcio
pip install grpcio-tools
```

## Running Procedures
Before you start running the main script, it is essential to use the Protocol Buffers (protobuf) compiler to generate Python code for gRPC based on a specified protobuf file. 
```
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./tenseal_allreduce_data.proto
```

Within `/conf/args.py`, you'll find various parameter definitions. Update the values according to your experiment requirements.


Run the command to start the gRPC server program.
```
python ../vfl_diversity_selection/transmission/tenseal_shapley/tenseal_all_reduce_server.py
```

Once the server is running, you can proceed with executing the client script. And you can choose the dataset which you need.
```
python ../vfl_diversity_selection/tenseal_script/knn_diversity/diversity_knn_fagin.py
```