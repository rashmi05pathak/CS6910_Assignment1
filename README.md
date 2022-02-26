# CS6910_Assignment1
This repo is created to solve assignment1 of Deep Learning

Few learning and findings:

This assignment is mainly focused to implement Feedforward Neural Network, Backpropogation, the optimizers from scratch and integrating the code with wandb api in order to run the experiment and track the progress on wandb. We have added the file EE20S051_ED21S007_DL_Assignment1.ipynb which contains colab link, the code can be run from colab and even the results can be checked in this file itself.

We have modularized the code and operations and created several files to answer the questions of assignment. I avoided implementing class to modularize the code here,because security was not our priority here.

We have implemented the FFN, Backprop and optimisers mainly using dictionary datastructure, earlier I was trying to implement these using list, but later I realised disctionary comes more handy while debugging the network.

We have integrated the wandb api codes to log the display of data, accuracy and confusion matrix. We have used sweep config file which also resembles dictionary in our code. sweep will be triggered by the code sweep_id = wandb.sweep(sweep_config,entity='rashmi05pathak',project='assignment1_try2') 
wandb.agent(sweep_id, function=Model_train,count = 5)

Sweep config with bayes search is as below:

sweep_config = {
        'method': 'bayes',
        "name": "assignement1-Sweep2",
        'metric':{
            'goal': 'maximize',
            'name': 'val_accuracy'
        },
        'parameters': {
        'epochs': {
            'values': [5,10,15]
        },
        'no_hidden_layers':{
            'values': [3,4,5]
        },
        'size_hidden_layers':{
            'values': [64,128]
        },
        'learning_rate':{
            'values': [0.001,0.01,0.0001,0.05,0.02]
        },
        'optimizer':{
            'values': ['momentum','sgd','rmsprop','nesterov','adam']
        },
        'batch_size':{
            'values': [32,64,128]
        },
        'activation':{
            'values': ['sigmoid','tanh','Relu']
        },
        'weight_initializations':{
            'values': ['random','xavier']
        },
        'weight_decay':{
            'values': [0,0.0005,0.05]
        }

    }
    }
