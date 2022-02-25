#Setting activation function and its derivative
activation_fun = config['activation']
if activation_fun == 'sigmoid':
    activ = sigmoid
    der_activ = grad_sigmoid
elif activation_fun == 'tanh':
    activ = tanh
    der_activ = grad_tanh
elif activation_fun == 'relu':
    activ = relu
    der_activ = grad_relu

#Setting output function
output = softmax 
activation = activ
  

#Forward propogation
def forward_propogation(weights, bias, x):
    num_weight_mat = len(weights)
    num_hidden_layers = num_weight_mat-1
    Acti = {}
    H = {}
    
    H['H0'] = x
    
    for i in range(1,num_hidden_layers+1):
      Acti[('A%s'% i)] = np.matmul(weights['W%s'% i],H[('H%s'% (i-1))])+bias['b%s'% i]
      H['H%s'% i] = activation(Acti['A%s'% i])
    
    Acti['A%s'%(num_hidden_layers+1)]= np.matmul(weights['W%s'% (num_hidden_layers+1)],H['H%s'% (num_hidden_layers)])+bias['b%s'% (num_hidden_layers+1)]
    y_hat = output(Acti['A%s'% (num_hidden_layers+1)])  
    
    return y_hat, Acti, H
 
