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
  

  #Backward propogation
def backward_propogation(w, b, x, y):
    grad_A = {}
    grad_W = {}
    grad_b = {}
    grad_H = {}  

    num_weight_mat = len(w)
    num_hidden_layers = num_weight_mat-1

    y_hat, A, H = forward_propogation(w, b, x)

    grad_A['A%s'% (num_hidden_layers+1)] = -(y-y_hat)  #cross entropy loss
    
    for i in range(num_hidden_layers+1, 1, -1):
      
      grad_W['W%s'%i] = np.matmul(grad_A['A%s'%i].reshape(-1,1), H['H%s'%(i-1)].reshape(1,-1))

      grad_b['b%s'%i] = grad_A['A%s'%i]

      grad_H['H%s'%(i-1)] = np.dot(np.transpose(w['W%s'%i]), grad_A['A%s'%i]) 

      grad_A['A%s'%(i-1)] = np.multiply(grad_H['H%s'%(i-1)], der_activ(A['A%s'%(i-1)]))

    grad_W['W1'] = np.multiply(grad_A['A1'].reshape(-1,1), H['H0'].reshape(1,-1))

    grad_b['b1'] = grad_A['A1']

    return grad_W, grad_b
