#forward propagation framework for predicting the output 
def Model_predict(weights, bias, x):
  A = {}
  H = {}    
  H['H0'] = x
  n=len(weights)
  if config['activation'] == 'sigmoid':
    activ = sigmoid
  elif config['activation'] == 'tanh':
    activ = tanh
  elif config['activation'] == 'relu':
    activ = relu
  for i in range(1,n):
    A[('A%s'% i)] = np.matmul(weights['W%s'% i],H[('H%s'% (i-1))])+bias['b%s'% i]
    H['H%s'% i] = activ(A['A%s'% i])
    
  A['A%s'%(n)]= np.matmul(weights['W%s'% (n)],H['H%s'% (n-1)])+bias['b%s'% (n)]
  y_hat = softmax(A['A%s'% (n)])  
    
  return y_hat
