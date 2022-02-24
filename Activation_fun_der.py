#Defining activation functions and their derivatives

def sigmoid(x):
  return 1/(1+np.exp(-x))

def tanh(x):
  return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
  t=np.array((np.maximum(0,x)),dtype=np.longdouble)
  return t

def grad_tanh(x):
  return 1-(tanh(x))**2

def grad_relu(x):
  t = np.where(x < 0, 0, x)
  return t

def grad_sigmoid(x):
  return (sigmoid(x)*(1-sigmoid(x)))

def softmax(x):
  exps = np.exp(x )
  return exps / np.sum(exps)
