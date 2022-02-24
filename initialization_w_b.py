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

#Defining initialization of weight and bias: random and xavier initialization
def init_random(num_weight_mat,num_nodes):
    weights={}
    bias = {}
    for i in  range(1,num_weight_mat+1):
      weights['W%s'% i] = np.random.randn(int(num_nodes[i]),int(num_nodes[i-1]))
      bias['b%s'% i] = np.random.randn(int(num_nodes[i]))
    
    return weights, bias

def init_xavier(num_weight_mat,num_nodes):
    weights={}
    bias = {}
    for i in  range(1,num_weight_mat+1):
      weights['W%s'%i] = np.random.normal(0, math.sqrt(1/num_nodes[i-1]) ,size=([int(num_nodes[i]),int(num_nodes[i-1])]))
      bias['b%s'%i] = np.zeros(int(num_nodes[i]))
    
    return weights, bias

def init_zeros(num_weight_mat,num_nodes):
    #print(num_weight_mat)
    weights={}
    bias = {}
    for i in  range(1,num_weight_mat+1):
      weights['W%s'% i] = np.zeros([int(num_nodes[i]),int(num_nodes[i-1])])
      bias['b%s'% i] = np.zeros(int(num_nodes[i]))

    return weights,bias
