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
    weights={}
    bias = {}
    for i in  range(1,num_weight_mat+1):
      weights['W%s'% i] = np.zeros([int(num_nodes[i]),int(num_nodes[i-1])])
      bias['b%s'% i] = np.zeros(int(num_nodes[i]))

    return weights,bias
