def Model_train():            #define inputs
  
  X = x_train[:50000]
  Y = y_train[:50000]

  x_val = x_train[50000:]
  y_val = y_train[50000:]
  

  #get inputs from config
  eta = config['eta']
  epochs = config['epochs']
  batch_size=config['batch_size']
  activation=config['activation']
  optimization=config['optimizer']
  num_hidden_layers = config['hidden_layers'] 
  size=config['size']        
  wt_decay=config['wt_decay']   
  initialization=config['wt_init']
  input_layer_size = 784  #input number of inputs here

  output_layer_size = 10  #input number of output nodes here

  num_nodes = [input_layer_size]

  for i in range(num_hidden_layers):
    num_nodes.append(size)
  num_nodes.append(output_layer_size)

  num_weight_mat = num_hidden_layers+1

  
  #Setting weight initialization function
  if initialization == 'random':
    wt_init=init_random(num_weight_mat,num_nodes)
  elif initialization == 'xavier':
    wt_init = init_xavier(num_weight_mat,num_nodes)
  else:
    print('weight initialization given is not available')

  #Setting weights
  weights = {}
  bias = {}
  weights, bias = wt_init


  #Setting activation function and its derivative
  if activation == 'sigmoid':
    activ = sigmoid
    der_activ = grad_sigmoid
  elif activation == 'tanh':
    activ = tanh
    der_activ = grad_tanh
  elif activation == 'relu':
    activ = relu
    der_activ = grad_relu

  #Setting output function
  output = softmax 
  activation = activ
  
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
  
  
   #defining function for sgd optimizer
  def sgd():

    for k in range(epochs):
      for x,y in zip(X,Y):
        grad_w,grad_b = backward_propogation(weights, bias, x,y)
        for j in range(1,num_hidden_layers+2):
          
          weights['W%s'%j]-=(grad_w['W%s'%j])*eta
          bias['b%s'%j]-=(grad_b['b%s'%j])*eta

   #defining function for momentum gd optimizer
  def momentum_gd():

    gamma=0.9
    prev_dw,prev_db=init_zeros(num_weight_mat,num_nodes)

    for k in range(epochs):
      count=0   
    
      dw,db=init_zeros(num_weight_mat,num_nodes)
      
      for x,y in zip(X,Y):
        
        if count==0:
          dw,db=init_zeros(num_weight_mat,num_nodes)
        
        grad_w,grad_b = backward_propogation(weights, bias, x, y)

        for j in range(1,num_hidden_layers+2):
          dw['W%s'%j]+=(grad_w['W%s'%j])
          db['b%s'%j]+=(grad_b['b%s'%j])



        count+=1

        if count%batch_size==0:
          
          weights['W%s'%j]-=((prev_dw['W%s'%j])*gamma+(dw['W%s'%j])*eta+(weights['W%s'%j])*wt_decay*eta)
          bias['b%s'%j]-=((prev_db['b%s'%j])*gamma+(db['b%s'%j])*eta+(bias['b%s'%j])*wt_decay*eta)

          prev_dw['W%s'%j]=(prev_dw['W%s'%j])*gamma+(dw['W%s'%j])*eta
          prev_db['b%s'%j]=(prev_db['b%s'%j])*gamma+(db['b%s'%j])*eta

          count=0

   #defining function for NAG optimizer
  def nesterov():

    gamma=0.9
    prev_vw,prev_vb=init_zeros(num_weight_mat,num_nodes)
    temp_w,temp_b=init_zeros(num_weight_mat,num_nodes)
    dw,db=init_zeros(num_weight_mat,num_nodes)
    v_w,v_b=init_zeros(num_weight_mat,num_nodes)

    for k in range(epochs):
      count=0

      if count==0:
        for j in range(1,num_hidden_layers+2):
            v_w['W%s'%j]=prev_vw['W%s'%j]*gamma 
            v_b['b%s'%j]=prev_vb['b%s'%j]*gamma 


      for x,y in zip(X,Y):
        if count==0:
          dw,db=init_zeros(num_weight_mat,num_nodes)

        for j in range(1,num_hidden_layers+2):
          temp_w['W%s'%j]=weights['W%s'%j]-v_w['W%s'%j]
          temp_b['b%s'%j]=bias['b%s'%j]-v_b['b%s'%j]
        
        grad_w,grad_b=backward_propogation(temp_w,temp_b,x,y)      
        for j in range(1,num_hidden_layers+2):
          dw['W%s'%j]+=(grad_w['W%s'%j])
          db['b%s'%j]+=(grad_b['b%s'%j])
        count+=1

        if count%batch_size==0:
          
          for j in range(1,num_hidden_layers+2):
            v_w['W%s'%j]=(prev_vw['W%s'%j])*gamma + (dw['W%s'%j])*eta
            v_b['b%s'%j]=(prev_vb['b%s'%j])*gamma + (db['b%s'%j])*eta
            weights['W%s'%j]-=(v_w['W%s'%j]+(weights['W%s'%j])*wt_decay*eta)
            bias['b%s'%j]-=(v_b['b%s'%j]+(bias['b%s'%j])*wt_decay*eta)
            prev_vw['W%s'%j]=v_w['W%s'%j]
            prev_vb['b%s'%j]=v_b['b%s'%j]

          count=0

   #defining function for adagrad optimizer
  def adagrad():
  
    eps = 1e-8

    v_w,v_b=init_zeros(num_weight_mat,num_nodes)

    for k in range(epochs):
      count=0
      dw,db=init_zeros(num_weight_mat,num_nodes)
      
      for x,y in zip(X,Y):

        if count==0:
          dw,db=init_zeros(num_weight_mat,num_nodes)
        
        grad_w,grad_b=backward_propogation( weights,bias,x,y)
        for j in range(1,num_hidden_layers+2):
          dw['W%s'%j]+=(grad_w['W%s'%j])
          db['b%s'%j]+=(grad_b['b%s'%j])
        count+=1

        if count%batch_size==0:

          for j in range(1,num_hidden_layers+2):
            v_w['W%s'%j] += np.square(dw['W%s'%j])
            v_b['b%s'%j] += np.square(db['b%s'%j]) 
            weights['W%s'%j] -= ((dw['W%s'%j])*(eta/np.sqrt(v_w['W%s'%j] + eps))+(weights['W%s'%j])*wt_decay*eta)
            bias['b%s'%j] -= ((db['b%s'%j])*(eta/np.sqrt(v_b['b%s'%j] + eps))+(bias['b%s'%j])*wt_decay*eta)

            count=0
  #defining function for rmsprop optimizer
  def rmsprop():
    eps = 1e-8
    beta1 = 0.9

    v_w,v_b=init_zeros(num_weight_mat,num_nodes)

    for k in range(epochs):
      count=0
      dw,db=init_zeros(num_weight_mat,num_nodes)
      
      for x,y in zip(X,Y):

        if count==0:
          dw,db=init_zeros(num_weight_mat,num_nodes)
        
        grad_w,grad_b= backward_propogation( weights,bias,x,y)
        for j in range(1,num_hidden_layers+2):
          dw['W%s'%j]+=(grad_w['W%s'%j])
          db['b%s'%j]+=(grad_b['b%s'%j])
        count+=1

        if count%batch_size==0:

          for j in range(1,num_hidden_layers+2):
            v_w['W%s'%j] = beta1*v_w['W%s'%j] + (1-beta1)*np.square(dw['W%s'%j])
            v_b['b%s'%j] = beta1*v_b['b%s'%j] + (1-beta1)*np.square(db['b%s'%j]) 
            weights['W%s'%j] -= ((dw['W%s'%j])*(eta/np.sqrt(v_w['W%s'%j] + eps))+(weights['W%s'%j])*wt_decay*eta)
            bias['b%s'%j] -= ((db['b%s'%j])*(eta/np.sqrt(v_b['b%s'%j] + eps))+(bias['b%s'%j])*wt_decay*eta)

            count=0

  
   #defining function for adam optimizer
  def adam():
  
    eps = 1e-8
    beta1 = 0.9
    beta2 = 0.999

    v_w,v_b=init_zeros(num_weight_mat,num_nodes)
    v_w_hat,v_b_hat=init_zeros(num_weight_mat,num_nodes)

    m_w,m_b=init_zeros(num_weight_mat,num_nodes)
    m_w_hat,m_b_hat=init_zeros(num_weight_mat,num_nodes)

    for k in range(epochs):
      
      count=0
      dw,db=init_zeros(num_weight_mat,num_nodes)
      
      for x,y in zip(X,Y):

        if count==0:
          dw,db=init_zeros(num_weight_mat,num_nodes)
        
        grad_w,grad_b = backward_propogation( weights,bias,x,y)
        for j in range(1,num_hidden_layers+2):
          dw['W%s'%j]+=(grad_w['W%s'%j])
          db['b%s'%j]+=(grad_b['b%s'%j])
        count+=1

        if count%batch_size==0:

          for j in range(1,num_hidden_layers+2):
            
            v_w['W%s'%j] = beta2*v_w['W%s'%j] + (1-beta2)*np.square(dw['W%s'%j])
            v_b['b%s'%j] = beta2*v_b['b%s'%j] + (1-beta2)*np.square(db['b%s'%j]) 

            v_w_hat['W%s'%j] = v_w['W%s'%j]*1/(1-math.pow(beta2,k+1))
            v_b_hat['b%s'%j] = v_b['b%s'%j]*1/(1-math.pow(beta2,k+1))

            m_w['W%s'%j] = beta1*m_w['W%s'%j] + (1-beta1)*(dw['W%s'%j])
            m_b['b%s'%j] = beta1*m_b['b%s'%j] + (1-beta1)*(db['b%s'%j]) 

            m_w_hat['W%s'%j] = m_w['W%s'%j]*1/(1-math.pow(beta1,k+1))
            m_b_hat['b%s'%j] = m_b['b%s'%j]*1/(1-math.pow(beta1,k+1))
            
            weights['W%s'%j] -= ((m_w_hat['W%s'%j])*(eta/np.sqrt(v_w_hat['W%s'%j] + eps))+(weights['W%s'%j])*wt_decay*eta) 
            bias['b%s'%j] -= ((m_b_hat['b%s'%j])*(eta/np.sqrt(v_b_hat['b%s'%j] + eps))+(bias['b%s'%j])*wt_decay*eta)

            count=0

  #Setting optimization function
  if optimization == 'sgd':
    sgd()
  elif optimization == 'momentum':
    momentum_gd()
  elif optimization == 'nesterov':
    nesterov()
  elif optimization == 'rmsprop':
    rmsprop()
  elif optimization =='adam':
    adam()
  elif optimization =='adagrad':
    adagrad()  
  else:
    print('Optimization algo given is not available')
  #function for errors and validation accuracy 

  def cross_entropy(p, q):
      return -np.log(np.dot(p, q))

  def val_accuracy(weights,bias,x_val,y_val):
     y_pred=np.zeros(10000)
     fx=np.zeros([10000,10])
     val_loss=0
     for i in range(10000):
        fx[i]= forward_propogation(weights,bias,x_val[i])[0]
        val_loss+= cross_entropy(y_val[i],fx[i])
     val_loss/=10000
     y_pred=np.argmax(fx,axis=1)   
     y_val_=np.argmax(y_val,axis=1)

     validation_acc = accuracy_score(y_pred,y_val_)
     return val_loss,validation_acc

  val_loss,val_accuracy = val_accuracy(weights,bias,x_val,y_val)
  print(val_loss) #to know how the model is performing on validation set

  params={'epochs':epochs,'learning rate':eta,'no of hidden layers':num_hidden_layers,'layer size':size,'batch size':batch_size,'optimizer':optimization,'val loss':val_loss,'validation accuracy':val_accuracy}

  return weights,bias
