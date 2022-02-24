# getting one sample image from each class and visualizing 
import wandb

(trainx, trainy), (testx, testy)= fashion_mnist.load_data()
classes = np.unique(trainy)
index_mat = [0]*10
for i in classes:
  index_mat[i] = trainy.tolist().index(i)

plt.figure(figsize=(10,10))

for j, i in enumerate(index_mat):
  plt.subplot(3,4,j+1)
  plt.imshow(trainx[i], cmap=plt.get_cmap('gray'))
  
plt.show()

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
wandb.init(entity='kunal_patil',project='Assignment 1')
wandb.log({'sample':[wandb.Image(trainx[i],caption=class_names[trainy[i]]) for i in index_mat]})
