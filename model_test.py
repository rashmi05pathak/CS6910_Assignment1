#following block of code to get test set error and accuracy
W,b = Model_train()

# function to return test error and accuracy
def test_model():
  f_x=np.zeros([10000,10])
  for i in range(10000):
    f_x[i],A,H=forward_propogation(W,b,x_test[i]) #Ingnore A and H values, here we need only f_x value
  test_pred=np.argmax(f_x,axis=1)
  test_acc=accuracy_score(test_pred,y_test)
  return test_acc,test_pred

accuracy,pred = test_model()
print(accuracy)

#Confusion Matrix
test_acc,test_pred=test_model()
ytest_pred=[class_names[k] for k in test_pred]
ytest=[class_names[k]  for k in y_test]
confusion_mat=skplt.metrics.plot_confusion_matrix(ytest,ytest_pred,figsize=(10,10))

