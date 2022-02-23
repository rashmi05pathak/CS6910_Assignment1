# getting one sample image from each class and visualizing 
fig = pyplot.figure(figsize=(10, 7))
# setting values to rows and column variables
rows = 2
columns = 5
for i in range(10):
	# Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, i+1)
    j = list_of_index.get(i)
    # showing image
    pyplot.imshow(trainX[j])
    pyplot.axis('off')
    pyplot.title("class"+ str(i))
# show the figure
pyplot.show()
