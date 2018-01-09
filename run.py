import load_mnist
import network

train,cv,test = load_mnist.load_data_wrapper()

##use  network
# net = network.Network([784, 40, 10])
# net.SGD(train,30,10,3.0,test)


#use network2  v1  use crossentropy as cost function
import network2
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(train, 30, 10, 0.5, evaluation_data=test,monitor_evaluation_accuracy=True)

#use network2  v2 overfit
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(train[:1000], 400, 10, 0.5, evaluation_data=test,monitor_evaluation_accuracy=True, monitor_training_cost=True)

#use network2  v3  cross val
net.large_weight_initializer()
net.SGD(train, 30, 10, 0.1,evaluation_data=test, lmbda = 5.0,monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,monitor_training_cost=True, monitor_training_accuracy=True)


