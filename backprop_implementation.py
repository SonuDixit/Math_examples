'''
implement backprop in numpy
consider mnist dataset
2 layer network - input, 30, relu, 10, softmax 
'''

import numpy as np
from datasets import load_dataset

def softmax(x):
    '''
    x: Bxd
    '''
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    '''
    x: Bxd
    '''
    return np.maximum(0, x)

def grad_relu(x):
    '''
    x: Bxd
    '''
    return np.where(x > 0, 1, 0)

def cross_entropy_loss(y, y_hat):
    '''
    y: Bxd
    y_hat: Bxd
    '''
    return -np.sum(y * np.log(y_hat+1e-8))

class Network:
    def __init__(self, 
                lr=0.01, 
                init_dim=784, 
                n_class=10,
                d1=30):
        '''
        initialize weights and biases
        weights - uniform random initialization (-1,1)
        bias - zeros
        '''
        d1 = 30
        d2 = n_class
        self.W1 = np.random.uniform(-1,1, (init_dim, d1))
        self.b1 = np.zeros(d1)
        self.W2 = np.random.uniform(-1,1, (d1, d2)) # num class
        self.b2 = np.zeros(d2)
        self.lr = lr
    
    def forward(self, 
                x, 
                y=None, 
                backprop=True):
        '''
        x: Bxd
        y: Bxc
        '''
        z1 = np.dot(x, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = softmax(z2)
        loss = None
        gradient = {}
        if y is not None:
            loss = cross_entropy_loss(y=y, y_hat=a2)
            # back prop here
        if loss and backprop:
            grad_z2 = a2 - y # Bxd2 - Bxd2 = Bxd2
            grad_W2 = np.dot(a1.T, grad_z2) # d1xB * Bxd2 = d1xd2
            grad_b2 = np.sum(grad_z2, axis=0) # d2
            grad_a1 = np.dot(grad_z2, self.W2.T) # Bxd2 * d2xd1 = Bxd1
            grad_z1 = grad_a1 * grad_relu(a1) # Bxd1 * Bxd1 = Bxd1
            grad_W1 = np.dot(x.T, grad_z1) # dxB * Bxd1 = dxd1
            grad_b1 = np.sum(grad_z1, axis=0) # d1
            gradient = {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 'b2': grad_b2}

            # update weights
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

        return loss, gradient
    
    def predict(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = softmax(z2)
        return a2

def main():
    # load mnist data
    # preprocess data
    # train test split
    # create network
    # train
    # test

    # ds = load_dataset("ylecun/mnist")
    # randomly generate data for now, X from normal distribution, Y from multinomial distribution
    np.random.seed(0)
    X = np.random.standard_normal((100, 784))
    Y = np.random.multinomial(1, [0.1]*10, size=100)
    # model = Network(lr=0.01)
    # for i in range(100):
    #     loss, grad = model.forward(x=X, y=Y)
    #     print(f"step:{i+1}, Loss: {loss}")
        # print(f"step:{i+1}, Grad: {grad}")

    # load mnist data
    ds = load_dataset("ylecun/mnist")
    train_data = ds['train']
    test_data = ds['test']
    train_data = train_data.take(100)
    def preprocess(x):
        x_ = x['image']
        x_ = np.asarray(x_)
        x_ = x_.flatten()
        x_ = x_/255
        y_ = x['label']
        y_ = np.eye(10)[y_]
        return {'image':x_, 'label':y_}
    
    train_data = train_data.map(preprocess)
    print(train_data)
    X = np.array(train_data['image']) # Bx784x1
    print(X.shape)
    X = X.squeeze(-1)
    Y = np.array(train_data['label'])
    model = Network(lr=0.005)
    for i in range(200):
        loss, grad = model.forward(x=X, y=Y)
        print(f"step:{i+1}, Loss: {loss}")
        # print(f"step:{i+1}, Grad: {grad}")
    y_hat = model.predict(X)
    accuracy = np.mean(np.argmax(y_hat, axis=1) == np.argmax(Y, axis=1))    
    print(f"Accuracy: {accuracy}")        


if __name__ == "__main__":
    main()