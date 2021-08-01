import numpy as np

def cross_entropy(ytrue, yhat, eps=1e-8):

    return -(ytrue * np.log(yhat + eps)).sum(axis=1).mean()

def accuracy(ytrue, ypred):

    ytrue = ytrue.argmax(axis=1)
    ypred = ypred.argmax(axis=1)
    acc = (ytrue == ypred).mean()

    return acc

class LinearClassifier(object):

    def __init__(self, input_dim, output_dim, lr):

        """
        W: (output_dim, input_dim) weight matrix
        b: (output_dim, 1) bias vector
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.W = np.random.normal(size=(input_dim, output_dim))
        self.b = np.zeros(shape=(1, output_dim))

    def predict(self, X):

        logit  = X.dot(self.W) + self.b  # (batch_size, input_dim) X (input_dim, output_dim) = (batch_size, output_dim)
        logit -= logit.max(1, keepdims=True)
        output = np.exp(logit)
        output /= np.sum(output, axis=1, keepdims=True)

        return output

    def train_step(self, X, y):

        """
        X: (n, p) features
        y: (n, 1) labels
        """

        yhat = self.predict(X)
        grad = -y * (1 - yhat)

        grad_W = X.T.dot(grad)
        grad_b = grad.sum(axis=0, keepdims=True)

        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

    def train(self, X, y, batch_size, epoch, validset=None):

        for i in range(epoch):

            Xgen = (X[i:i+batch_size] for i in range(0, X.shape[0], batch_size))
            ygen = (y[i:i+batch_size] for i in range(0, y.shape[0], batch_size))
            
            for Xbatch, ybatch in zip(Xgen, ygen):
                self.train_step(Xbatch, ybatch)

            train_ypred = model.predict(X)
            train_loss = cross_entropy(y, train_ypred)
            train_acc = accuracy(y, train_ypred)

            if validset:

                Xval, yval = validset
                valid_ypred = model.predict(Xval)
                valid_loss = cross_entropy(yval, valid_ypred)
                valid_acc = accuracy(yval, valid_ypred)

                print(f"epoch: {i:02d} batch train_loss: {train_loss:7.4f} train_acc: {train_acc:7.4f} valid_loss: {valid_loss:7.4f} valid_acc: {valid_acc:7.4f}")

            else:

                print(f"epoch: {i:02d} batch train_loss: {train_loss:7.4f} train_acc: {train_acc:7.4f}")

if __name__ == "__main__":

    import tensorflow as tf
    import matplotlib.pyplot as plt

    (Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()
    Xtrain = Xtrain.astype(np.float64) / 255.
    Xtest = Xtest.astype(np.float64) / 255.

    train_size, *input_shape = Xtrain.shape
    test_size = Xtest.shape[0]
    input_dim = np.prod(input_shape)

    # flatten
    Xtrain = Xtrain.reshape(train_size, -1)
    Xtest = Xtest.reshape(test_size, -1)

    output_dim = np.unique(ytrain).size
    ytrain = np.eye(output_dim)[ytrain.flatten()]
    ytest = np.eye(output_dim)[ytest.flatten()]

    idx = np.arange(train_size)
    idx = np.random.shuffle(idx)
    valid_size = 10000
    
    Xtrain, ytrain = Xtrain[:-valid_size], ytrain[:-valid_size]
    Xvalid, yvalid = Xtrain[-valid_size:], ytrain[-valid_size:]

    # train model
    model = LinearClassifier(input_dim=input_dim, output_dim=output_dim, lr=1e-3)
    model.train(Xtrain, ytrain, batch_size=32, epoch=10, validset=(Xvalid, yvalid))

    # visualize feature map for each class
    W = model.W.reshape(*input_shape, output_dim)
    W = W.transpose(-1, *range(len(W.shape)-1))

    fig, axes = plt.subplots(2, 5, figsize=(30, 7))

    for feature_map, ax in zip(W, axes.flatten()):

        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        ax.imshow(feature_map)

    plt.suptitle("Feature map for each class")
    plt.show();
