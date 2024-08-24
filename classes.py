import torch


class LinearRegression:
    def __init__(self, n_features, lr):
        self.W = torch.randn(n_features, dtype=torch.double, requires_grad=True)
        self.b = torch.randn(1, dtype=torch.double, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.W) + self.b

    def loss(self, y_pred, y):
        return ((y_pred - y) ** 2).mean()

    def configure_optim(self, lr):
        return GD([self.W, self.b], lr)

    # BGD
    def fit(self, X, y, epochs, lr):
        optim = self.configure_optim(lr)
        batch_size = 10
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self.forward(X_batch)
                loss = self.loss(y_pred, y_batch)
                loss.backward()
                optim.step()
                optim.zero_grad()
                print(f'Epoch: {epoch}, Loss: {loss.item()}')


class GD():
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = None