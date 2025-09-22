import numpy as np
import os

# 加载数据集
def load_npy_dataset_uniform(root_dir, samples_per_class=500):
    X_list, y_list = [], []
    labels = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]) # 标签名构成的列表
    # os.listdir返回一个包含输入目录中所有条目（文件、文件夹）名称的列表，条目名称为字符串类型
    # os.path.isdir用于判断给定的路径是否为一个存在的文件夹，返回True或False
    # os.path.join输入一个或或多个字符串参数，输出一个拼接后的路径字符串

    for label_str in labels:
        label_dir = os.path.join(root_dir, label_str)
        files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')]) # 单个标签下的样本名字符串构成的列表
        for f in files[:samples_per_class]: # 如果某类样本不足，取全部
            path = os.path.join(label_dir, f) # 单个样本的路径
            img = np.load(path).reshape(-1) # np.load输入路径，输出numpy数组，.reshape(-1)作用是把数组拉成一维向量
            X_list.append(img) # 所有样本numpy数组构成的列表
            y_list.append(int(label_str))

    X = np.array(X_list, dtype=np.float32) / 255.0 # 除以255，把数据归一化到0~1
    y = np.array(y_list, dtype=np.int64)
    return X, y

# 激活函数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    # 防止溢出：减去每行最大值
    exps = np.exp(x - np.max(x, axis=1, keepdims=True)) # x 是一个二维数组，axis=0 是列方向（垂直），axis=1 是行方向（水平）
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    # y_pred: (n_samples, n_classes), 概率分布
    # y_true: (n_samples,), 每个元素是 0 ~ n_classes-1
    n_samples = y_pred.shape[0]
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps) # 从预测中取出真实标签对应的概率（避免 log(0) 加 epsilon）
    correct_probs = y_pred_clipped[np.arange(n_samples), y_true] #输出的是一个一维数组，长度为 n_samples
    loss = -np.mean(np.log(correct_probs))
    return loss

def one_hot(y, num_classes):  # 求D2备用，正确类别从向量转化为矩阵形式
    onehot = np.zeros((y.size, num_classes)) # y.size是样本量，num_classes是类别数
    onehot[np.arange(y.size), y] = 1
    return onehot

def accuracy(y_pred_probs, y_true):
    y_pred = np.argmax(y_pred_probs, axis=1) # 输出每行最大值的列索引，输出一维数组，维数等于输入数组行数
    return np.mean(y_pred == y_true)

class TwoLayerNN:
    def __init__(self, n_input, n_hidden, n_output, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(n_input, n_hidden) * 0.1
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * 0.1
        self.b2 = np.zeros((1, n_output))

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2, self.A1, self.Z1  # 中间变量供反向传播用

class Backward:
    def __init__(self, model):
        self.model = model

    def run(self, y_true):
        X = self.model.X
        A2 = self.model.A2
        A1 = self.model.A1
        Z1 = self.model.Z1
        n_samples = X.shape[0]
        n_outputs = A2.shape[1]
        y_onehot = one_hot(y_true, n_outputs)

        D3 = (A2 - y_onehot) / n_samples  # 每行是DL+1
        dW2 = A1.T @ D3  # 这种.T乘矩阵的全部当作(a1,a2,...,an) @ 列
        db2 = np.sum(D3, axis=0, keepdims=True)

        D2 = D3 @ self.model.W2.T  # 每行是DL
        dW1 = X.T @ (D2 * relu_derivative(Z1)) # *是逐元素相乘
        db1 = np.sum(D2 * relu_derivative(Z1), axis=0, keepdims=True)

        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return grads

X_train, y_train = load_npy_dataset_uniform('C:/python/pythonProject/deep_learning/mnist_numpy/train',
                                            samples_per_class=4000)
X_test, y_test = load_npy_dataset_uniform('C:/python/pythonProject/deep_learning/mnist_numpy/test',
                                          samples_per_class=200)

n_samples, n_features = X_train.shape
n_hidden = 300
n_outputs = 10

# 初始化模型
model = TwoLayerNN(n_input=n_features, n_hidden=n_hidden, n_output=n_outputs)
backward = Backward(model)

# 参数加载
def load_model(model, filepath):
    data = np.load(filepath) # 这里np.load输出一个类似字典的对象，需要通过键名访问对应的数组
    model.W1 = data['W1']
    model.b1 = data['b1']
    model.W2 = data['W2']
    model.b2 = data['b2']
    print(f"已从 {filepath} 加载模型参数")

filepath = "C:/python/pythonProject/deep_learning/mnist_numpy/save_parameter/save_parameter.npz"
load_model(model, filepath)

# 训练参数
learning_rate = 0.15
epochs = 1000

for epoch in range(epochs):
    A2, A1, Z1 = model.forward(X_train) # 正向传播
    loss = cross_entropy_loss(A2, y_train) # 计算损失
    grads = backward.run(y_train) # 反向传播

    # 梯度下降更新参数
    model.W1 -= learning_rate * grads['dW1']
    model.b1 -= learning_rate * grads['db1']
    model.W2 -= learning_rate * grads['dW2']
    model.b2 -= learning_rate * grads['db2']

    # 每 100 次打印一次损失
    if (epoch + 1) % 100 == 0 or epoch == 0:
        train_acc = accuracy(A2, y_train)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")

# 参数保存
def save_model(model, filepath):
    np.savez(filepath, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    # np.savez将多个多个numpy数组保存为一个.npz文件，每个数组被赋予一个键，后续读取时按键读取
    print(f"模型参数已保存到 {filepath}")

save_model(model, filepath)

# 训练完后测试准确率
A2_test, _, _ = model.forward(X_test)
test_acc = accuracy(A2_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

