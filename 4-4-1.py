import numpy as np
import scipy

'''
MNIST 数据集下载
https://doc.codingdict.com/tensorflow/tfdoc/tutorials/mnist_download.html

'''
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置输入层节点个数
        self.inodes = inputnodes
        # 隐藏层节点个数
        self.hnodes = hiddennodes
        # 输出层节点个数
        self.onodes = outputnodes
        # 设置学习率
        self.lr = learningrate
        # 隐藏层 权重矩阵 正态分布
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 输出层 权重矩阵 正态分布
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 激活函数 sigmod函数
        self.activation_function = lambda x: scipy.special.expit(x)

    '''训练神经网路'''
    def train(self, input_list, target_list):
        # 转换输入/输出列表为二维数组
        # ndmin = 2的意思是 最小生成的矩阵维度是2，例子::
        # import numpy as np
        # arr = np.array([1, 2, 3], ndmin=2)
        # print(arr)  # 输出: [[1 2 3]]
        # print(arr.shape)  # 输出: (1, 3)

        # ndarray.T : view of the transposed array. 转置函数
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # 计算到隐藏层的信号
        # np.dot 点乘， 为什么这里的np.dot接受的两个参数是矩阵呢？
        '''
        答案：
        np.dot(a, b) 是 NumPy 库中的一个函数，用于计算两个数组的点积。具体来说：
        如果 a 和 b 都是一维数组，它计算的是向量的内积（不进行复数共轭）。
        如果 a 和 b 都是二维数组，它计算的是矩阵乘法。但是，建议使用 matmul 或 a @ b 来执行矩阵乘法。
        如果 a 或 b 是 0-D (标量)，那么它等同于相乘操作，建议使用 numpy.multiply(a, b) 或 a * b。
        如果 a 是一个 N-D 数组，b 是一个 1-D 数组，它计算的是 a 和 b 的最后一个轴上的和积。
        如果 a 是一个 N-D 数组，b 是一个 M-D 数组（其中 M >= 2），它计算的是 a 的最后一个轴和 b 的倒数第二个轴的和积。
        在可能的情况下，它使用了优化的 BLAS 库（参见 numpy.linalg）。

        总之，np.dot(a, b) 的行为根据输入数组的维度和形状而有所不同，可以执行向量的内积、矩阵乘法、和积等操作。
        
        '''
        # 在神经网络中，输入信号通过权重相乘来生成下一层的信号。在这里，self.wih 表示输入层和隐藏层之间的权重，inputs 表示输入信号。通过使用 np.dot，代码执行了权重矩阵和输入矩阵之间的矩阵乘法运算，得到了隐藏层的信号。
        hidden_inputs = np.dot(self.wih, inputs)

        # 计算隐藏层输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算到输出层的信号
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_error = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_error)

        # 更新隐藏层和输出层之间的权重
        # np.transpose 也是转置
        self.who += 2 * self.lr * np.dot((output_error * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # 更新输入层和隐藏层之间的权重
        self.wih += 2 * self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


    '''查询神经网路'''
    def query(self, input_list):
        # 转换输入/输出列表为二维数组
        inputs = np.array(input_list, ndmin=2).T
        # 计算到隐藏层的信号
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算到输出层的信号
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
# 设置学习率
learning_rate = 0.3

neuralNetwork = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 这个csv文件的格式是：首列是标签，后面784列是图像，784列的每一列代表一个像素点的灰度值
# mnist_train.csv文件内含有60000行这样的数值进行训练，以得出最好状态的权重神经网络
training_data_file = open("data/c4/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
print(training_data_list[0])

# 训练神经网路
# 遍历训练数据集文件的每一行
for record in training_data_list:
    all_values = record.split(',')
    # 取值，将后784列的值文本字符串转化为小数形式的灰度值，并且创建这些数字的数组
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # 创建用0填充的数组，数组的长度为output_nodes，加0.01解决输入值为0造成的问题
    targets = np.zeros(output_nodes) + 0.01
    # 取第0列的正确值，使用目标标签，将正确的值设为0.99
    targets[int(all_values[0])] = 0.99
    neuralNetwork.train(inputs, targets)


# mnist_test.csv文件内含有10000行这样的数值进行测试
test_data_file = open("data/c4/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

score_card = []
total = 0
correct = 0

for record in test_data_list:
    total += 1
    all_values = record.split(',')
    # 正确的数字
    correct_label = int(all_values[0])
    
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = neuralNetwork.query(inputs)

    # 返回最大值对应的索引
    label = np.argmax(outputs)
    if label == correct_label:
        score_card.append(1)
        correct += 1
    else:
        score_card.append(0)

print(score_card)
print('正确率：', (correct / total) * 100, '%')

