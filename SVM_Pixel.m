clear;
clc;

% 加载数据
dataPath = 'D:\桌面\ML\ML\zuoye1\ML-SVM\fashion_MNIST.mat';
load(dataPath);

% 确保数据正确加载
assert(exist('data', 'var') == 1, 'data variable not found');
assert(exist('real_label', 'var') == 1, 'real_label variable not found');

% 设置图像尺寸和参数
imgSize = 28; % Fashion MNIST图像尺寸为28x28
numClasses = 10; % 类别数量
numSamples = size(data, 1);

% 划分训练集和测试集
trainRatio = 0.8;
numTrain = floor(trainRatio * numSamples);
numTest = numSamples - numTrain;

% 随机打乱数据
randIndices = randperm(numSamples);
trainIndices = randIndices(1:numTrain);
testIndices = randIndices(numTrain+1:end);

trainData = data(trainIndices, :);
trainLabels = real_label(trainIndices);
testData = data(testIndices, :);
testLabels = real_label(testIndices);

% 使用像素值作为特征
pixelTrainData = trainData;
pixelTestData = testData;

% 定义核函数类型
kernelTypes = {'linear', 'polynomial', 'rbf'};
results = struct();

for k = 1:length(kernelTypes)
    kernel = kernelTypes{k};
    fprintf('使用 %s 核函数训练模型...\n', kernel);
    
    if strcmp(kernel, 'polynomial')
        SVMModel = fitcecoc(pixelTrainData, trainLabels, 'Learners', templateSVM('KernelFunction', kernel, 'PolynomialOrder', 2));
    else
        SVMModel = fitcecoc(pixelTrainData, trainLabels, 'Learners', templateSVM('KernelFunction', kernel));
    end
    
    % 测试模型
    predictedLabels = predict(SVMModel, pixelTestData);
    
    % 计算准确率
    accuracy = sum(predictedLabels == testLabels) / numTest;
    fprintf('测试集准确率 (%s 核): %.2f%%\n', kernel, accuracy * 100);
    
    % 混淆矩阵
    confMat = confusionmat(testLabels, predictedLabels);
    results.(kernel) = struct('accuracy', accuracy, 'confMat', confMat);
end

% 显示结果
for k = 1:length(kernelTypes)
    kernel = kernelTypes{k};
    fprintf('核函数: %s\n', kernel);
    fprintf('准确率: %.2f%%\n', results.(kernel).accuracy * 100);
    disp('混淆矩阵:');
    disp(results.(kernel).confMat);
end
