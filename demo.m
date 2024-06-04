clear
clc
% Author：kailugaji https://www.cnblogs.com/kailugaji/
filename_data='C:\Users\yuhannong\Documents\MATLAB\ML-SVM\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte'; %自行修改路径
data = loadMNISTImages(filename_data);
data=data';
filename_label='C:\Users\yuhannong\Documents\MATLAB\ML-SVM\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte'; %自行修改路径
real_label = loadMNISTLabels(filename_label);
%     标签    所代表的意思
%     0 短袖圆领T恤
%     1 裤子
%     2 套衫
%     3 连衣裙
%     4 外套
%     5 凉鞋
%     6 衬衫
%     7 运动鞋
%     8 包
%     9 短靴
%  real_label(real_label==0)=10;
save fashion_MNIST data real_label
 
Image_samples=Image_integration(data, real_label, 10);
A=mat2gray(Image_samples);
figure(1)
imshow(A, 'Border','tight');
print(gcf,'-r1000','-djpeg','My_Fashion_MNIST.jpg');