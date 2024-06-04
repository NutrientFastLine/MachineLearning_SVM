% 混淆矩阵C  
C = [  
     3     0   206     0     0     0     0     0     0     0;  
     0    18   187     0     0     0     0     0     0     0;  
     0     0   169     0     0     0     0     0     0     0;  
     0     0   210     0     0     0     0     0     0     0;  
     0     0   184     0     1     0     0     0     0     0;  
     0     0   184     0     0     0     0     0     0     0;  
     0     0   185     0     0     0     5     0     0     0;  
     0     0   207     0     0     0     0     0     0     0;  
     0     0   232     0     0     0     0     0     0     0;  
     0     0   209     0     0     0     0     0     0     0; 
];
  
% 使用imagesc创建热图  
figure;  
imagesc(C);  
colormap(jet(10)); % 调整颜色映射  
colorbar;  
  
% 添加标题和轴标签  
title('FashionMNIST混淆矩阵');  
xlabel('预测类别');  
ylabel('实际类别');  
  
% 假设你有一个类别标签向量labels（这里我们手动定义）  
labels = {'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'};  
  
% 在X轴和Y轴上添加标签  
set(gca, 'XTick', 1:size(C,2), 'XTickLabel', labels, 'YTick', 1:size(C,1), 'YTickLabel', labels);  
  
% 旋转Y轴标签以节省空间  
set(gca, 'YTickLabelRotation', 45);  
  
% 显示网格线  
grid on;  
  
% 在每个单元格上显示数字  
% x和y已经是二维索引  
for i = 1:size(C, 1)  
    for j = 1:size(C, 2)  
        text(j, i, num2str(C(i, j)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');  
    end  
end  
  
% 如果需要，保存图形到文件  
saveas(gcf, 'confusion_matrix_with_numbers.png');