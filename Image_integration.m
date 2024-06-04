function Image_samples=Image_integration(data, real_label, N_samples)
% Gray image integration
% This code only applies to square matrices
% Input:
% data: dataset. N*Dim
% real_label: GroundTruth. N*1
% N_samples: number of selected samples
% Output:
% Image_samples:Integrated image
% Author: kailugaji https://www.cnblogs.com/kailugaji/
[~, Dim]=size(data);
[real_label, b]=sort(real_label);
data=data(b, :);
K=length(unique(real_label)); % number of cluster
[~, ID]=unique(real_label);
ID=ID-1;
image_10=cell(N_samples, K);
temp=cell(N_samples, K);
Image_samples=[];
for i=1:N_samples
    for j=1:K
        temp{i, j}=reshape(data(ID(j)+i, :), sqrt(Dim), sqrt(Dim)); % you can change its size
        image_10{i, j}=[image_10{i, j}, temp{i, j}];
    end
    Image_samples=[Image_samples; image_10{i, :}];
end