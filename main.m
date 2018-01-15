%%%test for Random MultiGraphs
clear all;
clc;

addpath('./rmg/');
addpath('./util/');

%===== basic parameters =================================
bandNum = 3; % number of band for LPE band selection  
w = 10;       % patch size
% LBP feature extraction
r = 2;  nr = 8;
% number of graphs
kg = 6;

fprintf('... ... loading data begin ...\n');
load SeaIceDataset.mat;
fprintf('... ... loading data finished !!! \n');

% training number for Inidan_Pines dataset
CTrain = [5 143 83 24 48 73 3 48 2 97 246 59 21 127 39 9];
% training number for Pavia_University dataset
% CTrain = [ 66 186 21 31 13 50 13 37 9 ];
% training number for Baffin Bay dataset
%CTrain = [75 136 526 114];
%===== basic parameters =================================




no_class = max(gth(:));

% 数据归一化
fprintf(' ... ... data normalization    ... ...\n');
Data = z./max(z(:));
[ylen, xlen, spec_dim] = size(Data);

% band selection  
fprintf(' ... ... band selection        ... ...\n');
X = reshape(Data, ylen*xlen, spec_dim);
Psi = PCA_Train(X', bandNum);
X = X*Psi;
DataTmp = reshape(X, ylen, xlen, size(Psi,2));
clear X Psi;

mapping = getmapping(nr,'u2'); 
fprintf(' ... ... LBP feature extraction begin ... ...\n');
Feature_P = LBP_feature_global(DataTmp, r, nr, mapping, w, gth);
clear nr r z DataTmp;


lbp_dim = size(Feature_P, 3);
% spatial  data
DataSpat = NewScale(reshape(Feature_P, ylen*xlen, lbp_dim));
% spectral data
DataSpec = NewScale(reshape(Data, ylen*xlen, spec_dim));
% spatial and spectral data combination
Data_spec_spat = [DataSpat, DataSpec];
clear DataSpat DataSpec Data Feature_P bandNum;
clear lbp_dim mapping w;

Data = []; Labels = [];

for i = 1: no_class
    pos = find(gth==i);
    Data = [Data; Data_spec_spat(pos, :)];
    Labels = [Labels, length(pos)];
end
clear  Data_spec_spat;


DataTrn = []; DataTst = [];  CTest = [];
k = 0; 
for i = 1: no_class
    Data_tmp = Data((k+1):(Labels(i)+k), :);
    k = Labels(i) + k;
    rand('seed', 2);
    index_i = randperm(Labels(i));
    DataTrn = [DataTrn; Data_tmp(index_i(1:CTrain(i)), :)];
    DataTst = [DataTst; Data_tmp(index_i(CTrain(i)+1:end), :)];
    CTest =  [CTest length(index_i(CTrain(i)+1:end))];
end
clear k Data_tmp Data;

TrnLab = []; TstLab = [];
for jj = 1: length(CTrain)
   TrnLab = [TrnLab; jj * ones(CTrain(jj),1)];
end
for jj = 1: length(CTest)
   TstLab = [TstLab; jj * ones(CTest(jj),1)];
end
 
% Scale the data
X=[DataTrn;DataTst];
[N,Dim] = size(X);
clear CTest CTrain gth DataTrn DataTst; 



fprintf('... ... Graph number:%d ... ...\n', kg);

%kf=floor(log2(Dim)+1);
% gaofeng revised code 2017/04/08
kf = floor(Dim/4);

% 得到训练样本在整体样本集中的 index 
label_index = find(TrnLab~=0);
labels = [TrnLab;TstLab];
[G,F]  = MultiGraphs(X,labels,label_index,kg,kf);


% 获得分类结果
[val, predict_res]=max(F,[],2);
        
[Pr, ConfMat] = GetAccuracy(predict_res(length(label_index)+1:end), ...
                            labels(length(label_index)+1:end));

fprintf(' ... ... Final Accuracy: %f\n', Pr.OA);
fprintf(' ... ... Final Kappa: %f\n', Pr.Kappa);

