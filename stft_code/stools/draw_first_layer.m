use_gpu = 1; gpu_id = 0;
cpath='/home/liuyh/FAFFE/examples/lyhnet_fp_img_sz80/';
net_model = [cpath , 'deploy.prototxt'];
net_weights = [cpath , 'snapshots/fp_iter_85000.caffemodel'];
layer_name = {'conv1'};
phase = 'test';

%%
%InitialNetwork
% Add caffe/matlab to you Matlab search PATH to use matcaffe
addpath('/home/liuyh/FAFFE/matlab/');
% Set caffe mode
if use_gpu == 1
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end
% Initialize the network
if ~exist(net_model, 'file')
    error(['Can not find file:', net_model]);
elseif ~exist(net_weights, 'file')
    error(['Can not find file:', net_weights]);
end
net = caffe.Net(net_model, net_weights, phase);
% Load Data
data = cell(size(layer_name));
for i = 1:size(layer_name, 1)
    data{i} = net.params(layer_name{i}, 1).get_data();
    data{i} = permute(data{i}, [2, 1, 3, 4]);
    S = size(data{i});
    data{i} = squeeze(reshape(data{i}, [S(1) * S(2), S(3) * S(4)]));
end
%show
figure
display_network_gaof(data{1});
% print -dbmp ../../python/netff/finger_1:10_weightloss5_155155151.bmp
% print -dbmp ../../python/netff/finger_noise_fine_1.4w.bmp
% figure
% display_network(data{2});
caffe.reset_all();
