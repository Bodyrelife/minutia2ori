function scores=useCNN(net,im)

% % Add caffe/matlab to you Matlab search PATH to use matcaffe
% addpath('+caffe');
% % Set caffe mode
%  caffe.set_mode_cpu();
% 
% model = '/home/tangy/caffe-master/examples/minutia/lenet.prototxt';
% weights = '/home/tangy/caffe-master/examples/minutia/snapshot/net612_batch512_newalldata_iter_40000.caffemodel';
% net = caffe.Net(model, weights, 'test'); % create net and load weights

if nargin < 1
    im=imread('testimg.bmp');
end
% im = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im, [2, 1, 3, 4]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
% im_data = imresize(im_data, [256 256], 'bilinear');  % resize im_data
input_data{1}=im_data;
scores = net.forward(input_data);
scores=scores{1};
% [~, maxlabel] = max(scores);
end