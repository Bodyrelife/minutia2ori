function TY_train_minutia2ori()

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 6;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% do validation, or not 
opts.do_val                 = true; 
% model
% model                       = Model.TY_ZF_for_Faster_RCNN;
model                       = Model.TY_VGG16_for_Faster_RCNN;
% model                       = Model.TY_VGG16_finetune_for_Faster_RCNN;
% cache base
cache_base_proposal         = 'faster_rcnn_minutia_VGG16';
cache_base_fast_rcnn        = '';

% train/test data
dataset = [];
if ~exist('data_minutia.mat','file')
    dataset = load_minutia_data();
else  
    load data_minutia;
end
% load data_minutia_6_6
% end
SCALE = 800;
% load NIST27ROI_minutia;
% dataset.imdb_train           = {dir('/media/ssd2/tangy/data_finger/train/*.bmp')};
% dataset.imdb_test            = {dir('/media/ssd2/tangy/data_finger/test/*.bmp')};
% dataset.roidb_train          = {dir('/media/ssd2/tangy/data_finger/train/*.mnt')};
% dataset.roidb_test           = {dir('/media/ssd2/tangy/data_finger/test/*.mnt')};
%% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config('image_means', model.mean_image, 'feat_stride', model.feat_stride);
conf_proposal.fg_thresh = 9; 
conf_proposal.bg_thresh_lo = 15;
conf_proposal.bg_thresh_hi = inf;
conf_proposal.scales = SCALE;
conf_proposal.test_scales = SCALE;
% conf_proposal.fg_fraction = 0.25;
% conf_proposal.bg_weight = 0.5;
% conf_proposal.batch_size = 128;
conf_fast_rcnn              = fast_rcnn_config('image_means', model.mean_image);
conf_fast_rcnn.bbox_thresh = 9;
conf_fast_rcnn.fg_thresh = 9; 
conf_fast_rcnn.bg_thresh_lo = 15;
conf_fast_rcnn.bg_thresh_hi = inf;
conf_fast_rcnn.scales = SCALE;
conf_fast_rcnn.test_scales = SCALE;
% conf_fast_rcnn.fg_fraction = 0.5;
% conf_fast_rcnn.batch_size = 64;
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);
% generate anchors and pre-calculate output size of rpn network 
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

%%  stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn            = Faster_RCNN_Train.TY_do_proposal_train(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% test
aboxes_train        	= cellfun(@(x, y) Faster_RCNN_Train.TY_do_proposal_test(conf_proposal, model.stage1_rpn, x, y, 0), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
aboxes_test        	= cellfun(@(x, y) Faster_RCNN_Train.TY_do_proposal_test(conf_proposal, model.stage1_rpn, x, y, 0), dataset.imdb_test, dataset.roidb_test, 'UniformOutput', false);

% save final models, for outside tester
% Faster_RCNN_Train.TY_gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file);
%     anchors                = proposal_generate_anchors(cache_name, ...
%                                     'scales',  2.^[3:5]);
    anchors = [-15 -15 16 16];
end

function dataset = load_minutia_data()
dataset                     = [];
% use_flipped                 = true;
% dataset                     = Dataset.voc2007_trainval(dataset, 'train', use_flipped);
% dataset                     = Dataset.voc2007_test(dataset, 'test', false);
Files = dir('/media/ssd2/tangy/FVC/FVC2002/DB2_A/*.bmp');
randidx = randperm(length(Files));
for i=1:floor(length(Files)/5*4)
    dataset.imdb_train{1}{i}.image_path = ['/media/ssd2/tangy/FVC/FVC2002/DB2_A/' Files(randidx(i)).name];
    dataset.roidb_train{1}{i}.image_path = ['/media/ssd2/tangy/FVC/FVC2002/DB2_A_mnt/' Files(randidx(i)).name(1:end-3) 'mnt'];
    file_mnt = fopen(dataset.roidb_train{1}{i}.image_path);
    str=fgetl(file_mnt);
    num_minutia=fscanf(file_mnt,'%d',1);
    W=fscanf(file_mnt,'%d',1);
    H=fscanf(file_mnt,'%d',1);
    data_minutia=zeros(num_minutia,4);
    for j=1:num_minutia;
        data_minutia(j,1)= fscanf(file_mnt,'%f',1);
        data_minutia(j,2)= fscanf(file_mnt,'%f',1);
        data_minutia(j,3)=fscanf(file_mnt,'%f',1);
%         data_minutia(j,4)=fscanf(file_mnt,'%d',1);
%         data_minutia(j,5)=fscanf(file_mnt,'%f',1);
        data_minutia(j,4)=1;
    %     data_minutia(j,5)=1.0;
    end;
    fclose(file_mnt);
    dataset.imdb_train{1}{i}.boxes = data_minutia;
    dataset.imdb_train{1}{i}.W = W;
    dataset.imdb_train{1}{i}.H = H;
    dataset.imdb_train{1}{i}.gt = true(num_minutia,1);
end
for i=floor(length(Files)/5*4)+1:length(Files)
    dataset.imdb_test{1}{i-floor(length(Files)/5*4)}.image_path = ['/media/ssd2/tangy/FVC/FVC2002/DB2_A/' Files(randidx(i)).name];
    dataset.roidb_test{1}{i-floor(length(Files)/5*4)}.image_path = ['/media/ssd2/tangy/FVC/FVC2002/DB2_A_mnt/' Files(randidx(i)).name(1:end-3) 'mnt'];
    file_mnt = fopen(dataset.roidb_test{1}{i-floor(length(Files)/5*4)}.image_path);
    str=fgetl(file_mnt);
    num_minutia=fscanf(file_mnt,'%d',1);
    W=fscanf(file_mnt,'%d',1);
    H=fscanf(file_mnt,'%d',1);
    data_minutia=zeros(num_minutia,4);
    for j=1:num_minutia;
        data_minutia(j,1)= fscanf(file_mnt,'%f',1);
        data_minutia(j,2)= fscanf(file_mnt,'%f',1);
        data_minutia(j,3)=fscanf(file_mnt,'%f',1);
%         data_minutia(j,4)=fscanf(file_mnt,'%d',1);
%         data_minutia(j,5)=fscanf(file_mnt,'%f',1);
        data_minutia(j,4)=1;
    %     data_minutia(j,5)=1.0;
    end;
    fclose(file_mnt);
    dataset.imdb_test{1}{i-floor(length(Files)/5*4)}.boxes = data_minutia;
    dataset.imdb_test{1}{i-floor(length(Files)/5*4)}.W = W;
    dataset.imdb_test{1}{i-floor(length(Files)/5*4)}.H = H;
    dataset.imdb_test{1}{i-floor(length(Files)/5*4)}.gt = true(num_minutia,1);
end
save('data_minutia.mat','dataset');
end