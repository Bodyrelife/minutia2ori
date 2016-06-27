function TY_finetune_faster_rcnn()

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 5;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% do validation, or not 
opts.do_val                 = true; 
% model
% model                       = Model.TY_ZF_for_Faster_RCNN;
% model                       = Model.TY_VGG16_for_Faster_RCNN;
model                       = Model.TY_VGG16_finetune_for_Faster_RCNN;
% cache base
cache_base_proposal         = 'faster_rcnn_nist274205_VGG16_finetune2_thresh9_lw5';
cache_base_fast_rcnn        = '';

% train/test data
dataset = [];
% if ~exist('data_minutia.mat','file')
%     dataset = load_minutia_data();
% else  
% load data_minutia;
%     load data_minutia_mix;
%     dataset = dataset_test_nist27;
%      load data_minutia_nist27test;
%      load data_minutia_train-27-4205_test-27;
      load data_minutia_27;
%       tempi=[];
%       for i=1:length(dataset.imdb_train{1})
%           if dataset.imdb_train{1}{i}.image_path(end-10)=='G'
%               tempi=[tempi,i];
%           end
%       end
%       dataset.imdb_train{1}(tempi)=[];
%       dataset.roidb_train{1}(tempi)=[];
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

% %%  stage one proposal
% fprintf('\n***************\nstage one proposal \n***************\n');
% % train
% model.stage1_rpn            = Faster_RCNN_Train.TY_do_proposal_train(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% % test
% dataset.imdb_train        	= cellfun(@(x, y) Faster_RCNN_Train.TY_do_proposal_test(conf_proposal, model.stage1_rpn, x, y, 0), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
% dataset.imdb_test        	= cellfun(@(x, y) Faster_RCNN_Train.TY_do_proposal_test(conf_proposal, model.stage1_rpn, x, y, 0), dataset.imdb_test, dataset.roidb_test, 'UniformOutput', false);
% 
% %%  stage one fast rcnn
% fprintf('\n***************\nstage one fast rcnn\n***************\n');
% % train
% % model.stage1_fast_rcnn.init_net_file = model.stage1_rpn.output_model_file;
% 
% model.stage1_fast_rcnn      = Faster_RCNN_Train.TY_do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);
% % test
% opts.mAP                    = Faster_RCNN_Train.TY_do_fast_rcnn_test(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test, 1);

%%  stage two proposal
% net proposal
fprintf('\n***************\nstage two proposal\n***************\n');
% train
model.stage2_rpn.init_net_file = '/media/ssd2/tangy/faster_rcnn-master-hd/output/rpn_cachedir/faster_rcnn_nist274205_VGG16_thresh9_lw5_top-1_nms0_6_top400_stage2_rpn/finger/final';
% model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_rpn            = Faster_RCNN_Train.TY_do_proposal_train(conf_proposal, dataset, model.stage2_rpn, opts.do_val);
% test
dataset.imdb_train       	= cellfun(@(x, y) Faster_RCNN_Train.TY_do_proposal_test(conf_proposal, model.stage2_rpn, x, y, 0), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.imdb_test       	= cellfun(@(x, y) Faster_RCNN_Train.TY_do_proposal_test(conf_proposal, model.stage2_rpn, x, y, 0), dataset.imdb_test, dataset.roidb_test, 'UniformOutput', false);

%%  stage two fast rcnn
fprintf('\n***************\nstage two fast rcnn\n***************\n');
% train
model.stage2_fast_rcnn.init_net_file = model.stage2_rpn.output_model_file;
% model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_fast_rcnn      = Faster_RCNN_Train.TY_do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage2_fast_rcnn, opts.do_val);


%% final test
fprintf('\n***************\nfinal test\n***************\n');
     
model.stage2_rpn.nms        = model.final_test.nms;
dataset.imdb_test       	= cellfun(@(x, y) Faster_RCNN_Train.TY_do_proposal_test(conf_proposal, model.stage2_rpn, x, y, 1), dataset.imdb_test, dataset.roidb_test, 'UniformOutput', false);
opts.final_mAP              = Faster_RCNN_Train.TY_do_fast_rcnn_test(conf_fast_rcnn, model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test,2);

% save final models, for outside tester
Faster_RCNN_Train.TY_gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
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
Files = dir('/media/ssd2/tangy/data_finger/*.bmp');
randidx = randperm(length(Files));
for i=1:floor(length(Files)/5*4)
    dataset.imdb_train{1}{i}.image_path = ['/media/ssd2/tangy/data_finger/' Files(randidx(i)).name];
    dataset.roidb_train{1}{i}.image_path = ['/media/ssd2/tangy/data_finger/' Files(randidx(i)).name(1:end-3) 'mnt'];
    file_mnt = fopen(dataset.roidb_train{1}{i}.image_path);
    str=fgetl(file_mnt);
    num_minutia=fscanf(file_mnt,'%d',1);
    W=fscanf(file_mnt,'%d',1);
    H=fscanf(file_mnt,'%d',1);
    data_minutia=zeros(num_minutia,4);
    for j=1:num_minutia;
        data_minutia(j,1)= fscanf(file_mnt,'%d',1);
        data_minutia(j,2)= fscanf(file_mnt,'%d',1);
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
    dataset.imdb_test{1}{i-floor(length(Files)/5*4)}.image_path = ['/media/ssd2/tangy/data_finger/' Files(randidx(i)).name];
    dataset.roidb_test{1}{i-floor(length(Files)/5*4)}.image_path = ['/media/ssd2/tangy/data_finger/' Files(randidx(i)).name(1:end-3) 'mnt'];
    file_mnt = fopen(dataset.roidb_test{1}{i-floor(length(Files)/5*4)}.image_path);
    str=fgetl(file_mnt);
    num_minutia=fscanf(file_mnt,'%d',1);
    W=fscanf(file_mnt,'%d',1);
    H=fscanf(file_mnt,'%d',1);
    data_minutia=zeros(num_minutia,4);
    for j=1:num_minutia;
        data_minutia(j,1)= fscanf(file_mnt,'%d',1);
        data_minutia(j,2)= fscanf(file_mnt,'%d',1);
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