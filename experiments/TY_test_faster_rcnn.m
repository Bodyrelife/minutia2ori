function TY_test_faster_rcnn()
close all;
clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 4;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.per_nms_topN           = 1200;
opts.nms_overlap_thres      = 0;
opts.after_nms_topN         = 30;
opts.use_gpu                = true;

opts.test_scales            = 800;

%% -------------------- INIT_MODEL --------------------
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_vgg_16layers'); %% VGG-16
model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_nist274205_ZF_nmsprop'); 
proposal_detection_model    = load_proposal_detection_model(model_dir);

proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = gpuArray(proposal_detection_model.conf_proposal.image_means);
    proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);
end

% caffe.init_log(fullfile(pwd, 'caffe_log'));
% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end       

%% -------------------- WARM UP --------------------
% the first run will be slower; use an empty image to warm up

for j = 1:2 % we warm up 2 times
    im = uint8(ones(800, 800, 3)*128);

    if opts.use_gpu
        im = gpuArray(im);
    end
    [boxes, scores]             = TY_proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    if proposal_detection_model.is_share_feature
        [boxes, scores]             = TY_fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = TY_fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
end

%% -------------------- TESTING --------------------
% im_names = {'001763.jpg', '004545.jpg', '000542.jpg', '000456.jpg', '001150.jpg'};
im_names = {'A0003205008882003040093_03.bmp', 'A0003302110002003085187_01.bmp', 'A0003302820402003070384_01.bmp','G009L8U.bmp','B101L9U.bmp','U201L6U.bmp'};
% these images can be downloaded with fetch_faster_rcnn_final_model.m

running_time = [];
for j = 1:length(im_names)
    
    im = imread(fullfile(pwd, im_names{j}));
    im = repmat(im,1,1,3);
    if opts.use_gpu
        im = gpuArray(im);
    end
    
    % test proposal
    th = tic();
    [boxes, scores]             = TY_proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
    t_proposal = toc(th);
    th = tic();
    aboxes = [boxes,scores];
%     aboxes = cellfun(@(x) x(x(:,end)>0.8, :), aboxes, 'UniformOutput', false);
    aboxes = aboxes(aboxes(:,end)>0.8,:);
%     aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    t_nms = toc(th);
    
    % test detection
    th = tic();
    if proposal_detection_model.is_share_feature
        [boxes, scores]             = TY_fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = TY_fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
    t_detection = toc(th);
    
    fprintf('%s (%dx%d): time %.3fs (resize+conv+proposal: %.3fs, nms+regionwise: %.3fs)\n', im_names{j}, ...
        size(im, 2), size(im, 1), t_proposal + t_nms + t_detection, t_proposal, t_nms+t_detection);
    running_time(end+1) = t_proposal + t_nms + t_detection;
    
%     aboxes = [boxes,scores];
    boxes = boxes(scores>0.3,:);
    scores = scores(scores>0.3);
    aboxes = boxes_filter([boxes, scores],-1,1.0, -1, opts.use_gpu);
    TY_showboxes(imread(fullfile(pwd, im_names{j})),aboxes(:,1:3));
    % visualize
%     classes = {1};
%     boxes_cell = cell(length(classes), 1);
%     thres = 0.9;
%     for i = 1:length(boxes_cell)
%         boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
% %         boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);
%         
%         I = boxes_cell{i}(:, 5) >= thres;
%         boxes_cell{i} = boxes_cell{i}(I, :);
%     end
%     figure(j);
%     TY_showboxes(im, boxes_cell{1}(:,1:3));
%     pause(0.1);
end
fprintf('mean time: %.3fs\n', mean(running_time));

caffe.reset_all(); 
clear mex;

end

function proposal_detection_model = load_proposal_detection_model(model_dir)
    ld                          = load(fullfile(model_dir, 'model'));
    proposal_detection_model    = ld.proposal_detection_model;
    clear ld;
    
    proposal_detection_model.proposal_net_def ...
                                = fullfile(model_dir, proposal_detection_model.proposal_net_def);
    proposal_detection_model.proposal_net ...
                                = fullfile(model_dir, proposal_detection_model.proposal_net);
    proposal_detection_model.detection_net_def ...
                                = fullfile(model_dir, proposal_detection_model.detection_net_def);
    proposal_detection_model.detection_net ...
                                = fullfile(model_dir, proposal_detection_model.detection_net);
    
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(TY_nms(aboxes, nms_overlap_thres), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end
