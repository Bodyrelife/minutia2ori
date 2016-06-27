function imdb = TY_hard_negtive_test(conf, imdb, roidb, varargin)
% mAP = fast_rcnn_test(conf, imdb, roidb, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @iscell);
    ip.addRequired('roidb',                             @iscell);
    ip.addParamValue('net_def_file',    '', 			@isstr);
    ip.addParamValue('net_file',        '', 			@isstr);
    ip.addParamValue('cache_name',      '', 			@isstr);                                         
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('ignore_cache',    false,          @islogical);
    
    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
    

%%  set cache dir
    imdbs_name = 'finger';
    cache_dir = fullfile(pwd, 'output', 'fast_rcnn_cachedir', opts.cache_name, imdbs_name);
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
    diary(log_file);
    
    num_images = length(imdb);
    num_classes = 1;
    
    try
%       aboxes = cell(num_classes, 1);
%       if opts.ignore_cache
%           throw('');
%       end
%       for i = 1:num_classes
%         load(fullfile(cache_dir, ['1' '_boxes_' imdbs_name opts.suffix]));
%         aboxes{i} = boxes;
%       end
        load fdsafd;
    catch    
%%      testing 
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(opts.net_def_file, 'test');
        caffe_net.copy_from(opts.net_file);

        % set random seed
        prev_rng = seed_rand(conf.rng_seed);
        caffe.set_random_seed(conf.rng_seed);

        % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        % determine the maximum number of rois in testing 
        max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);
        
        %heuristic: keep an average of 40 detections per class per images prior to NMS
        max_per_set = 60 * num_images;
        min_img = 3;
%         max_per_set = 80 * num_images;
        % heuristic: keep at most 100 detection per class per image prior to NMS
        max_per_image = 200;
        % detection thresold for each class (this is adaptively set based on the max_per_set constraint)
        thresh = -inf * ones(num_classes, 1);
        % top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
        top_scores = cell(num_classes, 1);
        % all detections are collected into:
        %    all_boxes[cls][image] = N x 5 array of detections in
        %    (x1, y1, x2, y2, score)
        aboxes = cell(num_classes, 1);
        box_inds = cell(num_classes, 1);
        for i = 1:num_classes
            aboxes{i} = cell(length(imdb), 1);
            box_inds{i} = cell(length(imdb), 1);
        end

        count = 0;
        t_start = tic;
        for i = 1:num_images
            count = count + 1;
%             fprintf('%s: test (%s) %d/%d ', procid(), imdbs_name, count, num_images);
            th = tic;
%             d = roidb.rois(i);
            im = imread(opts.imdb{i}.image_path);
            im = repmat(im,1,1,3);
            gt = opts.imdb{i}.gt;
            [boxes, scores] = TY_fast_rcnn_im_detect(conf, caffe_net, im, opts.imdb{i}.boxes, max_rois_num_in_gpu);
            %drop_simple_negtive
            
            if sum(gt)==0
                continue;
            end
            boxes_gt = opts.imdb{i}.boxes(gt,:);
            dis = pointdis2(opts.imdb{i}.boxes,boxes_gt);
            [mindis , idx] = min(dis,[],2);
%             angle = abs(boxes(:,3) - boxes_gt(idx,3));
%             angle = min([angle 2*pi-angle],[],2);
%             thresh_s = sort(scores,'descend');
%             thresh_s = thresh_s(floor(end/3));
            randth = rand(length(scores),1)>(scores./sum(scores));
            imdb{i}.boxes(mindis > 15  & randth,:) = [];
            imdb{i}.gt( mindis > 15 & randth) = [];
            fprintf('%d---%d---',i,size(imdb{i}.boxes,1));
            for j = 1:num_classes
                inds = find(~opts.imdb{i}.gt & scores(:, j) > thresh(j));
                if ~isempty(inds)
                    [~, ord] = sort(scores(inds, j), 'descend');
                    ord = ord(1:min(length(ord), max_per_image));
                    inds = inds(ord);
                    cls_boxes = boxes(inds, (1+(j-1)*4):((j)*4));
                    cls_scores = scores(inds, j);
                    aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores))];
                    box_inds{j}{i} = [box_inds{j}{i}; inds];
                else
                    aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 5, 'single')];
                    box_inds{j}{i} = box_inds{j}{i};
                end
            end

 %           fprintf(' time: %.3fs\n', toc(th));  

            if mod(count, 1000) == 0
                for j = 1:num_classes
                [aboxes{j}, box_inds{j}, thresh(j)] = ...
                    keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, min_img, thresh(j));
                end
                disp(thresh);
            end    
        end
 

        for j = 1:num_classes
            [aboxes{j}, box_inds{j}, thresh(j)] = ...
                keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, min_img, thresh(j));
        end
        disp(thresh);

        for i = 1:num_classes

%             top_scores{i} = sort(top_scores{i}, 'descend');  
%             if (length(top_scores{i}) > max_per_set)
%                 thresh(i) = top_scores{i}(max_per_set);
%             end

%             % go back through and prune out detections below the found threshold
%             for j = 1:length(imdb)
%                 if ~isempty(aboxes{i}{j})
%                     I = find(aboxes{i}{j}(:,end) < thresh(i));
%                     aboxes{i}{j}(I,:) = [];
%                     box_inds{i}{j}(I,:) = [];
%                 end
%             end

%             save_file = fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdbs_name opts.suffix]);
            save_file = fullfile(cache_dir, ['1' '_boxes_' imdbs_name opts.suffix]);
            boxes = aboxes{i};
            inds = box_inds{i};
%             save(save_file, 'boxes', 'inds');
            clear boxes inds;
        end
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end

    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------
    precise_ave = 0;
    recall_ave = 0;
    num_precise = 0;
    num_recall = 0;
    for i= 1:num_images
        boxes_cal = aboxes{1}{i};
%         boxes_cal = boxes_cal(boxes_cal(:,end)>0.6,:);
%         if size(boxes_cal,1)<min_img
% %             [~ , sort_s] = sort(boxes_cal(:,end),'descend');
%             boxes_cal = [boxes_cal;aboxes{1}{i}(size(boxes_cal,1)+1:min_img,:)];
%         end
        gt = opts.imdb{i}.gt;
        boxes_gt = opts.imdb{i}.boxes(gt,:);
        if isempty(boxes_cal) || isempty(boxes_gt)
            continue;
        end
%         precise = 0;
%         recall = 0;
        dis = pointdis2(boxes_cal,boxes_gt);
        [mindis , idx] = min(dis,[],2);
        angle = abs(boxes_cal(:,3) - boxes_gt(idx,3));
        angle = min([angle 2*pi-angle],[],2);
        precise = sum(mindis <= 15 & angle<pi/6);
        recall = length(unique(idx(mindis <= 15 & angle<pi/6)));
%         precise = sum(mindis <= 15);
%         recall = length(unique(idx(mindis <= 15)));
        precise_ave = precise_ave + precise;
        recall_ave = recall_ave + recall;
        num_precise = num_precise + size(boxes_cal,1);
        num_recall = num_recall + size(boxes_gt,1);
        
    end
    precise_ave = precise_ave/num_precise;
    recall_ave = recall_ave/num_recall;

    recall_ave
    precise_ave
%     if isequal(imdb.eval_func, @imdb_eval_voc)
%         for model_ind = 1:num_classes
% %           cls = imdb.classes{model_ind};
%           cls = 1;
%           res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, opts.cache_name, opts.suffix);
%         end
%     else
%     % ilsvrc
%         res = imdb.eval_func(aboxes, imdb, opts.cache_name, opts.suffix);
%     end

%     if ~isempty(res)
%         fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
%         fprintf('Results:\n');
%         aps = [res(:).ap]' * 100;
%         disp(aps);
%         disp(mean(aps));
%         fprintf('~~~~~~~~~~~~~~~~~~~~\n');
%         mAP = mean(aps);
%     else
        mAP = precise_ave;
%     end
    
    diary off;
end

function max_rois_num = check_gpu_memory(conf, caffe_net)
%%  try to determine the maximum number of rois

    max_rois_num = 0;
    for rois_num = 500:500:5000
        % generate pseudo testing data with max size
        im_blob = single(zeros(conf.max_size, conf.max_size, 3, 1));
        rois_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
        rois_blob = permute(rois_blob, [3, 4, 1, 2]);

        net_inputs = {im_blob, rois_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);

        caffe_net.forward(net_inputs);
        gpuInfo = gpuDevice();

        max_rois_num = rois_num;
            
        if gpuInfo.FreeMemory < 2 * 10^9  % 2GB for safety
            break;
        end
    end

end


% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, min_img, thresh)
% ------------------------------------------------------------------------
    % Keep top K
    X = cat(1, boxes{1:end_at});
    if isempty(X)
        return;
    end
    scores = sort(X(:,end), 'descend');
    thresh = scores(min(length(scores), top_k));
    for image_index = 1:end_at
        if ~isempty(boxes{image_index})
            bbox = boxes{image_index};
            keep = find(bbox(:,end) >= thresh);
            if length(keep) < min_img
%                 [~,inds] = sort(bbox(:,end),'descend');
                keep = [keep;(length(keep)+1:min_img).'];
            end
            boxes{image_index} = bbox(keep,:);
            box_inds{image_index} = box_inds{image_index}(keep);
        end
    end
end