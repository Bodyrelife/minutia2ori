function mAP = TY_fast_rcnn_test(conf, imdb, roidb, varargin)
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
    ip.addParamValue('iter',             1,             @isscalar);
   
    
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
         if opts.iter==2
             load sdfd;
         end
%        load fdsafd;
       aboxes = cell(num_classes, 1);
       if opts.ignore_cache
           throw('');
       end
       for i = 1:num_classes
         load(fullfile(cache_dir, ['1' '_boxes_' imdbs_name opts.suffix]));
         aboxes{i} = boxes;
       end

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
        max_per_set = inf * num_images;
        min_img = 3;
%         max_per_set = 80 * num_images;
        % heuristic: keep at most 100 detection per class per image prior to NMS
        max_per_image = inf;
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
%            fprintf('%s: test (%s) %d/%d ', procid(), imdbs_name, count, num_images);
            th = tic;
%             d = roidb.rois(i);
            im = imread(imdb{i}.image_path);
            im = repmat(im,1,1,3);
            [boxes, scores] = TY_fast_rcnn_im_detect(conf, caffe_net, im, imdb{i}.boxes, max_rois_num_in_gpu);

            for j = 1:num_classes
                inds = find(~imdb{i}.gt & scores(:, j) > thresh(j));
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

%            fprintf(' time: %.3fs\n', toc(th));  

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
%             aboxes{i} = cellfun(@(x) x(x(:,end)>0.2, :), aboxes{i}, 'UniformOutput', false);

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
            save(save_file, 'boxes', 'inds');
            clear boxes inds;
        end
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end

    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------
    aboxes{i} = cellfun(@(x) x(x(:,end)>0.5, :), aboxes{i}, 'UniformOutput', false);
%     if opts.iter==2
        aboxes{1} = boxes_filter(aboxes{1}, -1, 1.0, -1, conf.use_gpu);    
%     end
    
    precise_ave = 0;
    recall_ave = 0;
    num_precise = 0;
    num_recall = 0;
    loc_ave = 0;
    ori_ave = 0;
    T = num_images;
    PRsave = zeros(3,4);
    Tsave = zeros(3,3);
%     mkdir('minutia_out_nist_1.0');
    for i= 1:num_images
        boxes_cal = aboxes{1}{i};
%         boxes_cal = boxes_cal(boxes_cal(:,end)>0.1,:);
%         if size(boxes_cal,1)<min_img
% %             [~ , sort_s] = sort(boxes_cal(:,end),'descend');
%             boxes_cal = [boxes_cal;aboxes{1}{i}(size(boxes_cal,1)+1:min_img,:)];
%         end
        gt = imdb{i}.gt;
        boxes_gt = imdb{i}.boxes(gt,:);
        if isempty(boxes_cal) || isempty(boxes_gt)
            T = T-1;
            continue;
        end
%         precise = 0;
%         recall = 0;
        dis = pointdis2(boxes_cal,boxes_gt);
        [mindis , idx] = min(dis,[],2);
        angle = abs(mod(boxes_cal(:,3),2*pi) - boxes_gt(idx,3));
        angle = min([angle 2*pi-angle],[],2);
        precise = sum(mindis <= 15 & angle<pi/6);
        recall = length(unique(idx(mindis <= 15 & angle<pi/6)));
%         precise = sum(mindis <= 15);
%         recall = length(unique(idx(mindis <= 15)));
        precise_ave = precise_ave + precise;
        recall_ave = recall_ave + recall;
        num_precise = num_precise + size(boxes_cal,1);
        num_recall = num_recall + size(boxes_gt,1);
        if recall==0
            T = T-1;
        else
            loc_ave = mean(mindis(mindis<15 & angle<pi/6)) + loc_ave;
            ori_ave = mean(angle(mindis<15 & angle<pi/6)) + ori_ave;
        end
        switch imdb{i}.image_path(end-10)
            case 'G'
                iidx=1;
                PRsave(iidx,1)=PRsave(iidx,1)+precise;
                PRsave(iidx,2)=PRsave(iidx,2)+recall;
                Tsave(iidx,1)=Tsave(iidx,1)+size(boxes_cal,1);
                Tsave(iidx,2)=Tsave(iidx,1)+size(boxes_gt,1);
                if recall~=0
                    Tsave(iidx,3) = Tsave(iidx,3)+1;
                    PRsave(iidx,3) = mean(mindis(mindis<15 & angle<pi/6)) + PRsave(iidx,3);
                    PRsave(iidx,4) = mean(angle(mindis<15 & angle<pi/6)) + PRsave(iidx,4);
                end
            case 'B'
                iidx=2;
                PRsave(iidx,1)=PRsave(iidx,1)+precise;
                PRsave(iidx,2)=PRsave(iidx,2)+recall;
                Tsave(iidx,1)=Tsave(iidx,1)+size(boxes_cal,1);
                Tsave(iidx,2)=Tsave(iidx,1)+size(boxes_gt,1);
                if recall~=0
                    Tsave(iidx,3) = Tsave(iidx,3)+1;
                    PRsave(iidx,3) = mean(mindis(mindis<15 & angle<pi/6)) + PRsave(iidx,3);
                    PRsave(iidx,4) = mean(angle(mindis<15 & angle<pi/6)) + PRsave(iidx,4);
                end
            case 'U'
                iidx=3;
                PRsave(iidx,1)=PRsave(iidx,1)+precise;
                PRsave(iidx,2)=PRsave(iidx,2)+recall;
                Tsave(iidx,1)=Tsave(iidx,1)+size(boxes_cal,1);
                Tsave(iidx,2)=Tsave(iidx,1)+size(boxes_gt,1);
                if recall~=0
                    Tsave(iidx,3) = Tsave(iidx,3)+1;
                    PRsave(iidx,3) = mean(mindis(mindis<15 & angle<pi/6)) + PRsave(iidx,3);
                    PRsave(iidx,4) = mean(angle(mindis<15 & angle<pi/6)) + PRsave(iidx,4);
                end
        end
           
%         if opts.iter==2
%             TY_showboxes(imread(imdb{i}.image_path),boxes_cal(:,1:3),boxes_gt(:,1:3));
%             print('-dbmp', ['minutia_out_nist_1.0/' imdb{i}.image_path(end-10:end)]);
%         end
    end
    precise2_ave = recall_ave/num_precise;
    precise_ave = precise_ave/num_precise;
    recall_ave = recall_ave/num_recall;
    

    disp('*******all*******');
    recall_ave
    precise_ave
    precise2_ave
    loc_ave = loc_ave/T
    ori_ave = ori_ave/T
    disp('*******seperate*******');
    disp([PRsave(:,2)./Tsave(:,2), PRsave(:,1)./Tsave(:,1), PRsave(:,2)./Tsave(:,1), PRsave(:,3)./Tsave(:,3), PRsave(:,4)./Tsave(:,3)]);
    
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
            if length(keep)>size(bbox,1)
                keep = keep(1:size(bbox,1));
            end
            boxes{image_index} = bbox(keep,:);
            box_inds{image_index} = box_inds{image_index}(keep);
        end
    end
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
%     after_nms_topN = 180;
    if per_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(length(x), per_nms_topN), :), aboxes, 'UniformOutput', false);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 10   
        parfor i = 1:length(aboxes)
            aboxes{i} = aboxes{i}(TY_nms_3(aboxes{i}, nms_overlap_thres), :);
        end       
    end
    aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
    fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
    if after_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(length(x), after_nms_topN), :), aboxes, 'UniformOutput', false);
    end
end