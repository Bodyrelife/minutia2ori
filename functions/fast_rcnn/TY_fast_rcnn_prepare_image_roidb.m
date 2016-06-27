function [image_roidb, bbox_means, bbox_stds] = TY_fast_rcnn_prepare_image_roidb(conf, image_roidb, bbox_means, bbox_stds)
% [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb(conf, imdbs, roidbs, cache_img, bbox_means, bbox_stds)
%   Gather useful information from imdb and roidb
%   pre-calculate mean (bbox_means) and std (bbox_stds) of the regression
%   term for normalization
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% -------------------------------------------------------- 
    
    if ~exist('bbox_means', 'var')
        bbox_means = [];
        bbox_stds = [];
    end
    
%     if ~iscell(imdbs)
%         imdbs = {imdbs};
%         roidbs = {roidbs};
%     end
% 
%     imdbs = imdbs(:);
%     roidbs = roidbs(:);
%     
%     image_roidb = ...
%         cellfun(@(x, y) ... // @(imdbs, roidbs)
%                 arrayfun(@(z) ... //@([1:length(x.image_ids)])
%                         struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
%                         'overlap', y.rois(z).overlap, 'boxes', y.rois(z).boxes, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
%                 [1:length(x.image_ids)]', 'UniformOutput', true),...
%         imdbs, roidbs, 'UniformOutput', false);
%     
%     image_roidb = cat(1, image_roidb{:});
    
    % enhance roidb to contain bounding-box regression targets
    [image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
end

function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
    % means and stds -- (k+1) * 4, include background class

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = 1;
    valid_imgs = true(num_images, 1);
    for i = 1:num_images
       rois = image_roidb{i}.boxes; 
       [image_roidb{i}.bbox_targets, image_roidb{i}.boxes, valid_imgs(i)] = ...
           compute_targets(conf, rois, image_roidb{i}.gt);
    end
    if ~all(valid_imgs)
        image_roidb = image_roidb(valid_imgs);
        num_images = length(image_roidb);
        fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
    end
        
    if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds))
        % Compute values needed for means and stds
        % var(x) = E(x^2) - E(x)^2
        class_counts = zeros(num_classes, 1) + eps;
        sums = zeros(num_classes, 4);
        squared_sums = zeros(num_classes, 4);
        for i = 1:num_images
           targets = image_roidb{i}.bbox_targets;
           for cls = 1:num_classes
              cls_inds = find(targets(:, 1) == cls);
              if ~isempty(cls_inds)
                 class_counts(cls) = class_counts(cls) + length(cls_inds); 
                 sums(cls, :) = sums(cls, :) + sum(targets(cls_inds, 2:end), 1);
                 squared_sums(cls, :) = squared_sums(cls, :) + sum(targets(cls_inds, 2:end).^2, 1);
              end
           end
        end

        means = bsxfun(@rdivide, sums, class_counts);
        stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
        means(stds==0) = 0;
        stds(stds==0) = 1;
%         means(3)=0;stds(3)=1;
        % add background class
        means = [0, 0, 0, 0; means]; 
        stds = [0, 0, 0, 0; stds];
    end
    
    % Normalize targets
    for i = 1:num_images
        targets = image_roidb{i}.bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                image_roidb{i}.bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb{i}.bbox_targets(cls_inds, 2:end), means(cls+1, :));
                image_roidb{i}.bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb{i}.bbox_targets(cls_inds, 2:end), stds(cls+1, :));
            end
        end
    end
end


function [bbox_targets, rois, is_valid] = compute_targets(conf, rois, gt)

%     overlap = full(overlap);

%     [max_overlaps, max_labels] = max(overlap, [], 2);

    % ensure ROIs are floats
    rois = single(rois);
    
    bbox_targets = zeros(size(rois, 1), 5, 'single');
    
    rois(~gt,1:2) = [(rois(~gt,3)+rois(~gt,1))./2 (rois(~gt,4)+rois(~gt,2))./2];
    
    % Indices of ground-truth ROIs
    gt_inds = find(gt == 1);
    dis = pointdis2(rois, rois(gt_inds,:));
%     dis(gt_inds,:) = 0;
    max_overlaps = min(dis,[],2);
    if ~isempty(gt_inds)
        % Indices of examples for which we try to make predictions
        ex_inds = find(max_overlaps <= conf.bbox_thresh);

        % Get IoU overlap between each ex ROI and gt ROI
        ex_gt_overlaps = pointdis2(rois(ex_inds, :), rois(gt_inds, :));

%         assert(all(abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4));

        % Find which gt ROI each ex ROI has max overlap with:
        % this will be the ex ROI's gt target
        [~, gt_assignment] = min(ex_gt_overlaps, [], 2);
        gt_rois = rois(gt_inds(gt_assignment), :);
        ex_rois = rois(ex_inds, :);

        [regression_label] = TY_fast_rcnn_bbox_transform2(ex_rois, gt_rois);

        bbox_targets(ex_inds, :) = [ones(length(ex_inds),1), regression_label];
    end
    
    % Select foreground ROIs as those with >= fg_thresh overlap
    is_fg = max_overlaps <= conf.fg_thresh;
    % Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
    is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;
    
    % check if there is any fg or bg sample. If no, filter out this image
    is_valid = true;
    if ~any(is_fg | is_bg)
        is_valid = false;
    end
end