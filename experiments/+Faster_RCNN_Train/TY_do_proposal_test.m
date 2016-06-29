function aboxes = TY_do_proposal_test(conf, model_stage, imdb, roidb, ff)
    aboxes                      = TY_proposal_test(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name);      
%     if ff==0
%     model_stage.nms.after_nms_topN = -1;
%     aboxes = cellfun(@(x) x(x(:,end)>0.005, :), aboxes, 'UniformOutput', false);
% %     model_stage.nms.nms_overlap_thres = 0;
%     else if ff==1
%     model_stage.nms.after_nms_topN = -1;
%     aboxes = cellfun(@(x) x(x(:,end)>0.8, :), aboxes, 'UniformOutput', false);
%     model_stage.nms.nms_overlap_thres = 0;
%         end
%     end
%     aboxes                      = boxes_filter(aboxes, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);    
%     
%     roidb_regions               = make_roidb_regions(aboxes, length(imdb));  
%     
%     imdb_new                   = TY_roidb_from_proposal(imdb, roidb, roidb_regions, ...
%                                         'keep_raw_proposal', false);    
%     precise_ave = 0;
%     recall_ave = 0;
%     num_precise = 0;
%     num_recall = 0;
% %     mkdir('minutia_prop');
%     for i= 1:length(imdb)
%         boxes_cal = aboxes{i};
%         boxes_cal(:,1) = (boxes_cal(:,1) + boxes_cal(:,3))/2;
%         boxes_cal(:,2) = (boxes_cal(:,2) + boxes_cal(:,4))/2;
% %         boxes_cal = boxes_cal(boxes_cal(:,end)>0.8,:);
% %         if size(boxes_cal,1)<min_img
% % %             [~ , sort_s] = sort(boxes_cal(:,end),'descend');
% %             boxes_cal = [boxes_cal;aboxes{1}{i}(size(boxes_cal,1)+1:min_img,:)];
% %         end
%         gt = imdb{i}.gt;
%         boxes_gt = imdb{i}.boxes(gt,:);
% %         precise = 0;
% %         recall = 0;
%         dis = pointdis2(boxes_cal,boxes_gt);
%         [mindis , idx] = min(dis,[],2);
% %         angle = abs(boxes_cal(:,3) - boxes_gt(idx,3));
%         precise = sum(mindis <= 15);
%         recall = length(unique(idx(mindis <= 15)));
%         precise_ave = precise_ave + precise;
%         recall_ave = recall_ave + recall;
%         num_precise = num_precise + size(boxes_cal,1);
%         num_recall = num_recall + size(boxes_gt,1);
% %         TY_showboxes(imread(imdb{i}.image_path),boxes_cal(:,1:3),boxes_gt(:,1:3));
% %             TY_showprop(imread(imdb{i}.image_path),boxes_cal(:,1:3),boxes_gt(:,1:3));
% %             print('-dbmp', ['minutia_prop/' imdb{i}.image_path(end-10:end-4) '_p' imdb{i}.image_path(end-3:end)]);
% 
%     end
%     precise_ave = precise_ave/num_precise;
%     recall_ave = recall_ave/num_recall;
% 
%     recall_ave
%     precise_ave
    
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
            aboxes{i} = aboxes{i}(TY_nms(aboxes{i}, nms_overlap_thres), :);
        end       
    end
    aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
    fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
    if after_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x,1), after_nms_topN), :), aboxes, 'UniformOutput', false);
    end
end

function regions = make_roidb_regions(aboxes, images)
    regions.boxes = aboxes;
    regions.images = images;
end
