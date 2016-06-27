function TY_generate_mnt()
%%
    THRESH = 0.5;
    MOUT_path = '/media/ssd2/tangy/nist27/MOUT';
    load data_minutia_mix;
    dataset = dataset_test_nist27;
    imdb = dataset.imdb_test{1};
    cache_name ='faster_rcnn_274205_VGG16_3w4w6w8w_top-1_nms0_6_top200_stage2_fast_rcnn';    
    cache_dir = fullfile(pwd, 'output', 'fast_rcnn_cachedir', cache_name, 'finger');
    
    num_classes = 1;
    min_img=3;
    
    aboxes = cell(num_classes, 1);
    for i = 1:num_classes
     load(fullfile(cache_dir, '1_boxes_finger'));
     aboxes{i} = boxes;
    end 
    aboxes{1} = boxes_filter(aboxes{1}, -1, 0.3, -1, 1);        
    precise_ave = 0;
    recall_ave = 0;
    num_precise = 0;
    num_recall = 0;
%     mkdir('minutia_out_nist_0.3');
    for i= 1:length(aboxes{1})
        boxes_cal = aboxes{1}{i};
        boxes_cal = boxes_cal(boxes_cal(:,end)>THRESH,:);
        if size(boxes_cal,1)<min_img
%             [~ , sort_s] = sort(boxes_cal(:,end),'descend');
            boxes_cal = [boxes_cal;aboxes{1}{i}(size(boxes_cal,1)+1:min_img,:)];
        end
        gt = imdb{i}.gt;
        boxes_gt = imdb{i}.boxes(gt,:);
        if isempty(boxes_cal) || isempty(boxes_gt)
            disp('empty');
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
%         if opts.iter==2
%             TY_showboxes(imread(imdb{i}.image_path),boxes_cal(:,1:3),boxes_gt(:,1:3));
%             print('-dbmp', ['minutia_out_nist_0.3/' imdb{i}.image_path(end-10:end)]);
%         end

        
    end
    precise_ave = precise_ave/num_precise;
    recall_ave = recall_ave/num_recall;
    recall_ave
    precise_ave
    
    MOUT_path = sprintf('%s_%.2f_%.2f/',MOUT_path,recall_ave,precise_ave);
    mkdir(MOUT_path);
    for i= 1:length(aboxes{1})
        boxes_cal = aboxes{1}{i};
        boxes_cal = boxes_cal(boxes_cal(:,end)>THRESH,:);
        if size(boxes_cal,1)<min_img
%             [~ , sort_s] = sort(boxes_cal(:,end),'descend');
            boxes_cal = [boxes_cal;aboxes{1}{i}(size(boxes_cal,1)+1:min_img,:)];
        end
        %% MNT_out
        Mout=fopen([MOUT_path, dataset.imdb_test{1}{i}.image_path(end-10:end-3), 'mnt'],'w+');
        fprintf(Mout,'%s\n',dataset.imdb_test{1}{i}.image_path(end-10:end));
        fprintf(Mout,'%d %d %d\n',size(boxes_cal,1),dataset.imdb_test{1}{i}.W,dataset.imdb_test{1}{i}.H);
        % X,Y,O,score
        for j=1:size(boxes_cal,1)
            fprintf(Mout,'%d %d %f %f\n',int16(boxes_cal(j,1)),int16(boxes_cal(j,2)),mod(boxes_cal(j,3),2*pi),boxes_cal(j,5));
        end
        fclose(Mout);
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