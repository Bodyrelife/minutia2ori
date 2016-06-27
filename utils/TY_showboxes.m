function TY_showboxes(pic, pos, pos_t, label)
%    pos(:,3) = 0;pos_t(:,3) = 0;
    if nargin == 2
        pos_t = pos;
        label = ones(length(pos));
    else nargin == 3
        label = compare(pos, pos_t, 0, size(pic));
    end
    Length = 20;
    Arrow = [0, Length, 4 * Length / 5, Length, 4 * Length / 5; 0, 0, Length / 5, 0, -Length / 5];
    figure(1);
    hold off
    imshow(pic);
    
    for i = 1:size(pos_t, 1)
        hold on
        alpha = pos_t(i, 3);
        Rotate = [cos(alpha), sin(alpha); -sin(alpha), cos(alpha)];
        temp = Rotate' * Arrow + repmat(pos_t(i, 1:2)', [1, size(Arrow, 2)]);
        plot(temp(1, :), temp(2, :), 'LineWidth', 2, 'Color', 'g');
    end
    
    for i = 1:size(pos, 1)
        hold on
        if label(i) == 1
            color = 'b';
        else
            color = 'r';
        end
        alpha = pos(i, 3);
        Rotate = [cos(alpha), sin(alpha); -sin(alpha), cos(alpha)];
        temp = Rotate' * Arrow + repmat(pos(i, 1:2)', [1, size(Arrow, 2)]);
        plot(temp(1, :), temp(2, :), 'LineWidth', 2, 'Color', color);
    end
    set(gcf, 'PaperPositionMode', 'auto');
%     print(1, '-dbmp', save);
end

function label=compare(test, groundtruth, ifmatch, T)
    R=15;
    T_ang=pi/6;

    data_minutia_s = round(groundtruth);
    data_minutia_s(:, 3) = groundtruth(:, 3) ;    
    num_minutia_s = size(data_minutia_s,1);

    data_minutia_t = round(test);
    data_minutia_t(:, 3) = test(:, 3) ;    
    num_minutia_t = size(data_minutia_t,1);
    
    H_s = T(1); W_s = T(2);
    
    I_minutia_s=zeros(H_s,W_s);
    I_minutia_t=zeros(H_s,W_s);
    ang_minutia_s=zeros(H_s,W_s);
    ang_minutia_t=zeros(H_s,W_s);

    for i=1:num_minutia_s;
        if data_minutia_s(i, 2) > 0    
            I_minutia_s(data_minutia_s(i,2),data_minutia_s(i,1))=1;
            ang_minutia_s(data_minutia_s(i,2),data_minutia_s(i,1))=data_minutia_s(i,3);
        else
            disp(-1);
        end
    end
    for i=1:num_minutia_t;
        if data_minutia_t(i, 2) > 0    
            I_minutia_t(data_minutia_t(i,2),data_minutia_t(i,1))=1;
            ang_minutia_t(data_minutia_t(i,2),data_minutia_t(i,1))=data_minutia_t(i,3);
        else
            disp(--1);
        end
    end

    label = zeros(num_minutia_t, 1);
    for i=1:num_minutia_t;
        x0=data_minutia_t(i,2);
        y0=data_minutia_t(i,1);
        x1=max(1,x0-R);
        x2=min(H_s,x0+R);
        y1=max(1,y0-R);
        y2=min(W_s,y0+R);
        dis_min=R;
        x_min=0;
        y_min=0;
        for x=x1:x2;
            for y=y1:y2;
                if (I_minutia_s(x,y)==1 && ifmatch == 1) || (I_minutia_s(x,y)>0 && ifmatch == 0)
                    dis=sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
                    if (dis<=R)
                        diff_ang=ang_minutia_t(x0,y0)-ang_minutia_s(x,y);
                        diff_ang_abs=min(abs(diff_ang),2*pi-abs(diff_ang));
                        if (diff_ang_abs<=T_ang)
                            if (dis<dis_min)
                                dis_min=dis;
                                x_min=x;
                                y_min=y;
                            end
                        end
                    end
                end
            end
        end
        if (x_min>0)
            I_minutia_t(x0,y0)=2;
            I_minutia_s(x_min,y_min)=2;
            label(i)=1;
        else
            label(i)=0;
        end
    end
end