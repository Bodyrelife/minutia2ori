%������ͼƬ��ϸ�ڵ�����ͷ��򣬽�ȡϸ�ڵ���ΧͼƬ�ĳ���
%OriginPic������ͼƬ
%CoorX��CoorY��ϸ�ڵ�����ͼƬ�е����
%Angle��ϸ�ڵ㷽��
%
% o--------- X
% |\ ) Angle
% | \
% |  \
% |   \
% Y
%%
function [DetailPart, invalidnum] = GetDetailPart(OriginPic, CoorX, CoorY, Angle)
    Range = 40;
    %����ͼƬ
    [m, n] = size(OriginPic); 
    EnlargePic = uint8(zeros(m + 2 * Range, n + 2 * Range));
    EnlargePic(Range + 1:m + Range, Range + 1:n + Range) = OriginPic(:, :);
    CoorX = CoorX + Range;
    CoorY = CoorY + Range;
    %������ȡϸ�ڵ�ͼƬ
    DetailPart = cell(size(CoorX, 1), 1);
    invalidnum = 0;
    for i = 1:size(CoorX, 1)
        if CoorX(i) < 1 + Range || CoorX(i) > n + Range || CoorY(i) < 1 + Range || CoorY(i) > m + Range || (CoorX(i) == Range && CoorY(i) == Range)
            DetailPart{i} = -1;
            invalidnum = invalidnum + 1;
            continue;
        end
        TempRot = imrotate(EnlargePic(CoorY(i) - Range:CoorY(i) + Range, CoorX(i) - Range:CoorX(i) + Range), Angle(i), 'bicubic', 'crop');
        DetailPart{i} = TempRot(1 + Range / 2:3 * Range / 2, 1 + Range / 2:3 * Range / 2);%Range * Range
        temp = double(DetailPart{i});
%         if norm(temp - mean(temp(:))) == 0
%             DetailPart{i} = -1;
%             invalidnum = invalidnum + 1;
%         end 
    end
end