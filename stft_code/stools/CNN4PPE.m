%%
clear
% load net612/testet612_batch512_newalldata_steplr2_iter_600000label_cal
% load net612/testlabel_grt
cpath='/home/tangy/caffe-master/python/';
load ([cpath , 'net612/test0000label_cal.mat']);
load ([cpath , 'net612/testlabel_grt.mat'])
z=(label_grt(:,1)=='1').';
l=round(1-label_cal);
truem=sum(l==z&l==1);
findm=sum(l);
% totalm=25005-2943;
% totalm=10810;ָ��
% totalm=27766;%palm
% totalm=10810;%finger
totalm=2647;%nist

file_res=fopen('PPR_record.txt','a');
fprintf(file_res,'er_test: %f er_recall: %f er_precise: %f P+R: %f %s nist dr_50\n',sum(l==z)/length(l),truem/totalm,truem/findm,truem/totalm+truem/findm,datestr(now));
fclose(file_res);

%%
recall=zeros(999,1);
precise=zeros(999,1);
for i=1:999
    l=round(1-label_cal-(i-500)/999);
    truem=sum(l==z&l==1);
    findm=sum(l);
    recall(i)=truem/totalm;
    precise(i)=truem/findm; 
end
[maxpr,idxpr]=max(recall+precise)
plot(recall,precise);
xlabel('recall');ylabel('precise');
legend(num2str([maxpr,idxpr]));
% title('1:10-cnnstft-fine');
% print -dbmp ../../ROC/1:10-cnnstft-fine.bmp
title('dr-50-nist');
print -dbmp ../../ROC/dr-50-nist.bmp
%%
% load valori2_iter_200000label_cal
% load valorilabel_grt
% z=str2num(label_grt);
% label_cal=label_cal.';
% label_calp=label_cal+1;
% label_calm=label_cal-1;
% label_calp2=label_cal+2;
% label_calm2=label_cal-2;
% num=sum(label_cal==z)+sum(label_calp==z)+sum(label_calm==z);
% top3=num/length(label_cal);