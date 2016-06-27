clear all
% Add caffe/matlab to you Matlab search PATH to use matcaffe
addpath('+caffe');
% Set caffe mode
caffe.set_mode_cpu();
model = '/home/tangy/caffe-master/examples/minutia/lenet_deployfvc.prototxt';
weights = '/home/tangy/caffe-master/examples/minutia/snapshot/net612_fvcfine_0.002_iter_100000.caffemodel';
% weights = '/home/tangy/caffe-master/examples/minutia/snapshot/netf612_batch512_alldata_STFT_iter_120000.caffemodel';
net = caffe.Net(model, weights, 'test'); % create net and load weights

% database ='/media/tangy/22CAAD8ACAAD5AB5/work/pic100_STFT/';
database='/media/tangy/22CAAD8ACAAD5AB5/work/FVC2002/DB2_A/';
database2='/media/tangy/22CAAD8ACAAD5AB5/work/FVC2002/DB2_A_Outp16_0.85/';
mkdir(database2);
% load([database, 'config_trainingset.mat']);
Files=dir([database '*.bmp']);
% FIles_mnt=dir([database '*.mnt']);
% parpool;
for i = 1:size(Files,1)
%     I = imread([database, 'data_pic/', Names{i}, '.bmp']);
    I = imread([database,  Files(i).name]);
    Mfid=fopen([database, Files(i).name(1:end-3), 'mnt']);
    Mout=fopen([database2, Files(i).name(1:end-3), 'mnt'],'a');
    filename=fscanf(Mfid,'%s',1);
    num_s=fscanf(Mfid,'%d',1);
    W=fscanf(Mfid,'%d',1);
    H=fscanf(Mfid,'%d',1);
    fprintf(Mout,'%s\n',filename);
    position=zeros(num_s,3);
    flagp=zeros(num_s,1);
    num_t=0;
    for j=1:num_s
        position(j,1)=fscanf(Mfid,'%d',1);
        position(j,2)=fscanf(Mfid,'%d',1);
        position(j,3)=fscanf(Mfid,'%f',1);
        useless=fscanf(Mfid,'%d',1);
        useless=fscanf(Mfid,'%f',1);
        cut = GetDetailPart(I, position(j,1), position(j,2), position(j,3)*180/pi);
    %use CNN here
        scores=useCNN(net,single(cut{1})./255);
        flagp(j)=(scores(1,:)<0.85);
    end
    fprintf(Mout,'%d %d %d\n',sum(flagp),W,H);
    for j=1:num_s
        if flagp(j)
            fprintf(Mout,'%d %d %d\n',position(j,1),position(j,2),position(j,3));
        end
    end
    fclose(Mfid);
    fclose(Mout);
end
% delete(gcp);