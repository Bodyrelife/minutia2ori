function minutia2ori()
clear all
% input_dir = 'D:\work\FVC2002\DB2_A_mnt\';
% output_dir = 'D:\work\FVC2002\DB2_A_m2o\';
% mkdir(output_dir);
% Files = dir('D:\work\FVC2002\DB2_A_mnt\*.mnt');
input_dir = 'D:\work\DataBase\data_finger\';
output_dir = 'D:\work\DataBase\latent_m2o\';
mkdir(output_dir);
Files = dir('D:\work\DataBase\data_finger\*.mnt');
for k = 1:length(Files)
    image_path = [input_dir,Files(k).name(1:end-4),'.bmp'];
    mnt_path = [input_dir,Files(k).name];
    mifile = fopen(mnt_path);
    fscanf(mifile,'%s',1)
    numm = fscanf(mifile,'%d',3);
    Minutia = zeros(numm(1),3);
    for i = 1:numm(1)
        Minutia(i,:) = fscanf(mifile,'%f',3);
%         fscanf(mifile,'%f',2);
    end
    fclose(mifile);
    Img = imread(image_path);
% imshow(Img);
% for k = 1:length(dataset.imdb_train{1})
%     Filename = dataset.imdb_train{1}{k}.image_path;
%     Img  = imread(Filename);
%     Minutia = dataset.imdb_train{1}{k}.boxes;
    Minutia=[200 200 pi/3; 400 400 pi*3/4; 50 300 pi/2; 150 250 pi*1.7]
%     Minutia=[200 200 pi/3]
%     Minutia=[200 200 pi/3; 400 400 pi/4]
    oimg_syn = minutia_to_orientation(Minutia);
    oimg_syn(size(oimg_syn,1)+1:size(Img,1),size(oimg_syn,2)+1:size(Img,2))=0;
    oimg_syn(size(Img,1)+1:end,:)=[];
    oimg_syn(:,size(Img,2)+1:end)=[];
    oimg_syn(1:max(0,min(Minutia(:,2))-16),:)=0;
    oimg_syn(:,1:max(0,min(Minutia(:,1))-16))=0;
    view_uint8_img(oimg_syn);
    [oimg,fimg,bwimg,eimg,enhimg]=fft_enhance_cubs(oimg_syn);
%     erode_img = imerode(enhimg,strel('disk',5));
    enhimg(enhimg>100)=255;
    enhimg(enhimg<=100)=0;
%     erode_img = imerode(enhimg,strel('disk',5));
%     oimg_syn(oimg_syn>100)=255;
%     oimg_syn(oimg_syn<=100)=0;
%     view_uint8_img(enhimg);
    [~,~,~,~,enhimg2]=fft_enhance_cubs(Img);
%     TY_showboxes(Img,Minutia);
%     TY_showboxes(enhimg,Minutia);
%     show_img = [Img,zeros(size(Img,1),10),enhimg];
%     show_Minutia = [Minutia;[Minutia(:,1)+10+size(Img,2),Minutia(:,2),Minutia(:,3)]];
    show_img = [Img,zeros(size(Img,1),10),enhimg2,zeros(size(Img,1),10),enhimg];
    show_Minutia = [Minutia;[Minutia(:,1)+10+size(Img,2),Minutia(:,2),Minutia(:,3)];[Minutia(:,1)+20+2*size(Img,2),Minutia(:,2),Minutia(:,3)]];
    TY_showboxes(show_img,show_Minutia);
    print('-dbmp',[output_dir,Files(k).name(1:end-4)]);
%     TY_showboxes(oimg_syn,Minutia);
%     view_orientation_image(oimg);
%     imwrite(enhimg,[outputDir Files(k).name],'bmp');
% end
end
end

function oimg_final = minutia_to_orientation(minutia)
    BLKSZ = 8;
%     minutia(:,3) = atan2(sin(minutia(:,3)),cos(minutia(:,3)));
    minW = max(0,min(minutia(:,1))-2*BLKSZ);
    minH = max(0,min(minutia(:,2))-2*BLKSZ);
%     minH = 0;
%     minW = 0;
    nWt = max(minutia(:,1))+2*BLKSZ;
    nHt = max(minutia(:,2))+2*BLKSZ;
    nBlkHt = floor(nHt/BLKSZ)+1;
    nBlkWt = floor(nWt/BLKSZ)+1;
    minBLKH = floor(minH/BLKSZ);
    minBLKW = floor(minW/BLKSZ);
    oimg = zeros(nBlkHt,nBlkWt);
%% orientation field
    for i = minBLKH:nBlkHt-1 
        for j = minBLKW:nBlkWt-1
            eudis = pointdis2([(i+0.5)*BLKSZ,(j+0.5)*BLKSZ],[minutia(:,2) minutia(:,1)]);
            sectors_loc = [minutia(:,2) minutia(:,1)]-repmat([(i+0.5)*BLKSZ,(j+0.5)*BLKSZ],size(minutia(:,1)));
            sectors_angle = angle(sectors_loc(:,1)+1i*sectors_loc(:,2));
            sectors_label = floor(sectors_angle/pi*4)+5; %-4:3 to 1:8
            u=0;
            v=0;
            for ss = 1:8
                [~,pick] = min(eudis(sectors_label==ss));
                if ~isempty(pick)
                    pick = find(sectors_label==ss,pick);
                    pick = pick(end);
                    u=u+cos(2*minutia(pick,3))/eudis(pick);
                    v=v+sin(2*minutia(pick,3))/eudis(pick);
                end
            end
            oimg(i+1,j+1) = atan2(v,u);
        end
    end
    oimg(oimg<0) = oimg(oimg<0)+2*pi;
    oimg = 0.5*oimg;
    for nn = 1:size(minutia,1)
        oimg(floor((minutia(nn,2)-1)/BLKSZ)+1,floor((minutia(nn,1)-1)/BLKSZ+1)) = minutia(nn,3);
    end
%     view_uint8_img(oimg);
    oimg = unwrap(oimg*2,pi,2)/2;
%     view_uint8_img(oimg);
    oimg = unwrap(oimg*2,pi,1)/2;
%     view_uint8_img(oimg);
%     view_orientation_image(oimg);
%     imshow(uint8((oimg-min(min(oimg)))/(max(max(oimg))-min(min(oimg)))*255));
%% gradient of continuous phase
    oimg = ones(nBlkHt,nBlkWt)*1.0472;
    Gradient = 2*pi*0.12*exp(1i*(oimg+pi/2));
    Phase_s = zeros(nHt,nWt);
    [xx,yy] = meshgrid(1:nHt,1:nWt);
    for nn = 1:size(minutia,1)
        Pimg = atan2((yy-minutia(nn,1)),(xx-minutia(nn,2))).';
%         Pimg(Pimg<0) = Pimg(Pimg<0)+2*pi;
        Phase_s = Phase_s-sign(cos(minutia(nn,3)-oimg(floor((minutia(nn,2)-1)/BLKSZ+1),floor((minutia(nn,1)-1)/BLKSZ)+1))+eps).*Pimg;            
    end
%     [Gradient_s1 , Gradient_s2] = gradient(Phase_s);
%     Gradient_s = Gradient_s1+1i*Gradient_s2;
%     Gradient_sb = zeros(nBlkHt,nBlkWt);
%     Gradient_sb = Gradient_s(BLKSZ/2:BLKSZ:end-BLKSZ/2,BLKSZ/2:BLKSZ:end-BLKSZ/2);


%     for i = minBLKH:nBlkHt-1 
%         for j = minBLKW:nBlkWt-1
%             Gradient_sb(i+1,j+1) = mean(mean(Gradient_s(i*BLKSZ+1:i*BLKSZ+BLKSZ,j*BLKSZ+1:j*BLKSZ+BLKSZ)));
%         end
%     end

%     Gradient_c = Gradient - 2*pi*0.12*Gradient_sb;
    Gradient_c = Gradient;
    
%     Gradient_ct = zeros(nBlkHt+2,nBlkWt+2);
%     Gradient_ct(2:nBlkHt+1,2:nBlkWt+1) = Gradient_c;
%     for nn = 1:size(minutia,1)
%         xxx=floor(minutia(nn,1)/BLKSZ)+1+1;
%         yyy=floor(minutia(nn,2)/BLKSZ)+1+1;
%         Gradient_c(xxx-1,yyy-1)=mean([Gradient_ct(xxx-1,yyy-1) Gradient_ct(xxx-1,yyy+1) Gradient_ct(xxx+1,yyy-1) Gradient_ct(xxx+1,yyy+1)]);
%     end 
    for i=1:3
        Gradient_c = smoothen_gradient_image(Gradient_c);
    end
%% continous phase
    phaseoff = zeros(nBlkHt,nBlkWt);
    queueoff = zeros(nBlkHt*nBlkWt,2);
    countp = zeros(nBlkHt,nBlkWt);
    countp(minBLKH+1,minBLKW+1) = 1;
    pointer = 1;
    pointend = 1;
    queueoff(1,:) = [minBLKH+1,minBLKW+1];
    temp = [-1 0;0 1;1 0;0 -1];
    while pointer <= pointend
        phase_sum = zeros(1,1);
        pkkk = 1;
        for i = 1:4
            if queueoff(pointer,1)+temp(i,1)>minBLKH && queueoff(pointer,1)+temp(i,1)<=nBlkHt && queueoff(pointer,2)+temp(i,2)>minBLKW && queueoff(pointer,2)+temp(i,2)<=nBlkWt
                if countp(queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2))==0
                    countp(queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2))=1;
                    pointend = pointend+1;
                    queueoff(pointend,:)=[queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2)];
               else
                    for j = 1:8
                        if temp(i,1)
                        phase_sum(pkkk) = imag(Gradient_c(queueoff(pointer,1),queueoff(pointer,2)))*(queueoff(pointer,1)*BLKSZ+temp(i,1)*BLKSZ)+...
                            real(Gradient_c(queueoff(pointer,1),queueoff(pointer,2)))*(queueoff(pointer,2)*BLKSZ-BLKSZ+j)-...
                            imag(Gradient_c(queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2)))*(queueoff(pointer,1)*BLKSZ+temp(i,1)*BLKSZ)-...
                            real(Gradient_c(queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2)))*(queueoff(pointer,2)*BLKSZ-BLKSZ+j)-...
                            phaseoff(queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2));
                        pkkk = pkkk+1;
                        else
                        phase_sum(pkkk) = imag(Gradient_c(queueoff(pointer,1),queueoff(pointer,2)))*(queueoff(pointer,1)*BLKSZ-BLKSZ+j)+...
                            real(Gradient_c(queueoff(pointer,1),queueoff(pointer,2)))*(queueoff(pointer,2)*BLKSZ+temp(i,2)*BLKSZ)-...
                            imag(Gradient_c(queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2)))*(queueoff(pointer,1)*BLKSZ-BLKSZ+j)-...
                            real(Gradient_c(queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2)))*(queueoff(pointer,2)*BLKSZ+temp(i,2)*BLKSZ)-...
                            phaseoff(queueoff(pointer,1)+temp(i,1),queueoff(pointer,2)+temp(i,2));
                        pkkk = pkkk+1; 
                        end
                    end
                end
            end
        end 
        tempphase = atan2(sum(sin(-phase_sum)),sum(cos(-phase_sum)));
%         tempphase = mean(phase_sum);
%         if tempphase<0
%             tempphase = tempphase+2*pi;
%         end
        phaseoff(queueoff(pointer,1),queueoff(pointer,2))=tempphase;
        pointer = pointer+1;
        if pointer==pointend
            queueoff(pointer+1,:)=queueoff(pointer,:);
        end
    end
%     phaseoff = zeros(nBlkHt,nBlkWt);
    Phase_c = zeros(nHt,nWt);
    for i=minH:nHt-1
        for j=minW:nWt-1
            nx = floor(i/BLKSZ)+1;
            ny = floor(j/BLKSZ)+1;
            Phase_c(i+1,j+1) = imag(Gradient_c(nx,ny))*(i+1)+real(Gradient_c(nx,ny))*(j+1)+phaseoff(nx,ny);
%             Phase_c(i+1,j+1) = real(Gradient_c(nx,ny))+imag(Gradient_c(nx,ny))+phaseoff(nx,ny);
        end
    end
%% reconstructed fingerprint
    Phase_final = Phase_s + Phase_c;
%     view_uint8_img(cos(Phase_s));
%     view_uint8_img(cos(Phase_c));
    oimg_final = (cos(Phase_final)+1)/2*255;
end

function [oimg,fimg,bwimg,eimg,enhimg]=fft_enhance_cubs(img)
    global NFFT;
    NFFT        =   32;     %size of FFT
    BLKSZ       =   12;     %size of the block
    OVRLP       =   6;      %size of overlap
    ALPHA       =   0.5;    %root filtering
    RMIN        =   3;      %min allowable ridge spacing
    RMAX        =   18;     %maximum allowable ridge spacing
    ESTRETCH    =   20;     %for contrast enhancement
    ETHRESH     =   6;      %threshold for the energy
    
    [nHt,nWt]   =   size(img);  
    img         =   double(img);    %convert to DOUBLE
    nBlkHt      =   floor((nHt-2*OVRLP)/BLKSZ);
    nBlkWt      =   floor((nWt-2*OVRLP)/BLKSZ);
    fftSrc      =   zeros(nBlkHt*nBlkWt,NFFT*NFFT); %stores FFT
    nWndSz      =   BLKSZ+2*OVRLP; %size of analysis window. 
    %-------------------------
    %allocate outputs
    %-------------------------
    oimg        =   zeros(nBlkHt,nBlkWt);
    fimg        =   zeros(nBlkHt,nBlkWt);
    bwimg       =   zeros(nBlkHt,nBlkWt);
    eimg        =   zeros(nBlkHt,nBlkWt);
    enhimg      =   zeros(nHt,nWt);
    
    %-------------------------
    %precomputations
    %-------------------------
    [x,y]       =   meshgrid(0:nWndSz-1,0:nWndSz-1);
    dMult       =   (-1).^(x+y); %used to center the FFT
    [x,y]       =   meshgrid(-NFFT/2:NFFT/2-1,-NFFT/2:NFFT/2-1);
    r           =   sqrt(x.^2+y.^2)+eps;
    th          =   atan2(y,x);
    th(th<0)    =   th(th<0)+pi;
    w           =   raised_cosine_window(BLKSZ,OVRLP); %spectral window

    %-------------------------
    %Load filters
    %-------------------------
    load angular_filters_pi_4;   %now angf_pi_4 has filter coefficients
    angf_pi_4 = angf;
    load angular_filters_pi_2;   %now angf_pi_2 has filter coefficients
    angf_pi_2 = angf;
    %-------------------------
    %Bandpass filter
    %-------------------------
    FLOW        =   NFFT/RMAX;
    FHIGH       =   NFFT/RMIN;    
    dRLow       =   1./(1+(r/FHIGH).^4);    %low pass butterworth filter
    dRHigh      =   1./(1+(FLOW./r).^4);    %high pass butterworth filter
    dBPass      =   dRLow.*dRHigh;          %bandpass
    for i = 0:nBlkHt-1
        nRow = i*BLKSZ+OVRLP+1;  
        for j = 0:nBlkWt-1
            nCol = j*BLKSZ+OVRLP+1;
            %extract local block
            blk     =   img(nRow-OVRLP:nRow+BLKSZ+OVRLP-1,nCol-OVRLP:nCol+BLKSZ+OVRLP-1);
            %remove dc
            dAvg    =   sum(sum(blk))/(nWndSz*nWndSz);
            blk     =   blk-dAvg;   %remove DC content
            blk     =   blk.*w;     %multiply by spectral window
            %--------------------------
            %do pre filtering
            %--------------------------
            blkfft  =   fft2(blk.*dMult,NFFT,NFFT);
            blkfft  =   blkfft.*dBPass;             %band pass filtering
            dEnergy =   abs(blkfft).^2;
            blkfft  =   blkfft.*sqrt(dEnergy);      %root filtering(for diffusion)
            fftSrc(nBlkWt*i+j+1,:) = transpose(blkfft(:));
            dEnergy =   abs(blkfft).^2;             %----REDUCE THIS COMPUTATION----
            %--------------------------
            %compute statistics
            %--------------------------
            dTotal          =   sum(sum(dEnergy))/(NFFT*NFFT);
            fimg(i+1,j+1)   =   NFFT/(compute_mean_frequency(dEnergy,r)+eps); %ridge separation
            eimg(i+1,j+1)   =   log(dTotal+eps);                        %used for segmentation
            oimg(i+1,j+1)   =   compute_mean_angle(dEnergy,th);         %ridge angle
        end;%for j
    end;%for i
    [x,y]       =   meshgrid(-NFFT/2:NFFT/2-1,-NFFT/2:NFFT/2-1);
    dMult       =   (-1).^(x+y); %used to center the FFT
    for i = 1:3
        oimg = smoothen_orientation_image(oimg);            %smoothen orientation image
    end;
    fimg    =   smoothen_frequency_image(fimg,RMIN,RMAX,5); %diffuse frequency image
    cimg    =   compute_coherence(oimg);                    %coherence image for bandwidth
    bwimg   =   get_angular_bw_image(cimg);                 %QUANTIZED bandwidth image
    for i = 0:nBlkHt-1
        for j = 0:nBlkWt-1
            nRow = i*BLKSZ+OVRLP+1;            
            nCol = j*BLKSZ+OVRLP+1;
            %--------------------------
            %apply the filters
            %--------------------------
            blkfft  =   reshape(transpose(fftSrc(nBlkWt*i+j+1,:)),NFFT,NFFT);
            %--------------------------
            %reconstruction
            %--------------------------
            af      =   get_angular_filter(oimg(i+1,j+1),bwimg(i+1,j+1),angf_pi_4,angf_pi_2);
            blkfft  =   blkfft.*(af); 
            blk     =   real(ifft2(blkfft).*dMult);
            enhimg(nRow:nRow+BLKSZ-1,nCol:nCol+BLKSZ-1)=blk(OVRLP+1:OVRLP+BLKSZ,OVRLP+1:OVRLP+BLKSZ);
        end;%for j
    end;%for i
    enhimg =sqrt(abs(enhimg)).*sign(enhimg);
    mx     =max(max(enhimg));
    mn     =min(min(enhimg));
    enhimg =uint8((enhimg-mn)/(mx-mn)*254+1);
end    

function y = raised_cosine(nBlkSz,nOvrlp)
    nWndSz  =   (nBlkSz+2*nOvrlp);
    x       =   abs(-nWndSz/2:nWndSz/2-1);
    y       =   0.5*(cos(pi*(x-nBlkSz/2)/nOvrlp)+1);
    y(abs(x)<nBlkSz/2)=1;
end

function w = raised_cosine_window(blksz,ovrlp)
    y = raised_cosine(blksz,ovrlp);
    w = y(:)*y(:)';
end

function r = get_angular_filter(t0,bw,angf_pi_4,angf_pi_2)
    global NFFT;
    TSTEPS = size(angf_pi_4,2);
    DELTAT = pi/TSTEPS;
    %get the closest filter
    i      = floor((t0+DELTAT/2)/DELTAT);
    i      = mod(i,TSTEPS)+1; 
    if(bw == pi/4)
        r      = reshape(angf_pi_4(:,i),NFFT,NFFT)';
    elseif(bw == pi/2)
        r      = reshape(angf_pi_2(:,i),NFFT,NFFT)';
    else
        r      = ones(NFFT,NFFT);
    end;
end

function bwimg = get_angular_bw_image(c)
    bwimg   =   zeros(size(c));
    bwimg(:,:)    = pi/2;                       %med bw
    bwimg(c<=0.7) = pi;                         %high bw
    bwimg(c>=0.9) = pi/4;                       %low bw
end

function mth = compute_mean_angle(dEnergy,th)
    global NFFT;
    sth         =   sin(2*th);
    cth         =   cos(2*th);
    num         =   sum(sum(dEnergy.*sth));
    den         =   sum(sum(dEnergy.*cth));
    mth         =   0.5*atan2(num,den);
    if(mth <0)
        mth = mth+pi;
    end;
end

function mr = compute_mean_frequency(dEnergy,r)
    global NFFT;
    num         =   sum(sum(dEnergy.*r));
    den         =   sum(sum(dEnergy));
    mr          =   num/(den+eps);
end

function [cimg] = compute_coherence(oimg)
    [h,w]   =   size(oimg);
    cimg    =   zeros(h,w);
    N       =   2;
    %---------------
    %pad the image
    %---------------
    oimg    =   [flipud(oimg(1:N,:));oimg;flipud(oimg(h-N+1:h,:))]; %pad the rows
    oimg    =   [fliplr(oimg(:,1:N)),oimg,fliplr(oimg(:,w-N+1:w))]; %pad the cols
    %compute coherence
    for i=N+1:h+N
        for j = N+1:w+N
            th  = oimg(i,j);
            blk = oimg(i-N:i+N,j-N:j+N);
            cimg(i-N,j-N)=sum(sum(abs(cos(blk-th))))/((2*N+1).^2);
        end;
    end;
end

function noimg = smoothen_orientation_image(oimg)
    %---------------------------
    %smoothen the image
    %---------------------------
    gx      =   cos(2*oimg);
    gy      =   sin(2*oimg);
    
    msk     =   fspecial('gaussian',5);
    gfx     =   imfilter(gx,msk,'symmetric','same');
    gfy     =   imfilter(gy,msk,'symmetric','same');
    noimg   =   atan2(gfy,gfx);
    noimg(noimg<0) = noimg(noimg<0)+2*pi;
    noimg   =   0.5*noimg;
end

function nfimg = smoothen_frequency_image(fimg,RLOW,RHIGH,diff_cycles)
    valid_nbrs  =   3; %uses only pixels with more then valid_nbrs for diffusion
    [ht,wt]     =   size(fimg);
    nfimg       =   fimg;
    N           =   1;
    
    %---------------------------------
    %perform diffusion
    %---------------------------------
    h           =   fspecial('gaussian',2*N+1);
    cycles      =   0;
    invalid_cnt = sum(sum(fimg<RLOW | fimg>RHIGH));
    while((invalid_cnt>0 &cycles < diff_cycles) | cycles < diff_cycles)
        %---------------
        %pad the image
        %---------------
        fimg    =   [flipud(fimg(1:N,:));fimg;flipud(fimg(ht-N+1:ht,:))]; %pad the rows
        fimg    =   [fliplr(fimg(:,1:N)),fimg,fliplr(fimg(:,wt-N+1:wt))]; %pad the cols
        %---------------
        %perform diffusion
        %---------------
        for i=N+1:ht+N
         for j = N+1:wt+N
                blk = fimg(i-N:i+N,j-N:j+N);
                msk = (blk>=RLOW & blk<=RHIGH);
                if(sum(sum(msk))>=valid_nbrs)
                    blk           =blk.*msk;
                    nfimg(i-N,j-N)=sum(sum(blk.*h))/sum(sum(h.*msk));
                else
                    nfimg(i-N,j-N)=-1; %invalid value
                end;
         end;
        end;
        %---------------
        %prepare for next iteration
        %---------------
        fimg        =   nfimg;
        invalid_cnt =   sum(sum(fimg<RLOW | fimg>RHIGH));
        cycles      =   cycles+1;
    end;
    cycles;
end

function noimg = smoothen_gradient_image(oimg)
    %---------------------------
    %smoothen the image
    %---------------------------
%     gfx = medfilt2(real(oimg),[10,10]);
%     gfy = medfilt2(imag(oimg),[10,10]);
    msk     =   fspecial('gaussian',5);
    gfx     =   imfilter(real(oimg),msk,'symmetric','same');
    gfy     =   imfilter(imag(oimg),msk,'symmetric','same');
    noimg   =   gfx+1i*gfy;
end