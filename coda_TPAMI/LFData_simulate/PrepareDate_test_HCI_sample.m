%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate test data from HCI & Stanford dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===> test_DatasetName.h5 (in python)
% uint8 0-255
%  ['GT_RGB']   [N,ah,aw,3,h,w]
%  ['GT_Y']     [N,ah,aw,h,w]
%  ['LR_ycbcr_up'] [N,ah,aw,3,h,w]
%  ['LR']       [N,ah,aw,h/scale,w/scale]
% ===> in matlab (inverse)
%  [w,h,3,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;close all;

%% params
dataset = 'HCI_5x5';
 
%% path
folder = 'Dataset\LF_Dataset\Dataset_HCI';
savepath = sprintf('test_%s.h5',dataset);


listname = './HCI_test.txt';

f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 

%% params

H = 512;
W = 512;
ah = 9;
aw = 9;

%% initialization
GT_y   = zeros(H, W, 5, 5, 1, 'uint8');
LR_ycbcr_2     = zeros(H/2, W/2, 3, 5, 5, 1, 'uint8');
LR_ycbcr_4     = zeros(H/4, W/4, 3, 5, 5, 1, 'uint8');
LR_ycbcr_8     = zeros(H/8, W/8, 3, 5, 5, 1, 'uint8');
LR_ycbcr_16     = zeros(H/16, W/16, 3, 5, 5, 1, 'uint8');

count = 0;
margain = 0;

%% generate data
for k = 1:size(list,1)
    lfname = list{k};
    lf_path = fullfile(folder,lfname);
    disp(lf_path);
    
    % read SAIs ==> lf[h,w,ah,aw]
%     lf_rgb = zeros(H,W,3,ah,aw,'uint8');
    lf_ycbcr = zeros(H,W,3,ah,aw,'uint8');

    for v = 1 : ah
        for u = 1 : aw
            ind = (v-1)*9+(u-1);
            imgname = strcat('input_Cam',num2str(ind,'%03d'),'.png');
            sub = imread(fullfile(lf_path,imgname));
%             lf_rgb(:,:,:,v,u) = sub;
            lf_ycbcr(:,:,:,v,u) = rgb2ycbcr(sub);                
        end
    end        

    % get data
    gt_ycbcr = lf_ycbcr(:,:,:,1:2:9,1:2:9);
    im_lr_ycbcr_2 = imresize(gt_ycbcr, 1/2, 'bicubic');
    im_lr_ycbcr_4 = imresize(gt_ycbcr, 1/4, 'bicubic');
    im_lr_ycbcr_8 = imresize(gt_ycbcr, 1/8, 'bicubic');
    im_lr_ycbcr_16 = imresize(gt_ycbcr, 1/16, 'bicubic');
            
    count = count+1;
    GT_y(:,:,:,:,count) = gt_ycbcr(:,:,1,:,:);
    LR_ycbcr_2(:,:,:,:,:,count) = im_lr_ycbcr_2;
    LR_ycbcr_4(:,:,:,:,:,count) = im_lr_ycbcr_4;
    LR_ycbcr_8(:,:,:,:,:,count) = im_lr_ycbcr_8;
    LR_ycbcr_16(:,:,:,:,:,count) = im_lr_ycbcr_16;
end

 
GT_y = permute(GT_y,[2,1,4,3,5]);  %[h,w,3,ah,aw,n]--->[w,h,3,aw,ah,n] 
LR_ycbcr_2 = permute(LR_ycbcr_2,[2,1,3,5,4,6]); 
LR_ycbcr_4 = permute(LR_ycbcr_4,[2,1,3,5,4,6]); 
LR_ycbcr_8 = permute(LR_ycbcr_8,[2,1,3,5,4,6]); 
LR_ycbcr_16= permute(LR_ycbcr_16,[2,1,3,5,4,6]);

%% save data
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath,'/GT_y',size(GT_y),'Datatype','uint8');
h5create(savepath,'/LR_ycbcr_2',size(LR_ycbcr_2),'Datatype','uint8');
h5create(savepath,'/LR_ycbcr_4',size(LR_ycbcr_4),'Datatype','uint8');
h5create(savepath,'/LR_ycbcr_8',size(LR_ycbcr_8),'Datatype','uint8');
h5create(savepath,'/LR_ycbcr_16',size(LR_ycbcr_16),'Datatype','uint8');
 
 
h5write(savepath, '/GT_y', GT_y);
h5write(savepath, '/LR_ycbcr_2', LR_ycbcr_2);
h5write(savepath, '/LR_ycbcr_4', LR_ycbcr_4);
h5write(savepath, '/LR_ycbcr_8', LR_ycbcr_8);
h5write(savepath, '/LR_ycbcr_16', LR_ycbcr_16);
  


h5disp(savepath);