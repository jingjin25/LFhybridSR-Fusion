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
dataset = 'HCI';
scale = 4;

%% path
folder = ['../../LFData/',dataset];
savepath = sprintf('test_%s_x%d.h5',dataset,scale);


listname = 'HCI_test.txt';

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
GT_rgb = zeros(H, W, 3, ah, aw, 1, 'uint8');
GT_y   = zeros(H, W, ah, aw, 1, 'uint8');
LR_up_ycbcr = zeros(H, W, 3, ah, aw, 1, 'uint8');
LR     = zeros(H/scale, W/scale, ah, aw, 1, 'uint8');
HR     = zeros(H, W, 1, 1, 1, 'uint8');
HR_2     = zeros(H/2, W/2, 1, 1, 1, 'uint8');
HR_4     = zeros(H/4, W/4, 1, 1, 1, 'uint8');

count = 0;
margain = 0;

%% generate data
for k = 1:size(list,1)
    lfname = list{k};
    lf_path = fullfile(folder,lfname);
    disp(lf_path);
    % read SAIs ==> lf[h,w,ah,aw]
    lf_rgb = zeros(H,W,3,ah,aw,'uint8');
    lf_ycbcr = zeros(H,W,3,ah,aw,'uint8');

    % HCI dataset
    for v = 1 : ah
        for u = 1 : aw
            ind = (v-1)*9+(u-1);
            imgname = strcat('input_Cam',num2str(ind,'%03d'),'.png');
            sub = imread(fullfile(lf_path,imgname));
            lf_rgb(:,:,:,v,u) = sub;
            lf_ycbcr(:,:,:,v,u) = rgb2ycbcr(sub);                
        end
    end        

    % get data

    im_gt_rgb = lf_rgb;
    im_gt_y = squeeze(lf_ycbcr(:,:,1,:,:));
    
    im_hr_y = im_gt_y(:,:,ceil(ah/2),ceil(aw/2));
    im_hr_y_2 = imresize(im_hr_y, 1/2, 'bicubic');
    im_hr_y_4 = imresize(im_hr_y, 1/4, 'bicubic');   
    
    lr_ycbcr = imresize(lf_ycbcr,1/scale,'bicubic');
    im_lr_y = squeeze(lr_ycbcr(:,:,1,:,:));

    im_lr_up_ycbcr = imresize(lr_ycbcr,scale,'bicubic');


    count = count+1;
    GT_rgb(:,:,:,:,:,count) = im_gt_rgb;
    GT_y(:,:,:,:,count) = im_gt_y;
    LR_up_ycbcr(:,:,:,:,:,count) = im_lr_up_ycbcr;    
    LR(:,:,:,:,count) = im_lr_y; 
    HR(:,:,:,:,count) = im_hr_y;
    HR_2(:,:,:,:,count) = im_hr_y_2;
    HR_4(:,:,:,:,count) = im_hr_y_4;
end

GT_rgb = permute(GT_rgb,[2,1,3,5,4,6]);   %[h,w,3,ah,aw,n]--->[w,h,3,aw,ah,n] 
GT_y = permute(GT_y,[2,1,4,3,5]);
LR_up_ycbcr = permute(LR_up_ycbcr,[2,1,3,5,4,6]); 
LR = permute(LR,[2,1,4,3,5]);
HR = permute(HR,[2,1,4,3,5]);
HR_2 = permute(HR_2,[2,1,4,3,5]);
HR_4 = permute(HR_4,[2,1,4,3,5]);

%% save data
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath,'/GT_rgb',size(GT_rgb),'Datatype','uint8');
h5create(savepath,'/GT_y',size(GT_y),'Datatype','uint8');
h5create(savepath,'/LR_up_ycbcr',size(LR_up_ycbcr),'Datatype','uint8');
h5create(savepath,'/LR',size(LR),'Datatype','uint8');
h5create(savepath,'/HR',size(HR),'Datatype','uint8');
h5create(savepath,'/HR_2',size(HR_2),'Datatype','uint8');
h5create(savepath,'/HR_4',size(HR_4),'Datatype','uint8');

h5write(savepath, '/GT_rgb', GT_rgb);
h5write(savepath, '/GT_y', GT_y);
h5write(savepath, '/LR_up_ycbcr', LR_up_ycbcr);
h5write(savepath, '/LR', LR);
h5write(savepath, '/HR', HR);
h5write(savepath, '/HR_2', HR_2);
h5write(savepath, '/HR_4', HR_4);


h5disp(savepath);