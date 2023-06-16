%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate training data  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUT
% In python:
% input_copy an*h*w + 1*h*w
% input_warp an*h/s*w/s + 1*h*w
% GT an*h*w
% ===> matlab
% LR w/s*h/s*aw*ah*N
% LR_up w*h*aw*ah*N
% HR w*h*1*1*N
% label w*h*aw*ah*N
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;close all;

%% params
dataset = 'real_rgb_resize';
scale = 8;

%% path
data_folder = 'dataset_wang';
savepath = sprintf('test_%s_x%d.h5',dataset,scale);

listname = 'list/real_test.txt';

f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 


%% params

W = floor(2846/2/64)*64 ;
H = floor(1729/2/64)*64 ;

% W_lr = W_hr / 8;
% H_lr = H_hr / 8;

ah = 3;
aw = 3;

%% initialization
% GT_rgb = zeros(H, W, 3, ah, aw, 1, 'uint8');
% GT_y   = zeros(H, W, ah, aw, 1, 'uint8');
% LR_up_ycbcr = zeros(H, W, 3, ah, aw, 1, 'uint8');
LR     = zeros(H/scale, W/scale, 3, ah, aw, 1, 'uint8');
HR     = zeros(H, W, 3, 1, 1, 1, 'uint8');
HR_2     = zeros(H/2, W/2, 3, 1, 1, 1, 'uint8');
HR_4     = zeros(H/4, W/4, 3, 1, 1, 1, 'uint8');

count = 0;
margain = 0;

%% generate data
for k =   2:size(list,1) %1:9 %10:size(list,1)
    lfname = list{k};
    lf_path = fullfile(data_folder,lfname);
    disp(lf_path);
 
    lf_lr_rgb = zeros(H/scale,W/scale,3,ah,aw,'uint8');
%     lf_lr_y = zeros(H/scale, W/scale, ah, aw, 'uint8');
%     lf_lr_up_ycbcr = zeros(H,W,3,ah,aw,'uint8');
    for v = 1 : ah
        for u = 1 : aw
            imgname = sprintf('%s/%s/im_t%03d_s%03d_GT.png',data_folder,lfname,98+v,98+u);
            sub_rgb = imread(imgname);
            if u~=2 || v~=2 %side view
                sub_rgb = imresize(sub_rgb, [H/scale, W/scale], 'bicubic');
            else % central view
                sub_rgb = imresize(sub_rgb, [H, W], 'bicubic'); 
                sub_rgb = imresize(sub_rgb, 1/scale, 'bicubic'); 
%                 sub_rgb = imresize(sub_rgb, [H/scale, W/scale], 'bicubic');  
            end 
            lf_lr_rgb(:,:,:,v,u) = sub_rgb;
%             sub_ycbcr = rgb2ycbcr(sub_rgb);
%             lf_lr_up_ycbcr(:,:,:,v,u) = imresize(sub_ycbcr, scale, 'bicubic');
%             lf_lr_y(:,:,v,u) = sub_ycbcr(:,:,1);
        end
    end
   
    hr_name = sprintf('%s/%s/im_t%03d_s%03d_GT.png',data_folder,lfname,100,100);
    hr_rgb = imread(hr_name);
    hr_rgb = imresize(hr_rgb, [H, W], 'bicubic');
%     hr_ycbcr = rgb2ycbcr(hr_rgb);
%     hr_y = squeeze(hr_ycbcr(:,:,1));
%     hr_y_2 = imresize(hr_y, 1/2, 'bicubic');
%     hr_y_4 = imresize(hr_y, 1/4, 'bicubic');
    hr_rgb_2 = imresize(hr_rgb, 1/2, 'bicubic');
    hr_rgb_4 = imresize(hr_rgb, 1/4, 'bicubic');
    
    count = count + 1;
%     LR_up_ycbcr(:,:,:,:,:,count) = lf_lr_up_ycbcr;
%     LR(:,:,:,:,count) = lf_lr_y;
%     HR(:,:,:,:,count) = hr_y;
%     HR_2(:,:,:,:,count) = hr_y_2;
%     HR_4(:,:,:,:,count) = hr_y_4;
    LR(:,:,:,:,:,count) = lf_lr_rgb;
    HR(:,:,:,:,:,count) = hr_rgb;
    HR_2(:,:,:,:,:,count) = hr_rgb_2;
    HR_4(:,:,:,:,:,count) = hr_rgb_4;
end

% LR_up_ycbcr = permute(LR_up_ycbcr,[2,1,3,5,4,6]);  %[h,w,3,ah,aw,N] -> [w,h,3,aw,ah,N]  
LR = permute(LR,[2,1,3,5,4,6]); 
HR = permute(HR,[2,1,3,5,4,6]);
HR_2 = permute(HR_2,[2,1,3,5,4,6]);
HR_4 = permute(HR_4,[2,1,3,5,4,6]);

%% writing to HDF5
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

% h5create(savepath,'/LR_up_ycbcr',size(LR_up_ycbcr),'Datatype','uint8');
h5create(savepath,'/LR',size(LR),'Datatype','uint8');
h5create(savepath,'/HR',size(HR),'Datatype','uint8');
h5create(savepath,'/HR_2',size(HR_2),'Datatype','uint8');
h5create(savepath,'/HR_4',size(HR_4),'Datatype','uint8');
% h5create(savepath,'/img_size',size(img_size),'Datatype','double');

% h5write(savepath, '/LR_up_ycbcr', LR_up_ycbcr);
h5write(savepath, '/LR', LR);
h5write(savepath, '/HR', HR);
h5write(savepath, '/HR_2', HR_2);
h5write(savepath, '/HR_4', HR_4);
% h5write(savepath, '/img_size', img_size);

h5disp(savepath);