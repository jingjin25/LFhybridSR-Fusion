%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate training data from HCI & Stanford dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===> train_DatasetName.h5 (in python)
% uint8 0-255
%  ['img_HR']   [N,ah,aw,h,w]
%  ['img_LR_2'] [N,ah,aw,h/2,w/2]
%  ['img_LR_4'] [N,ah,aw,h/4,w/4]
%  ['img_LR_8'] [N,ah,aw,h/8,w/8]

% ===> in matlab (inverse)
%  [w,h,3,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% path
dataset = 'Lytro';
 

folder = ['../../LFData/',dataset];
savepath = sprintf('train_%s.h5',dataset);

listname = 'Lytro_train.txt';
f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 

%% params
W = 536;
H = 368;

allah = 14;
allaw = 14;

ah = 8;
aw = 8;

%% initialization
data_label= zeros(H, W, ah, aw, 1, 'uint8');
data_LR_2 = zeros(H/2, W/2, ah, aw, 1, 'uint8');
data_LR_4 = zeros(H/4, W/4, ah, aw, 1, 'uint8');
data_LR_8 = zeros(H/8, W/8, ah, aw, 1, 'uint8');
% data_HR   = zeros(H, W, 1, 1, 1, 'uint8');
% data_HR_2   = zeros(H/2, W/2, 1, 1, 1, 'uint8');
% data_HR_4   = zeros(H/4, W/4, 1, 1, 1, 'uint8');

count = 0;
margain = 0;

%% generate data
for k = 1:size(list,1)
    lfname = list{k};
    lf_path = sprintf('%s/%s.png',folder,lfname);
    disp(lf_path);
    
    eslf = im2uint8(imread(lf_path));
    lf = zeros(H,W,allah,allaw,'uint8');    

    for v = 1 : allah
        for u = 1 : allah            
            sub = eslf(v:allah:end,u:allah:end,:);            
            sub = rgb2ycbcr(sub);           
            lf(:,:,v,u) = sub(1:H,1:W,1);           
        end
    end
        
    HR = lf(:,:,4:11,4:11);
    
    
    LR_2 = imresize(HR, 1/2, 'bicubic');
    LR_4 = imresize(HR, 1/4, 'bicubic');
    LR_8 = imresize(HR, 1/8, 'bicubic');
    

    % generate patches

    count = count+1;
    
    data_label(:, :, :, :, count) = HR;
    data_LR_2(:, :, :, :, count) = LR_2;
    data_LR_4(:, :, :, :, count) = LR_4;    
    data_LR_8(:, :, :, :, count) = LR_8;
%     data_HR(:,:,:,:,count) = HR(:,:,ceil(ah/2),ceil(aw/2));
%     data_HR_2(:,:,:,:,count) = LR_2(:,:,ceil(ah/2),ceil(aw/2));
%     data_HR_4(:,:,:,:,count) = LR_4(:,:,ceil(ah/2),ceil(aw/2));    
    
end  
 


%% generate dat

order = randperm(count);
data_label= permute(data_label(:, :, :, :, order),[2,1,4,3,5]); %[h,w,ah,aw,N] -> [w,h,aw,ah,N]  
data_LR_2 = permute(data_LR_2(:, :, :, :, order),[2,1,4,3,5]);
data_LR_4 = permute(data_LR_4(:, :, :, :, order),[2,1,4,3,5]);
data_LR_8 = permute(data_LR_8(:, :, :, :, order),[2,1,4,3,5]);
% data_HR   = permute(data_HR(:, :, :, :, order),[2,1,4,3,5]);
% data_HR_2   = permute(data_HR_2(:, :, :, :, order),[2,1,4,3,5]);
% data_HR_4   = permute(data_HR_4(:, :, :, :, order),[2,1,4,3,5]);


%% writing to HDF5
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath, '/img_label', size(data_label), 'Datatype', 'uint8'); % width, height, channels, number 
h5create(savepath, '/img_LR_2', size(data_LR_2), 'Datatype', 'uint8'); % width, height, channels, number 
h5create(savepath, '/img_LR_4', size(data_LR_4), 'Datatype', 'uint8');    
h5create(savepath, '/img_LR_8', size(data_LR_8), 'Datatype', 'uint8');   
% h5create(savepath, '/img_HR', size(data_HR), 'Datatype', 'uint8');
% h5create(savepath, '/img_HR_2', size(data_HR_2), 'Datatype', 'uint8');
% h5create(savepath, '/img_HR_4', size(data_HR_4), 'Datatype', 'uint8');

h5write(savepath, '/img_label', data_label);
h5write(savepath, '/img_LR_2', data_LR_2);  
h5write(savepath, '/img_LR_4', data_LR_4);
h5write(savepath, '/img_LR_8', data_LR_8);
% h5write(savepath, '/img_HR', data_HR);
% h5write(savepath, '/img_HR_2', data_HR_2);
% h5write(savepath, '/img_HR_4', data_HR_4);


h5disp(savepath);
