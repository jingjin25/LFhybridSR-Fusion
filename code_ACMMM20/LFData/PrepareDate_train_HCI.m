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
dataset = 'HCI';

folder = 'E:\LF_Data\Dataset_HCI\all';
savepath = sprintf('train_%s.h5',dataset);

listname = 'HCI_train.txt';

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
data_label= zeros(H, W, ah, aw, 1, 'uint8');
data_LR_2 = zeros(H/2, W/2, ah, aw, 1, 'uint8');
data_LR_4 = zeros(H/4, W/4, ah, aw, 1, 'uint8');
data_LR_8 = zeros(H/8, W/8, ah, aw, 1, 'uint8');

count = 0;
margain = 0;

%% generate data
for k = 1:size(list,1)
    lfname = list{k};
    lf_path = fullfile(folder,lfname);
    disp(lf_path);
    
    lf = zeros(H,W,ah,aw,'uint8');    

    for v = 1 : ah
        for u = 1 : aw
            ind = (v-1)*9+(u-1);
            imgname = strcat('input_Cam',num2str(ind,'%03d'),'.png');
            sub_rgb = imread(fullfile(lf_path,imgname));
            sub_ycbcr = rgb2ycbcr(sub_rgb);
            lf(:,:,v,u) = sub_ycbcr(:,:,1);
        end
    end
        
    HR = lf;
    
    LR_2 = imresize(HR, 1/2, 'bicubic');
    LR_4 = imresize(HR, 1/4, 'bicubic');
    LR_8 = imresize(HR, 1/8, 'bicubic');
   
    % generate patches

    count = count+1;

    data_label(:, :, :, :, count) = HR;
    data_LR_2(:, :, :, :, count) = LR_2;
    data_LR_4(:, :, :, :, count) = LR_4;    
    data_LR_8(:, :, :, :, count) = LR_8;
        
end  
 

%% generate dat

order = randperm(count);
data_label= permute(data_label(:, :, :, :, order),[2,1,4,3,5]); %[h,w,ah,aw,N] -> [w,h,aw,ah,N]  
data_LR_2 = permute(data_LR_2(:, :, :, :, order),[2,1,4,3,5]);
data_LR_4 = permute(data_LR_4(:, :, :, :, order),[2,1,4,3,5]);
data_LR_8 = permute(data_LR_8(:, :, :, :, order),[2,1,4,3,5]);

%% writing to HDF5
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath, '/img_label', size(data_label), 'Datatype', 'uint8'); % width, height, channels, number 
h5create(savepath, '/img_LR_2', size(data_LR_2), 'Datatype', 'uint8'); % width, height, channels, number 
h5create(savepath, '/img_LR_4', size(data_LR_4), 'Datatype', 'uint8');    
h5create(savepath, '/img_LR_8', size(data_LR_8), 'Datatype', 'uint8');   

h5write(savepath, '/img_label', data_label);
h5write(savepath, '/img_LR_2', data_LR_2);  
h5write(savepath, '/img_LR_4', data_LR_4);
h5write(savepath, '/img_LR_8', data_LR_8);

h5disp(savepath);
