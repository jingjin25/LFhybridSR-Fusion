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

%% params
dataset = 'HCI_Inria_5x5';
savepath = sprintf('train_%s.h5',dataset);

H = 512;
W = 512;
ah = 9;
aw = 9;

%% initialization
data_label= zeros(H, W, 5, 5, 1, 'uint8');
data_LR_2 = zeros(H/2, W/2, 5, 5, 1, 'uint8');
data_LR_4 = zeros(H/4, W/4, 5, 5, 1, 'uint8');
data_LR_8 = zeros(H/8, W/8, 5, 5, 1, 'uint8');
data_LR_16 = zeros(H/16, W/16, 5, 5, 1, 'uint8');
% data_HR   = zeros(H, W, 1, 1, 1, 'uint8');
% data_HR_2   = zeros(H/2, W/2, 1, 1, 1, 'uint8');
% data_HR_4   = zeros(H/4, W/4, 1, 1, 1, 'uint8');

count = 0;
margain = 0;
%% HCI
folder = 'Dataset\LF_Dataset\Dataset_HCI';
list = dir(folder);
list = list(3:end);

test_listname = './HCI_test.txt';
f = fopen(test_listname);
if( f == -1 )
    error('%s does not exist!', test_listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
test_list = C{1};
fclose(f); 

for k = 1:size(list,1)
    lfname = list(k).name;
    ind = find(ismember(test_list,lfname));
    if ~numel(ind)
        
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
        
    HR = lf(:,:,1:2:9,1:2:9);
    
    
    LR_2 = imresize(HR, 1/2, 'bicubic');
    LR_4 = imresize(HR, 1/4, 'bicubic');
    LR_8 = imresize(HR, 1/8, 'bicubic');
    LR_16= imresize(HR,1/8, 'bicubic');
    

    % generate patches

    count = count+1;

    data_label(:, :, :, :, count) = HR;
    data_LR_2(:, :, :, :, count) = LR_2;
    data_LR_4(:, :, :, :, count) = LR_4;    
    data_LR_8(:, :, :, :, count) = LR_8;
    data_LR_16(:, :, :, :, count) = LR_16;

    end
          
end  

%% Inria
folder = 'Dataset\LF_Dataset\Dataset_Inria_synthetic\DLFD';
list = dir(folder);
list = list(3:end);

test_listname = './Inria_DLFD_test.txt';
f = fopen(test_listname);
if( f == -1 )
    error('%s does not exist!', test_listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
test_list = C{1};
fclose(f); 

for k = 1:size(list,1)
    lfname = list(k).name;
    ind = find(ismember(test_list,lfname));
    if ~numel(ind)
        
    lf_path = fullfile(folder,lfname);
    disp(lf_path);
    
    lf = zeros(H,W,ah,aw,'uint8');    

    for v = 1 : ah
        for u = 1 : aw
            imgname = sprintf('lf_%d_%d.png',v,u);
            sub_rgb = imread(fullfile(lf_path,imgname));
            sub_ycbcr = rgb2ycbcr(sub_rgb);
            lf(:,:,v,u) = sub_ycbcr(:,:,1);
        end
    end
        
    HR = lf(:,:,1:2:9,1:2:9);
    
    
    LR_2 = imresize(HR, 1/2, 'bicubic');
    LR_4 = imresize(HR, 1/4, 'bicubic');
    LR_8 = imresize(HR, 1/8, 'bicubic');
    LR_16 = imresize(HR, 1/16, 'bicubic');
    

    % generate patches

    count = count+1;

    data_label(:, :, :, :, count) = HR;
    data_LR_2(:, :, :, :, count) = LR_2;
    data_LR_4(:, :, :, :, count) = LR_4;    
    data_LR_8(:, :, :, :, count) = LR_8;
    data_LR_16 (:,:, :, :, count) = LR_16;

    end
          
end  
 

%% generate dat

order = randperm(count);
data_label= permute(data_label(:, :, :, :, order),[2,1,4,3,5]); %[h,w,ah,aw,N] -> [w,h,aw,ah,N]  
data_LR_2 = permute(data_LR_2(:, :, :, :, order),[2,1,4,3,5]);
data_LR_4 = permute(data_LR_4(:, :, :, :, order),[2,1,4,3,5]);
data_LR_8 = permute(data_LR_8(:, :, :, :, order),[2,1,4,3,5]);
data_LR_16 = permute(data_LR_16(:, :, :, :, order),[2,1,4,3,5]);
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
h5create(savepath, '/img_LR_16', size(data_LR_16), 'Datatype', 'uint8');
% h5create(savepath, '/img_HR', size(data_HR), 'Datatype', 'uint8');
% h5create(savepath, '/img_HR_2', size(data_HR_2), 'Datatype', 'uint8');
% h5create(savepath, '/img_HR_4', size(data_HR_4), 'Datatype', 'uint8');

h5write(savepath, '/img_label', data_label);
h5write(savepath, '/img_LR_2', data_LR_2);  
h5write(savepath, '/img_LR_4', data_LR_4);
h5write(savepath, '/img_LR_8', data_LR_8);
h5write(savepath, '/img_LR_16', data_LR_16);

% h5write(savepath, '/img_HR', data_HR);
% h5write(savepath, '/img_HR_2', data_HR_2);
% h5write(savepath, '/img_HR_4', data_HR_4);

h5disp(savepath);
