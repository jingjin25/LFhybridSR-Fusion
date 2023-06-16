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
dataset = 'HCI_Inria_rgb_3x3';
savepath = sprintf('train_%s.h5',dataset);

H = 512;
W = 512;

ah = 9;
aw = 9;

%% initialization
data_label= zeros(H, W, 3, 3, 3, 1, 'uint8');
data_LR_2 = zeros(H/2, W/2, 3, 3, 3, 1, 'uint8');
data_LR_4 = zeros(H/4, W/4, 3, 3, 3, 1, 'uint8');
data_LR_8 = zeros(H/8, W/8, 3, 3, 3, 1, 'uint8');

count = 0;
margain = 0;

%% HCI
folder = 'Dataset\LF_Dataset\Dataset_HCI';
list = dir(folder);
list = list(3:end);

test_listname = './list/HCI_test.txt';
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
    
    lf = zeros(H,W,3,ah,aw,'uint8');    

    for v = 1 : ah
        for u = 1 : aw
            ind = (v-1)*9+(u-1);
            imgname = strcat('input_Cam',num2str(ind,'%03d'),'.png');
            sub_rgb = imread(fullfile(lf_path,imgname));
            lf(:,:,:,v,u) = sub_rgb;
        end
    end
        
    HR = lf(:,:,:,1:4:9,1:4:9);
    
    
    LR_2 = imresize(HR, 1/2, 'bicubic');
    LR_4 = imresize(HR, 1/4, 'bicubic');
    LR_8 = imresize(HR, 1/8, 'bicubic');
    

    % generate patches

    count = count+1;

    data_label(:, :, :, :, :, count) = HR;
    data_LR_2(:, :, :, :, :, count) = LR_2;
    data_LR_4(:, :, :, :, :, count) = LR_4;    
    data_LR_8(:, :, :, :, :, count) = LR_8;
    end
          
end  

%%
folder = 'Dataset\LF_Dataset\Dataset_Inria_synthetic\DLFD';
list = dir(folder);
list = list(3:end);

test_listname = './list/Inria_DLFD_test.txt';
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
    % read SAIs ==> lf[h,w,ah,aw]
    lf = zeros(H,W,3,ah,aw,'uint8');

    % HCI dataset
    for v = 1 : ah
        for u = 1 : aw
            imgname = sprintf('lf_%d_%d.png',v,u);
            sub = imread(fullfile(lf_path,imgname));
            lf(:,:,:,v,u) = sub;
        end
    end
    

    HR = lf(:, :, :, 1:4:9, 1:4:9);

    LR_2 = imresize(HR, 1/2, 'bicubic');
    LR_4 = imresize(HR, 1/4, 'bicubic');
    LR_8 = imresize(HR, 1/8, 'bicubic');


    count = count+1;
    data_label(:, :, :, :, :, count) = HR;
    data_LR_2(:, :, :, :, :, count) = LR_2;
    data_LR_4(:, :, :, :, :, count) = LR_4;    
    data_LR_8(:, :, :, :, :, count) = LR_8;
    end
end


order = randperm(count);
data_label= permute(data_label(:, :, :, :, :, order),[2,1,3,5,4,6]); %[h,w,3,ah,aw,N] -> [w,h,3,aw,ah,N]  
data_LR_2 = permute(data_LR_2(:, :, :, :, :, order),[2,1,3,5,4,6]);
data_LR_4 = permute(data_LR_4(:, :, :, :, :, order),[2,1,3,5,4,6]);
data_LR_8 = permute(data_LR_8(:, :, :, :, :, order),[2,1,3,5,4,6]);


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
