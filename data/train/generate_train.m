clear;close all;
%% settings
folder = './DIV2K/';
savepath = 'train.h5';
size_input = 63;
size_label = 63;
stride = 96;  % stride=56 in the paper
blur_sig = 0:0.5:5;
noi_sig = 0:5:50;
jpg_q = [100,80,60,50,40:-5:10];
level = [12, 17];
% moderate: [12, 17]
%     mild: [9, 11]
%   severe: [18, 20]

%% initialization
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
padding = abs(size_input - size_label)/2;
level_vec = noise_combination(10,level(1),level(2));
level_num = size(level_vec,1);
count = 0;
N = 1;

%% generate data
filepaths = dir(fullfile(folder,'*.png'));
    
for i = 1:length(filepaths)
    if mod(i,10)==0
        disp(i);
    end
    
    image = imread(fullfile(folder,filepaths(i).name));
    
    if size(image, 3) > 1
        image = im2double(image);
    else
        continue;
    end
    
    for k = 1:3  % 3 scales
        if k > 1
            image = imresize(image, 2/(k+1), 'bicubic');
        end
    
        for p = 1:N  % N kinds of distortions for each image  
            level_temp = level_vec(ceil(rand(1)*level_num), :); 
            im_label = image;
            [hei,wid,~] = size(im_label);

            %blur
            blur_idx = level_temp(1);
            blur_sigma = blur_sig(blur_idx) + rand(1) * (blur_sig(blur_idx+1) - blur_sig(blur_idx));
            kernel = fspecial('gaussian', 21, blur_sigma);
            im_input0 = imfilter(im_label, kernel, 'replicate');

            %noise
            noi_idx = level_temp(2);
            noi_sigma = noi_sig(noi_idx) + rand(1) * (noi_sig(noi_idx+1) - noi_sig(noi_idx));
            im_input1 = imnoise(im_input0, 'gaussian', 0, (noi_sigma/255)^2);

            %jpg
            jpg_idx = level_temp(3);
            quality = jpg_q(jpg_idx) + rand(1) * (jpg_q(jpg_idx+1) - jpg_q(jpg_idx));
            imwrite(im_input1, 'im_temp.jpg', 'jpg', 'Quality', quality); 
            im_input = im2double(imread('im_temp.jpg'));

            for x = 1 : stride : hei-size_input+1
                for y = 1 :stride : wid-size_input+1
                    subim_input = im_input(x : x+size_input-1, y : y+size_input-1, :);
                    subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1, :);
                    count=count+1;
                    data(:, :, :, count) = subim_input;
                    label(:, :, :, count) = subim_label;
                end
            end
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

% permutation
data = permute(data, [3, 1, 2, 4]);
label = permute(label, [3, 1, 2, 4]);

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
