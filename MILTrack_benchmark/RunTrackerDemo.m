%%
% Put 'img' folder and groundtruth_rect.txt into data and run the tracker
%%
clear all
close all
clc

video_path = './data/';
filename = strcat(video_path, 'groundtruth_rect', '.txt');
try
    ground_truth = load(filename);  
catch 
    fprintf('Cannot load groundtruth from %s', filename);
    return;
end


init_pos = ground_truth(1, 1:2);
target_sz = ground_truth(1, 3:4);

video_path = fullfile(video_path,'img');
img_files = dir(fullfile(video_path, '*.jpg'));
if isempty(img_files)
    img_files = dir(fullfile(video_path, '*.jpg'));
end
img_files = sort({img_files.name});
init_frame = 300; % Special case for David and three other videos, groundtruth starts from frame 300
img_files = img_files(init_frame:end);

visualize_track = true;
[positions, time] = mil_tracker(video_path, img_files, init_pos, target_sz, ...
    visualize_track);

precisions = precision_plot(positions, ground_truth, 'title', 1);
pn = load('position_neg.mat');
p = load('position_no_neg.mat');
precision_plot(pn.positions, ground_truth, 'title', 0);
precision_plot(p.positions, ground_truth, 'title', 0);
legend('original','ori_modified+neg_weight','ori_modified');

fprintf('Precision (20px):% 1.3f, FPS:% 4.2f\n',precisions(20), numel(img_files)/time)