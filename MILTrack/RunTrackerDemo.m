video_path = 'E:\UCSD\Winter 2019\ECE271B\project\data\Benchmark\David\';
filename = strcat(video_path,'groundtruth_rect', '.txt');
try
    ground_truth = load(filename);  
catch 
    fprintf('Cannot load groundtruth from %s', filename);
    return;
end


init_pos = ground_truth(1, 1:2);
target_sz = ground_truth(1, 3:4);

video_path = fullfile(video_path,'img');
img_files = dir(fullfile(video_path, '*.png'));
if isempty(img_files)
    img_files = dir(fullfile(video_path, '*.jpg'));
end
img_files = sort({img_files.name});
init_frame = 300; % Special case for David and three other videos, groundtruth starts from frame 300
img_files = img_files(init_frame:end);

visualize_track = true;
[positions, time] = mil_tracker(video_path, img_files, init_pos, target_sz, ...
    visualize_track);