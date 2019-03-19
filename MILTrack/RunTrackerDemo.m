clear all
close all
clc

% Get a list of all files and folders in this folder.
files = dir('./data/');
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
% Print folder names to command window.

f = fopen('results.txt', 'a');
fprintf(f, '/*************************************************\n');

for k = 3 : length(subFolders)

if convertCharsToStrings(subFolders(k).name) == 'Jogging'
    continue
end
    
fprintf('Sub folder #%d = %s\n', k - 2, subFolders(k).name);
video_path = sprintf('./data/%s', subFolders(k).name);

filename = fullfile(video_path,'groundtruth_rect.txt');
try
    ground_truth = load(filename);  
catch 
    fprintf('Cannot load groundtruth from %s', filename);
    return;
end

video_path = sprintf('./data/%s', subFolders(k).name);
video_path = fullfile(video_path, 'img');
init_pos = ground_truth(1, 1:2);
target_sz = ground_truth(1, 3:4);

img_files = dir(fullfile(video_path, '*.png'));
if isempty(img_files)
    img_files = dir(fullfile(video_path, '*.jpg'));
end
img_files = sort({img_files.name});

init_frame = 1; % Special case for David and three other videos, groundtruth starts from frame 300
if convertCharsToStrings(subFolders(k).name) == 'David'
    init_frame = 300;
end

img_files = img_files(init_frame:end);

visualize_track = true;
[positions, time] = mil_tracker(video_path, img_files, init_pos, target_sz, ...
    visualize_track);

%openfig(sprintf('./combined_fig/%s.fig', subFolders(k).name));
precisions = precision_plot(positions, ground_truth, 'title', 1);
%figure(4)
%title(sprintf('%s', subFolders(k).name))
fprintf(f, '%s-Precision (20px):% 1.3f, FPS:% 4.2f\n', subFolders(k).name, precisions(20), numel(img_files)/time);
saveas(figure(4), sprintf('./combined_fig/%s.fig', subFolders(k).name));
saveas(figure(4), sprintf('./combined_tif/%s.tif', subFolders(k).name));
close all

end
fprintf(f, '*************************************************/\n');
fclose(f);
