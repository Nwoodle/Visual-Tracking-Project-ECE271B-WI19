function [positions, time] = mil_tracker(video_path, img_files, init_pos, target_sz, ...
    visualize_track)
%%
% MODIFIED from:
% Demo for paper--Kaihua Zhang, Huihui song, 'Real-time Visual Tracking via
% Online Weighted Multiple Instance Learning', Pattern Recongtion.
% Author: Kaihua Zhang, Dept. of Computing, HK PolyU.
% Email: zhkhua@gmail.com
% Date: 8/8/1011
%%
% Parameters:
% video_path: Path to the images of the video
% img_files: a cell array of image file names
% init_pos: initial position of the cornor of the bounding box closet to
% the origin, in the format [col, row], i.e. [x, y]
% target_sz: size of the initial bounding box in the format [width, height]
% visualize: whether to show tracking visually
% Outputs:
% positions: location of the center of the target in the format [col, row]
% time: total time of tracking a video, w/o the time of loading images
%%
rng(0);
addpath(video_path);
%----------------------------------
initstate = [init_pos, target_sz];%initial tracker
%The object position in the first frame
% x = initstate(1);% the column where the target is x axis at the Top left corner
% y = initstate(2);% thw row where the target is y axis at the Top left corner
% w = initstate(3);% width of the rectangle
% h = initstate(4);% height of the rectangle
%----------------------------------
num_of_frames = length(img_files);
positions = zeros(num_of_frames, 2);
time = 0;
positions(1, :) = init_pos + floor(target_sz/2); 
%% Parameter Settings
trparams.init_negnumtrain = 50;%number of trained negative samples
trparams.init_postrainrad = 4.0;%radical scope of positive samples; boy 8
trparams.initstate = initstate;% object position [x y width height]
trparams.srchwinsz = max(initstate(3), initstate(4))/2;% MOODIFIED: size of search window; ori. 25, boy 35; changed to be 1/ of bounding box's max side length
%-------------------------
% classifier parameters
clfparams.width = trparams.initstate(3);
clfparams.height= trparams.initstate(4);
%-------------------------
% feature parameters:number of rectangle
ftrparams.minNumRect = 2;
ftrparams.maxNumRect = 4;
%-------------------------
lRate = 0.85;% learning rate parameter ; 0.7 for biker1
%-------------------------
M = 250;% number of all weak classifiers in feature pool
numSel = 50; % number of selected weak classifier 
%-------------------------Initialize the feature mean and variance
posx.mu = zeros(M,1);% mean of positive features
negx.mu = zeros(M,1);
posx.sig= ones(M,1);% variance of positive features
negx.sig= ones(M,1);
%-------------------------
%compute feature template
[ftr.px,ftr.py,ftr.pw,ftr.ph,ftr.pwt] = HaarFtr(clfparams,ftrparams,M);
%% initilize the first frame
%---------------------------
img = imread(img_files{1});
img = double(img(:,:,1));
[rowz,colz] = size(img);
%---------------------------
%compute sample templates
posx.sampleImage = sampleImg(img,initstate,trparams.init_postrainrad,0,100000);
negx.sampleImage = sampleImg(img,initstate,trparams.srchwinsz,1.5*trparams.init_postrainrad,500);
%--------extract haar features
iH = integral(img);%Compute integral image
selector = 1:M;% select all weak classifier in pool
posx.feature = getFtrVal(iH,posx.sampleImage,ftr,selector);
negx.feature = getFtrVal(iH,negx.sampleImage,ftr,selector);
%--------Update the weak classifiers
posx.sampleImage.weight = gen_sample_weight(posx.sampleImage, initstate(1:2), true);
negx.sampleImage.weight = gen_sample_weight(negx.sampleImage, initstate(1:2), false);
[posx.mu,posx.sig,negx.mu,negx.sig] = weakClfUpdate(posx,negx,posx.mu,posx.sig,negx.mu,negx.sig,lRate);% update distribution parameters
posx.pospred = weakClassifier(posx,negx,posx,selector);% Confidence values for positive samples
negx.negpred = weakClassifier(posx,negx,negx,selector);% ... for negative samples
%-----------------------------------Feature selection
selector = clfOriMilBoostUpdate(posx,negx,numSel);
%--------------------------------------------------------
%% Start tracking
for i = 2:num_of_frames
    fprintf('Frame %d ', i);
    img1 = imread(img_files{i});
    
    tic; % The total running time of the algo does not include image read time
    img = double(img1(:,:,1));% Only utilize one channel of image
    %if length(size(img))>2
    %    img = rgb2gray(img);
    %end
    detectx.sampleImage = sampleImg(img,initstate,trparams.srchwinsz,0,100000);
    iH = integral(img);%Compute integral image
    detectx.feature = getFtrVal(iH,detectx.sampleImage,ftr,selector);
    %----------------------------------
    r = weakClassifier(posx,negx,detectx,selector);% compute the confidence for all samples
    ensemble_confs = sum(r);% linearly combine the confidences from weak classifiers to get ensemble confidences
    %-------------------------------------
    [c,index] = max(ensemble_confs);
    %-------------------------------------
    x = detectx.sampleImage.sx(index);
    y = detectx.sampleImage.sy(index);
    w = detectx.sampleImage.sw(index);
    h = detectx.sampleImage.sh(index);
    initstate = [x y w h];
    %-----------------------------------------Show the tracking result
    if visualize_track
        figure(1);
        imshow(uint8(img1));
        rectangle('Position',initstate,'LineWidth',4,'EdgeColor','r');
        text(5, 18, strcat('#',num2str(i)), 'Color','y', 'FontWeight','bold', 'FontSize',20);
        set(gca,'position',[0 0 1 1]); 
        pause(0.00001); 
    end
    %------------------------------------------Sampling test
    %pix_var = getPixVarNearTarget(img, [x,y], w, h);
    pix_var = 0;
    [posx.sampleImage, negx.sampleImage] = sampleImgwVarConstraint(detectx.sampleImage, ensemble_confs, index, 60, 60, pix_var);
    %--------------------------------------------------Update all the features in pool
    selector = 1:M;
    posx.feature = getFtrVal(iH,posx.sampleImage,ftr,selector);
    negx.feature = getFtrVal(iH,negx.sampleImage,ftr,selector);
    %--------------------------------------------------
    [posx.mu,posx.sig,negx.mu,negx.sig] = weakClfUpdate(posx,negx,posx.mu,posx.sig,negx.mu,negx.sig,lRate);% update distribution parameters
    posx.pospred = weakClassifier(posx,negx,posx,selector);
    negx.negpred = weakClassifier(posx,negx,negx,selector);
    %--------------------------------------------------
    selector = clfOriMilBoostUpdate(posx,negx,numSel);% select the most discriminative weak classifiers 
    
    % Store target center position and accumulate tracking time
    positions(i, :) = [x, y] + floor([w, h]/2);
    time = time + toc();
end
%%