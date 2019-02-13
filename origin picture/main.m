for i = 1:50
    origin_img = imread(['0001 (', num2str(i,'%d'), ').jpg']);
    origin_img = im2double(origin_img);
    w = 4;  %% change w to change the size of kernel

    %% If it's a gray image, process as follows.
    if (length(size(origin_img)) == 2)
        output = cal_mean(origin_img, 100); 
        silent_feature = abs(origin_img - output);
        silent_feature = mat2gray(silent_feature);
        filename = ['E:\研究生学习\ECE 271B\project\silent feature\w = 4\', num2str(i,'%d'),'.png' ];
        imwrite(silent_feature, filename, 'png');
    end

    %% If it's a color image, process as follows.
    if (length(size(origin_img)) == 3)
        type = makecform('srgb2lab');
        Img = applycform(origin_img, type);
        L = Img(:, :, 1);
        a = Img(:, :, 2);
        b = Img(:, :, 3);
        output_L = cal_mean(L, w);
        output_a = cal_mean(a, w);
        output_b = cal_mean(b, w);    
        silent_feature = sqrt((L - output_L).^2 + (a - output_a).^2 + (b - output_b).^2);
        silent_feature = mat2gray(silent_feature);
        filename = ['E:\研究生学习\ECE 271B\project\silent feature\w = 4\', num2str(i,'%d'),'.png' ];
        imwrite(silent_feature, filename, 'png');
    end
end

