function y = cal_mean(x, w)
    Img = padarray(x, w/2 - 1, 'symmetric', 'pre');
    Img = padarray(Img, w/2, 'symmetric', 'post');
    Img = Img';
    Img = padarray(Img, w/2 - 1, 'symmetric', 'pre');
    Img = padarray(Img, w/2, 'symmetric', 'post'); 
    Img = Img';
    kernel = ones(w)/w^2; 
    y = conv2(Img, kernel, 'valid'); 
end

