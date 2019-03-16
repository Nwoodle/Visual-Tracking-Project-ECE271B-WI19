function in_pos_ellipse = prep_pos_bag(samples_sx, samples_sy, target_loc, ...
    focal_1, focal_2, half_maj_ax, smpl_ensemble_conf, bag_direction_positive, ...
    in_pos_ellipse)
    if isempty(in_pos_ellipse)
        in_pos_ellipse = false(1, length(samples_sx));
    end
    in_pos_ellipse_1 = (vecnorm([samples_sx; samples_sy]-focal_1') ...
        + vecnorm([samples_sx;samples_sy]-target_loc')) < 2*half_maj_ax(1);
    in_pos_ellipse_2 = (vecnorm([samples_sx; samples_sy]-focal_2') ...
        + vecnorm([samples_sx;samples_sy]-target_loc')) < 2*half_maj_ax(1);
    
    if length(find(in_pos_ellipse_1)) < 4 % If the ellipse is too small, use rectangle instead
        in_pos_ellipse_1 = bitand((samples_sx-focal_1(1)).*(samples_sx-target_loc(1)) <=0, ...
            (samples_sy-focal_1(2)).*(samples_sy-target_loc(2)) <= 0);
    end
    if length(find(in_pos_ellipse_2)) < 4
        in_pos_ellipse_2 = bitand((samples_sx-focal_2(1)).*(samples_sx-target_loc(1)) <=0, ...
            (samples_sy-focal_2(2)).*(samples_sy-target_loc(2)) <= 0);
    end
    
    if bag_direction_positive
        if mean(smpl_ensemble_conf(in_pos_ellipse_1))>mean(smpl_ensemble_conf(in_pos_ellipse_2))
            in_pos_ellipse = bitor(in_pos_ellipse, in_pos_ellipse_1);
        else
            in_pos_ellipse = bitor(in_pos_ellipse, in_pos_ellipse_2);
        end
    else
        if mean(smpl_ensemble_conf(in_pos_ellipse_1))<mean(smpl_ensemble_conf(in_pos_ellipse_2))
            in_pos_ellipse = bitor(in_pos_ellipse, in_pos_ellipse_1);
        else
            in_pos_ellipse = bitor(in_pos_ellipse, in_pos_ellipse_2);
        end
    end
end