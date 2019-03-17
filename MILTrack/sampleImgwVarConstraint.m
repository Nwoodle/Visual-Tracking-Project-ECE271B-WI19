function [pos_samples, neg_samples] = sampleImgwVarConstraint(samples, smpl_ensemble_conf, target_ind, pos_max_num, neg_max_num, pix_var)
    delta_x_to_target = samples.sx - samples.sx(target_ind);
    delta_y_to_target = samples.sy - samples.sy(target_ind);
    dist_to_target = sqrt(delta_x_to_target.^2 + ...
        delta_y_to_target.^2);
    
    max_conf = smpl_ensemble_conf(target_ind);
    %feat_to_dist_factor = mean(sqrt(var(samples.feature, [], 2)))/sqrt(pix_var);
    dist_var = min(sqrt(abs(max_conf)), 10)/2; % * 2 * mySigmoid(max_conf, [-0.01, 0]);
    fprintf('Confidence: %.2f\n', max_conf);
    
    % Under construction; adaptive sampling to follow the ridges in the
    % confidence space
    conf_img_sz = [max(samples.sy)-min(samples.sy)+1, max(samples.sx)-min(samples.sx)+1];
    conf_img = zeros(conf_img_sz);
    samples_sx_shifted = samples.sx-min(samples.sx)+1;
    samples_sy_shifted = samples.sy-min(samples.sy)+1;
    smpl_inds = sub2ind(conf_img_sz, samples_sy_shifted, samples_sx_shifted);
    target_loc = [samples_sx_shifted(target_ind), samples_sy_shifted(target_ind)];
    conf_img(smpl_inds) = (smpl_ensemble_conf - min(smpl_ensemble_conf(:)))./abs(max(smpl_ensemble_conf(:)));
    conf_img = reshape(conf_img, conf_img_sz);
    
    hog_blc_sz = [2,2];
    hog_cell_side_lengths = 10:10:(min((min(conf_img_sz)/max(hog_blc_sz))-1, 30));
    hog_cell_sz_options = arrayfun(@(a) repmat(a, 1, 2), hog_cell_side_lengths, 'UniformOutput', false);
    hog_num_bins = 8;
    use_signed_hog = false;
    if ~use_signed_hog
        hog_angles = ((1:hog_num_bins)*pi./hog_num_bins)-pi/(2*hog_num_bins);
    else
        hog_angles = ((-pi:hog_num_bins)*pi./hog_num_bins)-pi/hog_num_bins;
    end
    hog_multiscale = zeros(hog_num_bins, length(hog_cell_sz_options));
    pt_vis_list = [];
    for i=1:length(hog_cell_sz_options)
        hog_cell_sz = hog_cell_sz_options{i};
        [hog_feats, valid_pts, pt_vis] = extractHOGFeatures(conf_img, target_loc, 'CellSize', hog_cell_sz, 'BlockSize', hog_blc_sz, 'NumBins', hog_num_bins, 'UseSignedOrientation', use_signed_hog);
        if valid_pts
            hog_multiscale(:, i) = mean(reshape(hog_feats, hog_num_bins, prod(hog_blc_sz)), 2);
            pt_vis_list = [pt_vis_list, pt_vis];
        else % If the current target is close to search window boundary, we need to conduct HoG centered at a different point
            alt_hog_loc = target_loc; 
            hog_region_ind = [];
            if alt_hog_loc(1) <= hog_cell_sz(1)
                alt_hog_loc(1) = hog_cell_sz(1) + 1;
                alt_hog_loc(2) = alt_hog_loc(2) + 1;
                hog_region_ind = [hog_region_ind, 2];
            elseif conf_img_sz(1) - alt_hog_loc(1) <= hog_cell_sz(1)
                alt_hog_loc(1) = conf_img_sz(1) - hog_cell_sz(1) - 1;
                alt_hog_loc(2) = alt_hog_loc(2) - 1;
                hog_region_ind = [hog_region_ind, 3];
            end
            if alt_hog_loc(2) <= hog_cell_sz(2)
                alt_hog_loc(2) = hog_cell_sz(2) + 1;
                alt_hog_loc(1) = alt_hog_loc(1) - 1;
                hog_region_ind = [hog_region_ind, 4];
            elseif conf_img_sz(2) - alt_hog_loc(2) <= hog_cell_sz(2)
                alt_hog_loc(2) = conf_img_sz(2) - hog_cell_sz(2) - 1;
                alt_hog_loc(1) = alt_hog_loc(1) + 1;
                hog_region_ind = [hog_region_ind, 1];
            end    
            
            [hog_feats, valid_pts, pt_vis] = extractHOGFeatures(conf_img, alt_hog_loc, 'CellSize', hog_cell_sz, 'BlockSize', hog_blc_sz, 'NumBins', hog_num_bins, 'UseSignedOrientation', use_signed_hog);
            if valid_pts
                hog_candidate_regions = reshape(hog_feats, hog_num_bins, prod(hog_blc_sz));
                if length(hog_region_ind) == 1
                    hog_multiscale(:, i) = hog_candidate_regions(:, hog_region_ind);
                else
                    hog_multiscale(:, i) = mean(hog_candidate_regions(:, hog_region_ind), 2);
                end
                pt_vis_list = [pt_vis_list, pt_vis];
            else
                hog_multiscale(:, i) = zeros(hog_num_bins, 1);
            end
        end
    end
    
    if isempty(hog_multiscale)
        hog_multiscale = ones(hog_num_bins, 2);
    end
    hog_selected = max(hog_multiscale, [], 2);
    [~, dom_dirct_inds] = sort(hog_selected, 'descend');
    pos_ellipse_ax_inds = [dom_dirct_inds(1), mod(dom_dirct_inds(1)+hog_num_bins/2, hog_num_bins)];
    pos_ellipse_ax_inds(pos_ellipse_ax_inds==0) = hog_num_bins;
    dom_dirct_ratio_normed = (hog_selected(pos_ellipse_ax_inds)+eps)./(sum(hog_selected(pos_ellipse_ax_inds))+eps);
    pos_ellipse_axes = ceil(dist_var)*(mySigmoid(dom_dirct_ratio_normed, [15, 0.5]));
    focal_dist = norm(pos_ellipse_axes(1) - pos_ellipse_axes(2));
    in_pos_ellipse = false(1, length(samples_sx_shifted));
    walk_along_ridge = true; 
    walk_perpen_ridge = false;
    if walk_along_ridge
        focal_loc_delta = focal_dist*[sin(hog_angles(pos_ellipse_ax_inds(1))), cos(hog_angles(pos_ellipse_ax_inds(1)))];
        focal_1 = target_loc + 2*focal_loc_delta;
        focal_2 = target_loc - 2*focal_loc_delta;
        if max_conf > 0
            in_pos_ellipse = prep_pos_bag(samples_sx_shifted, samples_sy_shifted, target_loc, ...
                focal_1, focal_2, pos_ellipse_axes(1), smpl_ensemble_conf, true, ...
                in_pos_ellipse);
        else
            in_pos_ellipse = prep_pos_bag(samples_sx_shifted, samples_sy_shifted, target_loc, ...
                focal_1, focal_2, pos_ellipse_axes(1), smpl_ensemble_conf, false, ...
                in_pos_ellipse);
        end
    end
    if walk_perpen_ridge 
        focal_loc_delta = focal_dist*[cos(hog_angles(pos_ellipse_ax_inds(1))), sin(hog_angles(pos_ellipse_ax_inds(1)))];
        focal_1 = target_loc + 2*focal_loc_delta;
        focal_2 = target_loc - 2*focal_loc_delta;
        in_pos_ellipse = prep_pos_bag(samples_sx_shifted, samples_sy_shifted, target_loc, ...
            focal_1, focal_2, pos_ellipse_axes(1), smpl_ensemble_conf, true, ...
            in_pos_ellipse);
    end
    
    figure(2);
    imshow(conf_img, []);
    hold on;
    plot(samples_sx_shifted(in_pos_ellipse), samples_sy_shifted(in_pos_ellipse), 'rx');
    plot(target_loc(1), target_loc(2), 'bx')
    plot(focal_1(1), focal_1(2), 'gx');
    plot(focal_2(1), focal_2(2), 'gx');
    for i=1:length(pt_vis_list)
        plot(pt_vis_list(i), 'Color', 'green');
    end
    hold off;
    
    figure(3);
    clf;
    view([45, 30]);
    hold on;
    tri = delaunay(delta_x_to_target, delta_y_to_target);
    trisurf(tri, delta_x_to_target, delta_y_to_target, smpl_ensemble_conf);
    axis vis3d;
    scatter3(0,0,smpl_ensemble_conf(target_ind), 'r', 'filled'); % Location of the target in current frame
    thetas = 0:0.01:2*pi;
    plot3(dist_var*cos(thetas), dist_var*sin(thetas), ...
    smpl_ensemble_conf(target_ind) + zeros(size(thetas)), 'LineWidth', 2);
    set(gca, 'YDir', 'reverse');
    hold off;
    
    pos_candidate_inds = find(in_pos_ellipse);
    neg_candidate_inds = find(~in_pos_ellipse);    
    %pos_candidate_inds = find(dist_to_target<=dist_var);
    %neg_candidate_inds = find(dist_to_target>dist_var);
    if isempty(pos_candidate_inds) 
        pause(0.1);
    end
    if length(pos_candidate_inds) > pos_max_num
        pos_candidate_inds = pos_candidate_inds(randi(length(pos_candidate_inds), pos_max_num, 1));
    end
    if length(neg_candidate_inds) > neg_max_num
        neg_candidate_inds = neg_candidate_inds(randi(length(neg_candidate_inds), neg_max_num, 1));
    end

    pos_samples.sx = samples.sx(pos_candidate_inds);
    pos_samples.sy = samples.sy(pos_candidate_inds);
    pos_samples.sw = samples.sw(pos_candidate_inds);
    pos_samples.sh = samples.sh(pos_candidate_inds);
      
    neg_samples.sx = samples.sx(neg_candidate_inds);
    neg_samples.sy = samples.sy(neg_candidate_inds);
    neg_samples.sw = samples.sw(neg_candidate_inds);
    neg_samples.sh = samples.sh(neg_candidate_inds);
    
    norm_smpl_ensemble_conf = (smpl_ensemble_conf - mean(smpl_ensemble_conf))./std(smpl_ensemble_conf);
    pos_samples.weight = mySigmoid(norm_smpl_ensemble_conf(pos_candidate_inds), [1, 0]);
    pos_samples.weight = pos_samples.weight./sum(pos_samples.weight);
    neg_samples.weight = mySigmoid(norm_smpl_ensemble_conf(neg_candidate_inds), [-1, 0]) + mySigmoid(norm_smpl_ensemble_conf(neg_candidate_inds), [1, 0]);
    neg_samples.weight = neg_samples.weight./sum(neg_samples.weight);
end
