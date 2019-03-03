function [pos_samples, neg_samples] = sampleImgwVarConstraint(samples,pos_rad, smpl_ensemble_conf, target_ind, pos_max_num, neg_max_num, pix_var)
    delta_x_to_target = samples.sx - samples.sx(target_ind);
    delta_y_to_target = samples.sy - samples.sy(target_ind);
    dist_to_target = sqrt(delta_x_to_target.^2 + ...
    delta_y_to_target.^2);
    tri = delaunay(delta_x_to_target, delta_y_to_target);
    [dfx, dfy] = trigradient(tri, delta_x_to_target, delta_y_to_target, smpl_ensemble_conf);
    df = sqrt(dfx.^2 + dfy .^ 2);
    max_conf = smpl_ensemble_conf(target_ind);
    %feat_to_dist_factor = mean(sqrt(var(samples.feature, [], 2)))/sqrt(pix_var);
    dist_var = sqrt(abs(max_conf)) * (1 + mySigmoid(max_conf, [-1, 0]));
    fprintf('Confidence: %.2f, positive bag radius %.2f, scaled radius %.2f\n', max_conf, sqrt(abs(max_conf)), dist_var);
    pos_candidate_inds = find(df <= (df(target_ind)+0.1*(max(df)-df(target_ind))));
    neg_candidate_inds = find(df > df(target_ind));
    %     {
        figure(2);
        clf;
        hold on;

        trisurf(tri, delta_x_to_target, delta_y_to_target, smpl_ensemble_conf);
        axis vis3d;
        scatter3(0,0,smpl_ensemble_conf(target_ind), 'r', 'filled'); % Location of the target in current frame
        thetas = 0:0.01:2*pi;
        plot3(pos_rad*cos(thetas), pos_rad*sin(thetas), ...
        zeros(size(thetas)), 'LineWidth', 2);
        hold off;
        %-------------------
        figure(3);
        clf;
        hold on;

        trisurf(tri, delta_x_to_target, delta_y_to_target, df);
        axis vis3d;
        scatter3(0,0,250, 'r', 'filled'); % Location of the target in current frame
        thetas = 0:0.01:2*pi;
        plot3(pos_rad*cos(thetas), pos_rad*sin(thetas), ...
        zeros(size(thetas)), 'LineWidth', 2);
        hold off;
        %-------------------
%     }
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
