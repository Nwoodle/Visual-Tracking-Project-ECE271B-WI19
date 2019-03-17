function selector = clfOriMilBoostUpdate(posx, negx, numSel)
    selector = zeros(numSel, 1);
    pos_smpl_ensemble_conf = zeros(1, size(posx.pospred, 2));
    neg_smpl_ensemble_conf = zeros(1, size(negx.negpred, 2));
    for i=1:numSel
        pos_smpl_ensemble_prob = mySigmoid(pos_smpl_ensemble_conf + posx.pospred, [1,0]);
        pos_bag_ensemble_prob = 1 - prod(1 - pos_smpl_ensemble_prob, 2);
        
        neg_smpl_ensemble_prob = mySigmoid(neg_smpl_ensemble_conf + negx.negpred, [1,0]);
        neg_bags_ensemble_prob = neg_smpl_ensemble_prob;
        
        likelihood = sum([log(pos_bag_ensemble_prob), log(1-neg_bags_ensemble_prob)], 2);
        %[~, selector(i)] = max(likelihood); 
        [~, classifier_inds] = sort(likelihood, 'descend');
        classifier_candidates = setdiff(classifier_inds, selector);
        selector(i) = classifier_candidates(1);
        pos_smpl_ensemble_conf = pos_smpl_ensemble_conf + posx.pospred(selector(i), :);
        neg_smpl_ensemble_conf = neg_smpl_ensemble_conf + negx.negpred(selector(i), :);
    end
end