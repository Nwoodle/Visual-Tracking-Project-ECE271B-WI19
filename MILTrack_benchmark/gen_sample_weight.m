function weights = gen_sample_weight(samples, target_loc, is_positive_sample)
    synthetic_ensemble_conf = -((samples.sx-target_loc(1)).^2+(samples.sy-target_loc(2)).^2)./4 + 5;
    norm_synthetic_ensemble_conf = (synthetic_ensemble_conf - mean(synthetic_ensemble_conf))./std(synthetic_ensemble_conf);
    if is_positive_sample
        sigmoid_params = [1, 0];
    else
        sigmoid_params = [-1, 0];
    end
    
    weights = mySigmoid(norm_synthetic_ensemble_conf, sigmoid_params);
    weights = weights./sum(weights);
end