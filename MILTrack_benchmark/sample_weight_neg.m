function weights = sample_weight_neg(samples, target_loc, radius)
    % synthetic_ensemble_conf = -((samples.sx-target_loc(1)).^2+(samples.sy-target_loc(2)).^2)./4 + 5;
    
    synthetic_ensemble_conf = exp(-((samples.sx-target_loc(1)).^2+(samples.sy-target_loc(2)).^2 - (1.7*radius).^2));
    norm_synthetic_ensemble_conf = (synthetic_ensemble_conf - mean(synthetic_ensemble_conf))./std(synthetic_ensemble_conf);
    
    sigmoid_params = [1 0];
    
    weights = mySigmoid(norm_synthetic_ensemble_conf, sigmoid_params);
    weights = weights./sum(weights);
    
end