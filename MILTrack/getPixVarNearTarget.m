function pix_var = getPixVarNearTarget(img, target_loc, bbox_w, bbox_h)
    scale_fact = 0.1;
    patch_near_target = img(target_loc(2)-ceil(bbox_h*scale_fact):target_loc(2)+ceil(bbox_h*scale_fact), ...
    target_loc(1)-ceil(bbox_w*scale_fact):target_loc(1)+ceil(bbox_w*scale_fact));
    pix_var = var(patch_near_target(:));
end