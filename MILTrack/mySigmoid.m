function y = mySigmoid(x, params)
    y = 1./(1 + exp(-params(1).*(x-params(2))));
end