clear all, close all

load('cw1a.mat');
xs = linspace(-3, 3, 500)';
xtest = linspace(-3,3,10000)';

%returns a random scalar between (0,1), this will suffice for now
%expand to testing a range of values, say 10, across sufficient sample space

meanfunc = [];
covfunc = @covPeriodic;
likfunc = @likGauss;

hyp.mean = []; hyp.lik = 0; hyp.cov = [-0.5 1 0];
hyp2 = minimize(hyp, @gp,-500, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

xtestnext = xtest + exp(hyp2.cov(1,2));
hyp2.cov(1,2);

%add on the optimized period to xsingle, then fit model to both
 
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
%fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
%hold on; plot(xs, mu); 
%plot(x, y, '+')

%test strict periodicity by taking a random point at x, then at x+p
%repeat this a few times to find an average value for error/difference in
%mean and s2 (var)
%if strictly periodic, this should be zero or negligibles

[musingle s2single] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xtest);
[musinglenext s2singlenext] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xtestnext);
errors_mean = []; errors_s2 = []; errors = [];
for x = 1:length(xtest)
    error_mean = abs(musingle(x)-musinglenext(x))/(0.5*(musingle(x)+musinglenext(x)));
    error = abs(musingle(x)-musinglenext(x));
    error_s2 = abs(s2single(x)-s2singlenext(x))/s2single(x);
    %fix this appending business, something isn't working
    errors_mean=[errors_mean,error_mean]; errors_s2=[errors_s2,error_s2]; errors=[errors,error];
end

B = rmoutliers(errors_mean);
mean(errors_mean), mean(errors); mean(B),
hold on; plot(xtest, errors)

%suggests and average error of 60% from sampling 20 points. Some are only
%10% off, some are over 100% off?
%after sampling 1000 points and inceasing the upper limit on convergence,
%we come to a much better, and realistic, value of 7.8% error
%thus we can quantify 'how periodic' the function is.
