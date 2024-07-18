clear all, close all

load('cw1a.mat');
xs = linspace(-5, 5, 200)';
z = linspace(-5, 5, 500)';

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
covfunc = {@covProd, {@covPeriodic, @covSEiso}};
likfunc = @likGauss;

hyp.mean = [0,0]; hyp.lik = 0; hyp.cov = [-0.5 0 0 2 0];
hyp2 = minimize(hyp, @gp,-500, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
 
K = feval(covfunc{:}, hyp2.cov, xs);
K = K + 1e-6*(eye(200));
mu = feval(meanfunc{:}, hyp2.mean, xs);
y = chol(K)'*0.5*gpml_randn(2, 200, 1) + mu + exp(hyp2.lik)*1*gpml_randn(2, 200, 1);
%changing the influence of the K matrix and the noise error

nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, xs, y);
[m s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, xs, y, z);

f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(xs, y, '+')
