clear all, close all

load('cw1e.mat');

x1s = linspace(-10,10,101)';
x2s = linspace(-10,10,101)';
[X1S, X2S] = meshgrid(x1s,x2s);
MU = zeros(size(X2S));
S2 = zeros(size(X2S));

meanfunc = [];
covfunc1 = @covSEard;
covfunc2 = {@covProd, {@covSEard, @covSEard}};
likfunc = @likGauss;

hypf1.mean = []; hypf1.cov = [-1 -1 0]; hypf1.lik = 0;
hypf2.mean = []; hypf2.cov = 0.1*randn(6,1); hypf2.lik = 0;
hyp2f1 = minimize(hypf1, @gp,-1000, @infGaussLik, meanfunc, covfunc1, likfunc, x, y);
hyp2f2 = minimize(hypf2, @gp,-1000, @infGaussLik, meanfunc, covfunc2, likfunc, x, y);

for i = 1:length(x1s)
    for j = 1:length(x2s)
        xs = [x1s(i) x2s(j)];
        [MU(j,i), S2(j,i)]=gp(hyp2f1, @infGaussLik, meanfunc, covfunc1, likfunc, x, y, xs);
    end
end

mesh(X1S, X2S, MU); hold on; 
mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));
scatter3(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));
