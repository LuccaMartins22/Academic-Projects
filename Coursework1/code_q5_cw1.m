%clear all, close all

load('cw1e.mat');
mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));
xs = -3 + (3+3)*rand(100,2);
xlin = transpose([linspace(-3,3,100);linspace(-3,3,100)]);
%xs is a random sample of 100 points
%xlin is an example dataset which cuts a diagonal line across the model

meanfunc = [];
covfunc1 = @covSEard;
likfunc = @likGauss;

hypf1.mean = []; hypf1.cov = [-1 -1 0]; hypf1.lik = 0;
hypf2.mean = []; hypf2.cov = 0.1*randn(5,1); hypf2.lik = 0;
hyp2f1 = minimize(hypf1, @gp,-500, @infGaussLik, meanfunc, covfunc1, likfunc, x, y);
%hyp2f2 = minimize(hypf2, @gp,-500, @infGaussLik, meanfunc, covfunc2, likfunc, x, y);

nlml = gp(hyp2f1, @infGaussLik, meanfunc, covfunc1, likfunc, x, y);
[mf1 s2f1] = gp(hyp2f1, @infGaussLik, meanfunc, covfunc1, likfunc, x, y, xs);
[mf1lin s2f1lin] = gp(hyp2f1, @infGaussLik, meanfunc, covfunc1, likfunc, x, y, xlin);
%[mf2 s2f2] = gp(hyp2f2, @infGaussLik, meanfunc, covfunc2, likfunc, x, y, xs);

f = [mf1+2*sqrt(s2f1); flipdim(mf1-2*sqrt(s2f1),1)]; 
flin = [mf1lin+2*sqrt(s2f1lin); flipdim(mf1lin-2*sqrt(s2f1lin),1)]; 
fill3([xlin(:,1); flipdim(xlin(:,1),1)], [xlin(:,2); flipdim(xlin(:,1),1)], flin, [7 7 7]/8)
hold on; 
plot3(xs(:,1), xs(:,2), mf1, 'x'); 
mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));

[y_test dev_test] = gp(hyp2f1, @infGaussLik, meanfunc, covfunc1, likfunc, x(1:100,:), y(1:100,:), x(101:121,:));
mse = mean((y_test - y(101:121,:)).^2);