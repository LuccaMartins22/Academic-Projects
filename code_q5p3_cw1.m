clear all, close all

load('cw1e.mat');
mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));
xs = -3 + (3+3)*rand(1000,2);
xlin = transpose([linspace(-3,3,1000);linspace(-3,3,1000)]);

meanfunc = [];
covfunc2 = {@covProd, {@covSEard, @covSEard}};
likfunc = @likGauss;

hypf2.mean = []; hypf2.cov = 0.1*randn(6,1); hypf2.lik = 0;
hyp2f2 = minimize(hypf2, @gp,-500, @infGaussLik, meanfunc, covfunc2, likfunc, x, y);

[mf2 s2f2] = gp(hyp2f2, @infGaussLik, meanfunc, covfunc2, likfunc, x, y, xs);
[mf2lin s2f2lin] = gp(hyp2f2, @infGaussLik, meanfunc, covfunc2, likfunc, x, y, xlin);

f = [mf2+2*sqrt(s2f2); flipdim(mf2-2*sqrt(s2f2),1)]; 
flin = [mf2lin+2*sqrt(s2f2lin); flipdim(mf2lin-2*sqrt(s2f2lin),1)]; 
%fill3([xlin(:,1); flipdim(xlin(:,1),1)], [xlin(:,2); flipdim(xlin(:,1),1)], flin, [7 7 7]/8)
hold on; 
plot3(xs(:,1), xs(:,2), mf2, 'x'); 

[y_test dev_test] = gp(hyp2f2, @infGaussLik, meanfunc, covfunc2, likfunc, x(1:100,:), y(1:100,:), x(101:121,:));
mse = mean((y_test - y(101:121,:)).^2);

%now let's consider the log marginal likelihood

nlml = gp(hyp2f2, @infGaussLik, meanfunc, covfunc2, likfunc, x, y);
z = [linspace(-2.9, 2.9, 101);linspace(-2.9, 2.9, 101)]';
[m_nlml s2_nlml] = gp(hyp2f2, @infGaussLik, meanfunc, covfunc2, likfunc, x, y, z);
f_nlml = [m_nlml+2*sqrt(s2_nlml); flipdim(m_nlml-2*sqrt(s2_nlml),1)]; 
%fill3([z(:,1); flipdim(z(:,1),1)],[z(:,2); flipdim(z(:,2),1)], f_nlml, [7 7 7]/8)
%hold on; plot3(z(:,1), z(:,2), m_nlml);

%what about marginal likelihood?

%nlml(fn1) = -19.2187
%nlml(fn2) = -19.2187

%mse(fn1) = 0.0551
%mse(fn2) = 0.0551

%mean(s2f1) = 0.0136
%mean(s2f2) = 0.0136

%so what about model complexity? Clearly models achieve identical results,
%but fn1 does so with lower complexity (less hyperparams, less complex
%covfun etc)

