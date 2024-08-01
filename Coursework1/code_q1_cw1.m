load('cw1a.mat');
xs = linspace(-3, 3, 500)';                  

meanfunc = [];                    
covfunc = @covSEiso;              
likfunc = @likGauss;

hyp = struct('mean', [], 'cov', [-1 0], 'lik', 0);
hyp2 = minimize(hyp, @gp,-300, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
 
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu); plot(x, y, '+')

hyp2;
min(s2)