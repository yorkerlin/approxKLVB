function [alpha, sW, L, nlZ, dnlZ, sweep, err, mu, s2] = approxFV(hyper, covfunc, lik, x, y)

% Factorial Variational approximation to the posterior Gaussian Process as
% proposed by Csató, Fokoué, Opper & Schottky at NIPS 2000.
% The posterior is approximated by a distribution from the family of factorizing
% distributions. Marginal moments are computed by means of fixed point
% iteration.
% The function takes a specified covariance function (see 
% covFunction.m) and likelihood function (see likelihoods.m), and is designed to
% be used with binaryGP.m. See also approximations.m. The matrix L returned by
% the function is a full matrix and not a Cholesky factor.
%
% Written by Hannes Nickisch, 2007-05-11

eta0 = .5; tol = 5e-6; max_sweep = 500;         % learning rate, fault tolerance
n  = size(x,1); K = feval(covfunc{:}, hyper, x);           % evaluate covariance
R  = chol(K+1e-6*eye(n));
logdetK = 2*sum(log(diag(R)));                             % its log determinant
iK = inv(R); iK = iK*iK';                          % and its regularized inverse
m = zeros(n,1); err = Inf; sweep = 0; s2 = 1./diag(iK); eta = eta0;       % init

% hist_err = []; hist_eta = []; %%% history

while err>tol
   
    errold = err; 
    mu = m-s2.*(iK*m);
    [m0,m1] = feval(lik, y, mu, s2);
    dm = m1./m0 - m;
    err = max(dm.^2);

    m = m + eta*dm; % update mean parameters
 
    sweep = sweep+1;
    
%     hist_err = [ hist_err, err ]; hist_eta = [ hist_eta, eta ]; %%% history
    if sweep>max_sweep, break, end
    if errold>err, eta = min(eta*1.1,2*eta0); else eta = eta/2; end % adapt stepsize

end

% if sweep>0, semilogy(hist_err(2:end)), figure, semilogy(hist_eta), end %%% history

if sweep == max_sweep
    disp('Warning: maximum number of sweeps reached in function approxFV')
end

[m0, m1, m2] = feval(lik, y, mu, s2); m1 = m1./m0; m2 = m2./m0;  % final moments
alpha = iK*m1;                                     % construct return parameters
sW    = [];
L     = iK*diag(m2-m1.^2)*iK - iK;                        % return a full matrix

[l, loglik] = feval(lik, y, m1./sqrt(s2));
nlZ   = -sum(loglik) +(alpha'*(K-diag(s2))*alpha)/2 ...
        +logdetK/2 -sum(log(s2+1e-10))/2; 
    
dnlZ  = zeros(size(hyper));


