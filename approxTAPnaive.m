function [alpha, sW, L, nlZ, dnlZ] = approxTAPnaive(hyper, covfunc, lik, x, y)

% Variational approximation to the posterior Gaussian Process with predictive 
% uncertainty as proposed by Opper & Winter in NC 12, iss. 11 of 2000
%
% Naive version where lam = K(i,i)
%
% The function takes a specified covariance function (see 
% covFunction.m) and likelihood function (see likelihoods.m), and is designed to
% be used with binaryGP.m. See also approximations.m. The matrix L returned by
% the function is empty but can be calculated from sW and K.
%
% Written by Hannes Nickisch, 2007-05-11

eta = 5e-2; tol = 1e-5; max_sweep = 100;        % learning rate, fault tolerance
n = size(x,1); K = feval(covfunc{:}, hyper, x);            % evaluate covariance
alpha = zeros(n,1); lam = diag(K); dal = Inf; err = Inf; sweep = 0;       % init

while err>tol
   
    errold = err; err = max(dal.^2);
    m = K*alpha;
       
%     z = y.*(m-lam.*alpha)./sqrt(lam);    % cumGauss only
%     [dummy,n_p] = cumGauss([],z,'deriv');
%     dal = y.*n_p./sqrt(lam)-alpha;
    
    mu = m-lam.*alpha; s2 = lam-1;
    [m0,m1] = feval(lik, y, mu, s2);
    dal =(m1./m0-mu)./s2 - alpha;
    
    alpha = alpha + eta*dal; % update alpha parameters
    sweep = sweep+1;
    
    if sweep>max_sweep, break, end
    if errold>err, eta = eta*1.1; else eta = eta/2; end         % adapt stepsize

end

Lam = lam.*( 1./(m.*alpha) - 1 );
sW  = 1./(sqrt(abs(Lam))+eps);                     % return a reasonable sW only
nlZ=0; dnlZ = zeros(size(hyper)); L=[];                      % return parameters
