function [alpha, sW, L, nlZ, dnlZ] = approxEC(hyper, covfunc, lik, x, y)

% Expectation Consistent approximation to the posterior process. Single loop
% algorithm from
%    Opper&Winther: Expectation Consistent Approximate Inference, JMLR, 2005
%
% The function takes a specified covariance function (see covFunction.m) and
% likelihood function (see likelihoods.m), and is designed to be used with
% binaryGP.m. See also approximations.m. In the EC algorithm, all sites are 
% updated at once.
%
% Copyright (c) 2009 Hannes Nickisch 2009-02-02

tol = 1e-4; max_sweep = 30;           % tolerance for when to stop EP iterations

n = size(x,1);
K = feval(covfunc{:}, hyper, x);                % evaluate the covariance matrix

% init
lamr = zeros(2*n,1);

sweep = 0; 
delta = Inf;
while sweep<max_sweep && delta>tol

    % 1) send a message from r to q
    % cubic operations
    % get Gaussian factorial separator s(x), lams
    % get out params
    gamr = lamr(1:n);   Lamr = lamr(n+1:2*n);
    % compute marginal expectations
    % Xir  = inv(diag(Lamr)-J); % numerically unsafe
    [Xir,mr] = epComputeParams(K,y,Lamr,gamr,lik);
    mur  = [mr; -(diag(Xir)+mr.^2)/2];  % <g(x)>_r
    % compute lams
    mean = mur(1:n); var = -2*mur(n+1:2*n)-mean.^2;
    lams = [mean./var; 1./var]; % lamq + lamr
    % update
    lamq = lams-lamr;

    % 2) send a message from q to r
    % only scalar operations => exact
    % get Gaussian factorial separator s(x), lams
    % get out params
    gamq = lamq(1:n);   Lamq = lamq(n+1:2*n);
    % compute marginal expectations
    [m0, m1, m2] = feval(lik, y, gamq./Lamq, 1./Lamq);
    muq  = [m1./m0; -m2./m0/2];  % <g(x)>_q
    % compute lams
    mean = muq(1:n); var = -2*muq(n+1:2*n)-mean.^2;
    lams = [mean./var; 1./var]; % lamq + lamr
    % update
    lamr = lams-lamq;
    
    % bookkeeping
    delta = norm(mur-muq); sweep = sweep+1;
end
% the rest is like in EP
ttau = lamr(n+1:2*n); tnu = lamr(1:n);

if sweep == max_sweep
    disp('Warning: maximum number of sweeps reached in function approxEC')
end

[Sigma, mu, nlZ, L] = epComputeParams(K,y,ttau,tnu,lik);
sW = sqrt(ttau);                  % compute output arguments, L and nlZ are done
alpha = tnu-sW.*solve_chol(L,sW.*(K*tnu));

if nargout > 4                                         % do we want derivatives?
  dnlZ = zeros(size(hyper));                    % allocate space for derivatives
  F = alpha*alpha'-repmat(sW,1,n).*solve_chol(L,diag(sW)); 
  for j=1:length(hyper)
    dK = feval(covfunc{:}, hyper, x, j);
    dnlZ(j) = -sum(sum(F.*dK))/2;
  end
end

% function to compute the parameters of the Gaussian approximation, Sigma and
% mu, and the negative log marginal likelihood, nlZ, from the current site 
% parameters, ttau and tnu. Also returns L (useful for predictions).
function [Sigma, mu, nlZ, L] = epComputeParams(K, y, ttau, tnu, lik)

n = length(y);                                        % number of training cases
ssi = sqrt(ttau);                                         % compute Sigma and mu
L = chol(eye(n)+ssi*ssi'.*K);                            % L'*L=B=eye(n)+sW*K*sW
V = L'\(repmat(ssi,1,n).*K);
Sigma = K - V'*V;
mu = Sigma*tnu;

tau_n = 1./diag(Sigma)-ttau;               % compute the log marginal likelihood
nu_n  = mu./diag(Sigma)-tnu;                      % vectors of cavity parameters
nlZ   = sum(log(diag(L))) - sum(log(feval(lik, y, nu_n./tau_n, 1./tau_n)))   ...
       -tnu'*Sigma*tnu/2 - nu_n'*((ttau./tau_n.*nu_n-2*tnu)./(ttau+tau_n))/2 ...
       +sum(tnu.^2./(tau_n+ttau))/2-sum(log(1+ttau./tau_n))/2;
