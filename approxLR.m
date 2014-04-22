function [alpha, sW, L, nlZ, dnlZ] = approxLR(hyper, covfunc, lik, x, y)

% Label Regression
%    posterior does not depend on likelihood
%
% written by Hannes Nickisch, hn@tuebingen.mpg.de, 02/08
n     = size(x,1);
K     = feval(covfunc{:}, hyper, x);

% model selection over sn
sns   = linspace(0.5,2,6);     % noise standard deviations, y={+1,-1} thus sn<=1
ev = zeros(numel(sns),1);
for i=1:numel(sns)
    sn    = sns(i);
    alpha = (K+sn^2*eye(n))\y;
    sW    = sn*ones(n,1);
    ev(i) = -nlogZbound(hyper,covfunc,lik,x,y,alpha,sW.^2);
end
id = find(ev==max(ev));
sn = sns( id(1) );

% proceed with best value sn
alpha = (K+sn^2*eye(n))\y;
sW    = sn*ones(n,1);
L     = chol(eye(n)+sW*sW'.*K);

% stolen from GPR
R     = chol(K+sn^2*eye(n));
nlZ   = 0.5*y'*alpha + sum(log(diag(R))) + 0.5*n*log(2*pi);
dnlZ = zeros(size(hyper));               % set the size of the derivative vector
W    = R\(R'\eye(n))-alpha*alpha';
for i = 1:length(dnlZ)
    dnlZ(i) = sum(sum(W.*feval(covfunc{:}, hyper, x, i)))/2;
end


function nlogZ = nlogZbound(logtheta, covfunc, lik, x, y, alpha, W)

if any(isnan(full(alpha))), nlogZ=-Inf; return, end

n = length(y);
K = feval(covfunc{:}, logtheta, x); % evaluate the covariance matrix
m = K*alpha;

if numel(W)==length(W), W=diag(W); end

% original variables instead of mu and la
VinvK = inv(eye(n)+K*W); % A=V*inv(K)
V     = VinvK*K; V=(V+V')/2; % symmetry
v     = diag(V);

% calculate log(det( V*inv(K) ))
[L,U]=lu(VinvK); u=diag(U); logdet_VinvK=sum(log(abs(u))); 

% calculate alpha
[f,w]=gauher(20);
SumLog_lam = zeros(size(f)); % init accumulator
for i=1:n
    % squashing
    [dummy,log_lam] = feval( lik, y(i), sqrt(v(i))*f+m(i) );  
    % accumulation
    SumLog_lam = SumLog_lam+log_lam;
end
% Likelihood
nlogZ = -(w'*SumLog_lam) -logdet_VinvK/2 -n/2 +(alpha'*K*alpha)/2 +trace(VinvK)/2;