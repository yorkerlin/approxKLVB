function [alpha, sW, L, nlZ, dnlZ] = approxIVM(hyper, covfunc, lik, x, y)

% Informative Vector Machine: Lawrence, Seeger & Herbrich, 2003
%  - greedy forward selection
%  - 
%
% written by Hannes Nickisch 2007-04-10

n = size(x,1);

ivm.I    = false(n,1);  % active set indices
ivm.Id   = [];
ivm.J    = true( n,1);  % training set indices

ivm.n    =  0;   % size of active set
ivm.nmax = 25;   % desired sparsity, size of active set
ivm.etol = 1e-4; % threshold for numerical stability  

% representation
ivm.L    = zeros(0,0); % chol factor
ivm.M    = zeros(n,0); % stub
ivm.beta = zeros(0,1);

ivm.b = zeros(n,1); % active site
ivm.p = zeros(n,1); % active site 

ivm.a = zeros(n,1); % predictive marginal
for i=1:n, ivm.a(i) = feval(covfunc{:}, hyper, x(i,:)); end % diag(K)
ivm.h = zeros(n,1); % predictive marginal




for k=1:min(n,ivm.nmax)
    
    % EP step for the whole data set
    ac = ivm.a./(1-ivm.p.*ivm.a);          % cavity variance
    hc = ivm.h + ac.*(ivm.p.*ivm.h-ivm.b); % cavity mean
    
    % compute raw moments
    [m0, m1, m2] = feval(lik, y, hc, ac); 
    alpha = (m1./m0-hc)./ac;
    nu = -(m2 -2.*hc.*m1 +(hc.^2-ac).*m0) ./ (m0.*ac.^2) + alpha.^2;
    
    % compute new sites
    p_ = nu            ./(1-ac.*nu);
    b_ = (hc.*nu+alpha)./(1-ac.*nu);
    
    % compute new prediction
    h_ = hc + ac.*alpha;
%     a_ = (1-ac.*nu).*ac; % not needed
    
    % compute information gain score: depends on mean and variance change
    m  = 1 + ivm.a .* p_;
    D  = -( log(m) +1./m +(h_-ivm.h).^2 ./ ivm.a -1)/2; D = D(ivm.J);
    
    % compute entropy score: depends only on variance change
    D  = log( 1 - ivm.a.*nu )/2; D = D(ivm.J);

    % pick point to include
    i = 1:n; i = i(ivm.J); i = i( find(D==min(D),1,'first') );
    
    % update best site
    ivm.p(i) = p_(i);
    ivm.b(i) = b_(i);
    
    % update representation
    ivm.I(i) = true;
    ivm.Id   = [ivm.Id; i];
    ivm.J(i) = false;
    ivm.n = ivm.n+1;
    
    [Kii,Ki] = feval(covfunc{:}, hyper, x, x(i,:));
   
    Li  = sqrt(ivm.p(i))*ivm.M(i,:)';
    Lii = sqrt(1+ivm.p(i)*ivm.a(i));
    ivm.L = [ivm.L,Li; zeros(1,ivm.n-1),Lii];

    mu = (sqrt(ivm.p(i))*Ki -ivm.M*Li)/Lii;
    ivm.M = [ivm.M,mu];
    
    beta = alpha(i)*Lii/sqrt(ivm.p(i));
    ivm.beta = [ivm.beta; beta];
    ivm.h = ivm.h + beta*mu;
    ivm.a = ivm.a - mu.^2;
    
end

% % check representation
% K  = feval(covfunc{:}, hyper, x(ivm.I,:));
% sW = sqrt(ivm.p(ivm.I));
% norm( ivm.L - chol( eye(ivm.n)+sW*sW'.*K ) )
% norm( ivm.L'*ivm.L - (eye(ivm.n)+sW*sW'.*K) )
% norm( ivm.h - ivm.M*ivm.beta )
% [K,K]  = feval(covfunc{:}, hyper, x, x(ivm.I,:));
% norm( ivm.M - K*diag(sW)/ivm.L )
% K  = feval(covfunc{:}, hyper, x);
% norm( diag(K)-diag(ivm.M*ivm.M') - ivm.a )
% norm( ivm.beta - (diag(sW)*ivm.L')\ivm.b(ivm.I) )


% construct return parameters [alpha, sW, L, nlZ, dnlZ]

nlZ   = 0;
dnlZ  = 0;
L     = []; % set full
sW    = sqrt( ivm.p(ivm.Id) );
alpha = sparse( ivm.Id, 1, diag(sW)*inv(ivm.L)*ivm.beta, n, 1 );


