function [alpha, sW, C, nlZ, dnlZ] = approxSO(hyper, covfunc, lik, x, y)

% Sparse Online approximation to the posterior following the lines of Csató
% and Opper.
% The function takes a specified covariance function (see covFunction.m) and
% likelihood function (see likelihoods.m), and is designed to be used with
% binaryGP.m. See also approximations.m. 
% No approximation to the evidence is provided and hence no gradient of the
% latter can be calculated. The approximation to the posterior comes in the
% form (alpha,C) - the natural parameters for the posterior process.
%
% mean = alpha'*k(X,z)            with k being the covariance function
% var  = k(z,z) + k(z,X)*C*k(X,z) 
%
% Parameters you might want to set:
% info = true  displays some information on which data point was
%              processed/added/updated
% gp.nmax      maximum nuber of basis vectors
% gp.etol      lower bound on the novelty a new basis vector should have
%
% Written by Hannes Nickisch 2007-03-29

info = false;  % print some information on the screen
gp.etol= .15;  % tolerance on novelty for a new point to add
gp.nmax=  50;  % maximum number of basis vectors

n = size(x,1); % number of data points to process
sW = []; nlZ=0; dnlZ=0; % dummy return values

gp.BV  =  [];  % set of basis vectors
gp.n   =   0;  % actual number of basis vectors
id     =  [];  % indices of basis vectors
gp.K   =  [];  % prior covariance  
gp.iK  =  [];  % inverse prior covariance

gp.al  =  [];  % coefficients alpha
gp.C   =  [];  % C = -inv(K+inv(W)) 

etol = gp.etol*feval(covfunc{:},hyper,0); % relative tolerance

if info, ct = 0; end

for t=1:n % go only once through the data set
    
% a new data point (x_t,y_t) arrives
    y_t = y(t); x_t = x(t,:);

% evaluate covariances, 1d marginal Gaussian: N(f_t|mu,s2)
    if numel(gp.BV)>0                     % there are already some basis vectors
        [k_tt, k_t] = feval(covfunc{:}, hyper, gp.BV, x_t);
        mu = k_t'*gp.al; s2 = k_tt + k_t'*gp.C*k_t;
    else                                        % this is the first basis vector
        k_tt = feval(covfunc{:}, hyper, x_t); k_t=[];
        mu = 0;          s2 = k_tt;
    end    
    if s2<1e-10, warning('approxOL: marg var < 0'), s2=1e-10; end
   
% compute online coefficients q and r
    [m0,m1,m2] = feval(lik,y_t,mu,s2);
    q   = (m1/m0-mu)/s2;
    r   = (m2 -2*mu*m1 +(mu^2-s2)*m0) / (m0*s2^2) - q^2;
   
% do projection on basis vectors to determine novelty
    e   = gp.iK*k_t;                                    % projection coordinates
    if numel(k_t)==0, gam=etol; else gam = k_tt-k_t'*e; end           % residual    

% geometrical test and parameter update    
    if gam<etol
        gp = upd(gp,k_t,q,r,e,gam);        % update (al,C) without including x_t
        if info, fprintf(' upd(%03d)',t), ct=ct+1; end
    else                          % add a new basis point x_t and enlarge (al,C)
        gp = add(gp,x_t,k_t,k_tt,q,r,e,gam); id=[id;t];
        if info, fprintf(' add(%03d)',t), ct=ct+1; end
    end
    if info, fprintf('[%02d]',gp.n), if ct>5, fprintf('\n'), ct=0; end, end

% removal of the basis vector with the smallest score i.e. do the KL-optimal
% projection; the covariance term is ignored (Csató PhD thesis p. 45)
    if gp.n>gp.nmax
        ep = gp.al.^2./(diag(gp.iK)+diag(gp.C));            % compute the scores
        [epr,tr]=min(ep);              % pick the vector with the smallest score
        gp = rem(gp, tr);                                    % remove the vector
        id = id( setdiff(1:end,tr) );                     % adjust the index set       
        if info, fprintf(' rem(%03d)[%02d]',tr,gp.n), ct=ct+1; end
    end
    if info, if ct>5, fprintf('\n'), ct=0; end, end

end

alpha = sparse(id,1,gp.al,n,1); C=gp.C;                      % return parameters



%% remove some basis vectors indexed by tr from the GP
function gp = rem(gp, tr)  % tr: remove, tk: keep
    tk = setdiff(1:gp.n,tr);                              % basis points to keep
    gp.BV = gp.BV(tk,:);    gp.n = gp.n-numel(tr);     % add to basis vector set
    gp.K  = gp.K(tk,tk);                               % update prior covariance
    
    Q     = gp.iK(tk,tr);      q = gp.iK(tr,tr);    qQ = q\Q';     % auxiliaries
    QC    = Q+gp.C( tk,tr);   qc = q+gp.C( tr,tr); QQq = Q*qQ;              
    
    gp.al = gp.al(tk) - QC*(qc\gp.al(tr));                 % update coefficients
    gp.C  = gp.C( tk,tk) + QQq - QC*(qc\QC');
    gp.iK = gp.iK(tk,tk) - QQq;                % update inverse prior covariance

    gp.iK = (gp.iK+gp.iK')/2;                          % for numerical stability
    gp.C  = (gp.C+gp.C')/2;   



%% update the GP with a single data point
function gp = upd(gp, k_t, q, r, e, gam)
    eta = 1/(1+gam*r);    

    s     = gp.C*k_t + e;                                   % auxiliary quantity    
    gp.al = gp.al + q*eta*s;                               % update coefficients
    gp.C  = gp.C  + s*r*eta*s';    

    gp.C  = (gp.C+gp.C')/2;                            % for numerical stability
    


%% add a single basis vector to the GP
function gp = add(gp, x_t, k_t, k_tt, q, r, e, gam)
    gp.BV = [gp.BV; x_t]; gp.n = gp.n+1;               % add to basis vector set
  
    gp.K  = [gp.K,k_t; k_t',k_tt];                     % update prior covariance
    
    if gp.n>1                                  % update inverse prior covariance
        gp.iK(end+1,end+1) = 0; 
        gp.iK = gp.iK +[e;-1]*(1/gam)*[e',-1];
    else
        gp.iK = 1/k_tt;
    end
     
    s     = [gp.C*k_t;1];                                   % auxiliary quantity    
    gp.al = [gp.al;0] + s*q;                               % update coefficients
    gp.C(end+1,end+1) = 0; gp.C = gp.C + s*diag(r)*s';    

    gp.iK = (gp.iK+gp.iK')/2;                          % for numerical stability
    gp.C  = (gp.C+gp.C')/2;


