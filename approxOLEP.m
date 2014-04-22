function [alpha, sW, C, nlZ, dnlZ] = approxOLEP(hyper, covfunc, lik, x, y)
% approximation to the posterior using Lehel Csato's Sparse Online GPs
% written by Hannes Nickisch, 2007/04/04


% open questions: info, error do not depend on order of presentation
% wheras evidence strongly does

sW = []; nlZ=0; dnlZ=0; % dummy return values

info = false;  % print some information on the screen
n = size(x,1); % number of data points to process

gp.BV  =  [];  % set of basis vectors
gp.K   =  [];  % prior covariance  
gp.iK  =  [];  % inverse prior covariance

gp.al  =  [];  % coefficients alpha
gp.C   =  [];  % C = -inv(K+inv(W)) 

gp.etol= .05;  % tolerance on novelty for a new point to add
gp.n   =   0;  % actual number of basis vectors
gp.nmax=  50;  % maximum number of basis vectors

gp.ep  = true; % apply EP
if gp.ep, gp.P  = zeros(n,0); end % projection matrix
if gp.ep, gp.Z  = zeros(n,1); end % vector nlZ for EP
if gp.ep, gp.a  = zeros(n,1); end % vector a for EP
if gp.ep, gp.la = zeros(n,1); end % vector lambda for EP

gp.id = []; % indices of basis vectors in the initial training set
etol = gp.etol*feval(covfunc{:},hyper,0); % relative tolerance

if info, ct = 0; end

Nsw = 6; % number of sweeps through the data set

% put things in random order
order = randperm(n);

for sweep=1:Nsw % do several sweeps
    
    if info, fprintf('sweep[%02d/%02d]\n',sweep,Nsw), end
    
    for t=order

    % a new data point (x_t,y_t) arrives
        y_t = y(t); x_t = x(t,:);

    % EP adjustment: remove effect of actual data point
        if gp.ep && gp.la(t)>1e-10
            p  = gp.P(t,:)'; Kp = gp.K*p; CKp = gp.C*Kp;
            h  = CKp + p; 
            nu = 1/( 1/gp.la(t) - p'*Kp - Kp'*CKp );

            gp.al = gp.al + h*nu*(gp.al'*Kp-gp.a(t));
            gp.C  = gp.C  + nu*h*h';
        end  
        
    % evaluate covariances, 1d marginal Gaussian: N(f_t|mu,s2) (cavity)
        if numel(gp.BV)>0                 % there are already some basis vectors
            [k_tt, k_t] = feval(covfunc{:}, hyper, gp.BV, x_t);
            mu = k_t'*gp.al; s2 = k_tt + k_t'*gp.C*k_t;
        else                                    % this is the first basis vector
            k_tt = feval(covfunc{:}, hyper, x_t); k_t=[];
            mu = 0;          s2 = k_tt;
        end    
        if s2<1e-12, warning('approxOLEP: marg var < 0'), s2=1e-12; end  

    % compute online coefficients q and r
        [m0,m1,m2] = feval(lik,y_t,mu,s2);
        q   = (m1/m0-mu)/s2;
        r   = (m2 -2*mu*m1 +(mu^2-s2)*m0) / (m0*s2^2) - q^2;

    % EP update
        if gp.ep
            if numel(gp.BV)>0, m=gp.al'*k_t; else m=0; end % new mean        
            gp.a(t)  = m - q/r; 
            gp.la(t) = -r/(1+r*s2); 
            gp.Z(t)  = sqrt( (2*pi)/((1+r*s2)*gp.la(t)) ) * exp(-q^2/(2*r));  
        end

    % do projection on basis vectors to determine novelty of data point
        e   = gp.iK*k_t;                                % projection coordinates
        if numel(k_t)==0, gam=etol; else gam = k_tt-k_t'*e; end       % residual    
        
        if sweep<=1
            % geometrical test and parameter update    
            if gam<etol                      % not novel => update, novel => add
                gp = upd(gp,t,k_t,q,r,e,gam);     % update without including x_t
                if info, fprintf(' upd(%03d)',t), ct=ct+1; end
            else                  % add a new basis point x_t and enlarge (al,C)
                gp = add(gp,t,x_t,k_t,k_tt,q,r,e,gam);
                if info, fprintf(' add(%03d)',t), ct=ct+1; end
            end
            if info, fprintf('[%02d]',gp.n), if ct>5,fprintf('\n'),ct=0;end, end

            % remove basis vector with smallest score i.e. do the KL-optimal
            % projection; covariance term is ignored (Csató PhD thesis p. 45)
            if gp.n>gp.nmax                             % basis set is too large
                ep = gp.al.^2./(diag(gp.iK)+diag(gp.C));    % compute the scores
                [epr,tr]=min(ep);   % pick the vector(s) with the smallest score
                gp = rem(gp, tr);         % remove the vector(s) at positions tr
                if info, fprintf(' rem(%03d)[%02d]',tr,gp.n), ct=ct+1; end
            end
            if info, if ct>5, fprintf('\n'), ct=0; end, end
            
        else
            gp = upd(gp,t,k_t,q,r,e,gam); 
            if info,fprintf(' upd(%03d)',t), ct=ct+1; end
            if info,fprintf('[%02d]',gp.n), if ct>5,fprintf('\n'),ct=0; end, end
        end
        
    end
    
    if ~gp.ep, break; end
    if info, fprintf('\n'), end    
end

% sort indexes of basis vectors
[gp.id,idx]=sort(gp.id); gp.P=gp.P(:,idx); gp.al=gp.al(idx);
gp.K=gp.K(idx,idx); gp.iK=gp.iK(idx,idx); gp.C=gp.C(idx,idx); 
alpha = sparse(gp.id,1,gp.al,n,1); C=gp.C;                   % return parameters

% % check further consistency
% norm( gp.K*gp.iK - eye(gp.n) )
% norm( gp.P(gp.id,:)*gp.K - gp.K )
% Kfull=feval(covfunc{:},hyper,x); norm(Kfull(gp.id,gp.id)-gp.K)
% % check consistency using (4.34)
% norm( gp.al - gp.P'*((diag(1./gp.la)+gp.P*gp.K*gp.P')\gp.a) )
% norm( gp.C  + gp.P'*((diag(1./gp.la)+gp.P*gp.K*gp.P')\gp.P) )


if nargout > 3                                          % evidence approximation
    Kmm  = gp.K; P = gp.P; PKmm = P*Kmm; % K  = PKmm*P';
    LP   = repmat(gp.la,[1,gp.n]).*P;
    KLP  = PKmm*(P'*LP);
    
    % V = inv(inv(K) + diag(gp.la)); V = PKmm*almostV;
    almostV = P' - KLP' + P'*LP*inv(gp.iK+P'*LP)*KLP';        
    inv_diagV = 1./sum(PKmm.*almostV',2);
    m = PKmm*(almostV*(gp.la.*gp.a));

    si_n = 1./(inv_diagV-gp.la);
    mu_n = si_n.*(inv_diagV.*m-gp.la.*gp.a);
    Z0 = feval(lik, y, mu_n, si_n); 
    
    % logdet on small matrix only
    logdet_KpLa = logdet(eye(gp.n)+Kmm*P'*diag(gp.la)*P)-sum(log(gp.la));
    
    nlZ = - sum(log(Z0.*gp.Z/sqrt(2*pi)))              ...   K is Nyström approx
          + gp.al'*(Kmm+diag(1./gp.la(gp.id)))*gp.al/2 ... naturally sparse part
          + logdet_KpLa/2;
    
    if nargout > 4                                     % do we want derivatives?
        dnlZ = zeros(size(hyper));              % allocate space for derivatives
        sW = sqrt(gp.la(gp.id));
        L = chol(eye(gp.n)+sW*sW'.*Kmm);
        F = gp.al*gp.al'-repmat(sW,1,gp.n).*solve_chol(L,diag(sW)); 
        for j=1:length(hyper)
            dK = feval(covfunc{:}, hyper, x(gp.id,:), j);
            dnlZ(j) = -(F(:)'*dK(:))/2;
        end
    end
end


%% log(det(A)) for det(A)>0
function y = logdet(A)
    [L,U]=lu(A); 
    u=diag(U); 
    if prod(sign(u))~=det(L)
        error('det(A)<=0')
    end
    y=sum(log(abs(u))); % slower, but no symmetry needed 



%% remove some basis vectors indexed by tr from the GP
function gp = rem(gp, tr)  % tr: remove, tk: keep [1:gp.n]
    tk = setdiff(1:gp.n,tr);                             % basis vectors to keep
    gp.id = gp.id(tk);                                    % adjust the index set
    
    gp.BV = gp.BV(tk,:);    gp.n = gp.n-numel(tr);   % rem from basis vector set
    gp.K  = gp.K(tk,tk);                               % update prior covariance
    
    Q     = gp.iK(tk,tr);      q = gp.iK(tr,tr);    qQ = q\Q';     % auxiliaries
    QC    = Q+gp.C( tk,tr);   qc = q+gp.C( tr,tr); QQq = Q*qQ;              
    
    gp.al = gp.al(tk) - QC*(qc\gp.al(tr));                 % update coefficients
    gp.C  = gp.C( tk,tk) + QQq - QC*(qc\QC');
    gp.iK = gp.iK(tk,tk) - QQq;                % update inverse prior covariance

    gp.iK = (gp.iK+gp.iK')/2;                          % for numerical stability
    gp.C  = (gp.C+gp.C')/2;   
    
    if gp.ep, gp.P = gp.P(:,tk) - gp.P(:,tr)*qQ; end  % shrink projection matrix



%% update the GP with a single data point
function gp = upd(gp, t, k_t, q, r, e, gam)
    eta = 1/(1+gam*r);    

    s     = gp.C*k_t + e;                                   % auxiliary quantity    
    gp.al = gp.al + q*eta*s;                               % update coefficients
    gp.C  = gp.C  + s*r*eta*s';    

    gp.C  = (gp.C+gp.C')/2;                            % for numerical stability

    if gp.ep, gp.P(t,:) = e'; end              % update row of projection matrix
    


%% add a single basis vector to the GP
function gp = add(gp, t, x_t, k_t, k_tt, q, r, e, gam)
    gp.BV = [gp.BV; x_t]; gp.n = gp.n+1;               % add to basis vector set
    gp.id = [gp.id;t];                                      % also add the index
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
    
    if gp.ep, gp.P(t,end+1) = 1; end        % add a row to the projection matrix


