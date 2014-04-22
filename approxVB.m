function [alpha, sW, L, nlZ, dnlZ] = approxVB(hyper, covfunc, lik, x, y)

% Variational approximation to the posterior Gaussian Process for classification
% using the logistic or cumGauss likelihood function. The lower bound to be
% optimized for logistic regression was taken from Gibbs&McKay, Variational
% Gaussian Process Classifiers, IEEE NN, 2000
% The function takes a specified covariance function (see covFunction.m) and
% likelihood function (see likelihoods.m), and is designed to be used with
% binaryGP.m. See also approximations.m. 
%
% Written by Hannes Nickisch 2007-03-29

n = size(x,1);
K = feval(covfunc{:}, hyper, x);                % evaluate the covariance matrix
thresh = 200;                   % prevent infinite growing of variational params

% a) fixed initial values
switch lik
    case 'logistic' 
        nu_init{1}  = 2.5*ones(n,1);
    otherwise
        nu_init{1}  = 1.5*ones(n,1);
end

% b) initial values from Laplace approximation
nu = K*approxLA(hyper, covfunc, lik, x, y);
if strcmp(lik,'logistic')
    nu = abs(nu); % enforce symmetry for logistic
    nu(nu>thresh) = thresh;     % prevent infinite growing of variational params
end
nu_init{2}=nu;

% use only some inits
nu_init=nu_init(1);

max_it = 20;                               % maximum number of Newton iterations
tol = 1e-6;                   % tolerance for when to stop the Newton iterations

for nu_id = 1:length(nu_init)                  % iterate over initial conditions
    
    nu = nu_init{nu_id};
    
    nlZ_old = Inf; nlZ_new = 1e100; it=0;      % make sure the while loop starts

    while nlZ_new < nlZ_old - tol && it < max_it     % begin Newton's iterations

        nu_old   = nu;                                         % save old values
        [nlZ_old,dnlZ,d2nlZ] = derivs(nu,y,K,lik);    % calculate grad & Hessian

        nu = nu - d2nlZ\dnlZ;                    % update variational parameters
        if strcmp(lik,'logistic')
            nu = abs(nu); % enforce symmetry for logistic
            nu(nu>thresh) = thresh; % prevent inf. growing of variational params
        end
        nu = abs(nu);
        
        nlZ_new = derivs(nu,y,K,lik);                            % new objective

        i = 0;
        while i < 10 && nlZ_new > nlZ_old         % if objective didn't increase
            nu = (nu_old+nu)/2;                       % reduce step size by half
            nlZ_new = derivs(nu,y,K,lik);
            i = i+1;
        end
        if i==10 % give up
            nu = nu_old;
            nlZ_new = nlZ_old;
        end

        it=it+1;
    end

    if it == max_it
      disp('Warning: maximum number of iterations reached in function approxVB')
    end
    
    % check if some nu's have settled at the threshold
    % if sum(abs(nu)>=thresh)>0, disp('THRESHOLD met'), end
    
    % save results
    nu_result{ nu_id} = nu;
    nlZ_result(nu_id) = nlZ_new;
end

nu_id = find(nlZ_result==min(nlZ_result)); nu_id = nu_id(1);
nu    = nu_result{nu_id};                                  % extract best result

[a,da,dda, b,db,ddb, c,dc,ddc] = coeffs(nu,lik);
W  = -2*a;
sW = sqrt(W);                     
L  = chol(eye(n)+sW*sW'.*K);                             % L'*L=B=eye(n)+sW*K*sW
Kpred = diag(sW)*solve_chol(L,diag(sW));                     % inv(K+diag(1./W))
alpha = (eye(n)-Kpred*K)*(b.*y);

nlZ = nlZ_new;                            % bound on neg log marginal likelihood   

if nargout >=4                                         % do we want derivatives?                             
    [nlZ,dnlZ] = derivs_hyp(hyper,nu,y,K,x,covfunc,lik);
end


%% OBJECTIVE AND DERIVATIVES of a special approximation type
function [nlZ,dnlZ,d2nlZ,Kpred,Ktil] = derivs(nu,y,K,lik)

    %  # data points
    n = length(y); 

    % get coefficients depending on approximation type
    [a,da,dda, b,db,ddb, c,dc,ddc] = coeffs(nu,lik);
      
    % numerically stable calculation of 
    % Ktil   := inv( inv(K)-2*diag(a) )
    % Kpred  := inv( K -.5*diag(1./a) )
    % logdet := log(det(  eye(length(y))-2*diag(a)*K  ))
    if sum(a==0)>0 % A contains zeros on the diagonal
        Kpred  = zeros(size(K)); 
        Ktil   = K;
        logdet = 0;
    elseif sum(a>0)>0 % A contains positive entries on the diagonal
        Kpred  = inv(K-diag(.5./a));
        Ktil   = K-K*Kpred*K; 
        logdet = log(det(  eye(n)-2*diag(a)*K  )); % not symmetric
    else
        sW     = sqrt(-2*a);
        L      = chol(eye(n)+sW*sW'.*K);                 % L'*L=B=eye(n)+sW*K*sW
        Kpred  = repmat(sW,1,n).*solve_chol(L,diag(sW));     % inv(K+diag(1./W))
        Ktil   = L'\(repmat(sW,1,n).*K);
        Ktil   = K - Ktil'*Ktil;                                 % inv(inv(K)+W)       
        logdet = 2*sum(log(diag(L)));
    end
    
    % shortcuts
    btil    = y.*b;
    l       = Ktil*btil;
       
    % final calculations
    nlZ   = -sum(c) -(btil'*l)/2 +logdet/2;

    if nargout>1
        % shortcuts
        ydb    = y.*db;
        lda    = l.*da;
        
        % gradient w.r.t. variational parameters
        dnlZ  =   -dc  -l.*(ydb + lda)  -diag(Ktil).*da;
        if nargout>2
            ydb2lda = ydb+2*lda;
            
            % Hessian w.r.t. variational parameters
            d2nlZ  = -diag(ddc)  -(ydb2lda*ydb2lda').*Ktil                   ...
                     -diag( l.*(y.*ddb+l.*dda)+diag(Ktil).*dda )             ...
                     -((2*da)*da').*Ktil.^2;
        end
    end


%% OBJECTIVE AND DERIVATIVES w.r.t. to hyperparams for a special approximation
function [nlZ,dnlZ_hyp] = derivs_hyp(hyper,nu,y,K,x,covfunc,lik)
    
    n = length(y);                                              %  # data points
    [a,da,dda, b,db,ddb, c,dc,ddc] = coeffs(nu,lik);          % get coefficients
    [nlZ,dnlZ,d2nlZ,Kpred,Ktil] = derivs(nu,y,K,lik); % gradient & Hessian
    
    % shortcut
    btil    = y.*b;
    l       = Ktil*btil; 
           
    % inv(ddlZ)*dlZ
    v=(d2nlZ\dnlZ);
    
    dnlZ_hyp = zeros(size(hyper));
    for i=1:length(dnlZ_hyp)
        
        dKi = feval(covfunc{:},hyper,x,i);            % kernel matrix derivative
        
        Kpred_K = Kpred*K;
        
        % (K~) *inv(K) * dKi * inv(K) * (K~)
        Ktmp = (eye(n)-Kpred_K')*dKi*(eye(n)-Kpred_K);
               
        % explixit part                
        dnlZ_hyp(i)  = (-1/2)*btil'*Ktmp*btil ...
                         -sum(sum( (eye(n)-Kpred_K).*repmat(a',[n,1]).*dKi ));
                        %-trace( (eye(n)-Kpred_K) * diag(a) * dKi );
        
        % implicit part
        d2nlZ_nu_hyp= -da.*( 2*l.*(Ktmp*btil)+diag(Ktmp) )-db.*y.*(Ktmp*btil);
        dnlZ_hyp(i) = dnlZ_hyp(i) - v'*d2nlZ_nu_hyp;
         
    end

    
%% COEFFICIENTS FOR THE APPROXIMATION
function [a,da,dda, b,db,ddb, c,dc,ddc] = coeffs(nu,lik)

    switch lower(lik)
        case 'logistic' 
            % LOWER BOUND for LOGISTIC

            sig = feval(lik,[],nu); % Fermi sigmoid function
            % care for the case of very large negative values for which enu=Inf
            sig_enu = 1./(1+exp(nu));        % safe computation of sig.*exp(-nu)
         
            % take care of the limits nu -> 0
            ok = 1e-3<abs(nu);                                   % scale is O.K.    
            a  = zeros(size(nu));                                 % gap is 1e-14
            a( ok) = (1-2*sig(ok))./(4*nu(ok));              % normal evaluation
            a(~ok) = nu(~ok).^2/96-1/8;     % Taylor expansion up to third order
            
            ok = 1e-2<abs(nu);                      % scale is O.K.    
            da = zeros(size(nu));                   % gap is 5e-13
            da( ok) = -(1+2*sig(ok).*(nu(ok)-nu(ok).*sig(ok)-1)) ...
                      ./ (4*nu(ok).^2);                      % normal evaluation
            da(~ok) = nu(~ok)/48-nu(~ok).^3/240;    % Taylor series of 4th order

            dda = zeros(size(nu));                                % gap is 1e-11
            dda(ok) = -( nu(ok).*sig(ok).*(  sig_enu(ok).*(nu(ok)-1) ... normal
                                      + sig(ok).*(1-2*nu(ok).*sig_enu(ok))  )...
                      +2*sig(ok) -nu(ok).*sig(ok) -1 ) ./ (2*nu(ok).^3);  % eval           
            dda(~ok) = 1/48-nu(~ok).^2/80;  % Taylor expansion up to third order    
            
             b  = ones(size(nu))/2;
            db  = zeros(size(nu));
            ddb = zeros(size(nu));
             c  = -nu.^2.*a -nu/2 +log(sig+1e-100);% prevent log(0) as nu -> -oo
            dc  = sig_enu.*(1+nu/2.*sig) +sig/2 -3/4;
            ddc = sig_enu.^2 -sig_enu -2*a -4*nu.*da -nu.^2.*dda;
            
         case 'cumgauss'
            % LOWER BOUND for CUMULATIVE GAUSSIAN

            % Phi cumulative Gaussian sigmoid
            [dummy, log_Phi] = feval(lik,[],nu);                       % normcdf
                      
            % Gaussian divided by cumulative Gaussian           
            [dummy,Norm_Phi] = feval(lik,[],nu,'deriv');       % normpdf/normcdf
            
             a  = -ones(size(nu))/2;
            da  = zeros(size(nu));
            dda = zeros(size(nu));
             b  = nu+Norm_Phi;
            db  = 1-b.*Norm_Phi;
            ddb = (b.^2-db).*Norm_Phi;
             c  = (nu/2-b).*nu + log_Phi;
            dc  = -db.*nu;
            ddc = -db-nu.*ddb;

        otherwise
            error('unknown type of approximation or bound')
    end

