function [alpha, sW, L, nlZ, dnlZ] = approxLogPiecesVB(hyper, covfunc, lik, x, y)

% Approximation to the posterior Gaussian Process by minimization of the 
% KL-divergence. The function takes a specified covariance function (see 
% covFunction.m) and likelihood function (see likelihoods.m), and is designed to
% be used with binaryGP.m. See also approximations.m.
%
% Written by Hannes Nickisch, 2007-03-29

n = size(x,1);
K = feval(covfunc{:}, hyper, x);                % evaluate the covariance matrix

% a) simply start at zero
alla_init{1} = zeros(2*n,1);                       % stack alpha/lambda together

% b) initial values based on random values heuristic
%alla_init{2} = [y.*rand(n,1); -abs(randn(n,1))]/5000;     

% c) start from Laplace approximation
%[alpha,sW] = approxLA(hyper, covfunc, lik, x, y);
%alla_init{3} = [alpha; -sW.^2/2];

% d) use order of magnitude from Laplace starting point
%alla_init{4} = [y*min(abs(alpha)); ones(n,1)*(-mean(abs(sW))^2/2)];

% use only some inits
%alla_init=alla_init([1,3]);
alla_init=alla_init([1]);


bounds = load('llp.mat'); %load the pre-defined bound
bound = bounds.bound;
y_conv = y>0; %Note that the ElogLik function use {0, 1} as labels

for alla_id = 1:length(alla_init)              % iterate over initial conditions

    alla = alla_init{alla_id};

    use_pinv=false; check_cond=true;
    nlZ_old = Inf; nlZ_new = 1e100; it=0;      % make sure the while loop starts

    [alla nlZ_new] = lbfgs_VB(alla, K, y, y_conv, lik, bound); %use the lbfgs to find the opt alla

    % save results
    alla_result{alla_id} = alla;
    nlZ_result( alla_id) = nlZ_new;

end

alla_id = find(nlZ_result==min(nlZ_result)); alla_id = alla_id(1);
alla    = alla_result{alla_id};                            % extract best result

%display the result
nlZ_new = min(nlZ_result)
alla(end/2+1:end,1) = -exp(alla(end/2+1:end,1)); %convert log_neg_lambda to lambda
alla

alpha = alla(1:end/2,1)
W  = -2*alla(end/2+1:end,1)
%W  = 2*exp(alla(end/2+1:end,1));

% recalculate L
sW = sqrt(W);                     
L  = chol(eye(n)+sW*sW'.*K);                             % L'*L=B=eye(n)+sW*K*sW 

% [nlZ_result,alla_id]

% bound on neg log marginal likelihood
nlZ = nlZ_result( alla_id);       

%estimate the hpyer parameter
% do we want derivatives?
if nargout >=4                                     
    dnlZ = zeros(size(hyper));                  % allocate space for derivatives

    % parameters after optimization
    alpha  = alla(      1:end/2,1);
    lambda = alla(end/2+1:end  ,1);
    
    A = inv(  eye(n)-2*K*diag(lambda)  );

    [as, dm, dC] = ElogLik('bernLogit', y_conv, K*alpha, abs(diag(A*K)), bound);
    a = sum(as);


    for j=1:length(hyper)
        dK = feval(covfunc{:},hyper,x,j);
%           from the paper 
        %           -alpha'*dK*dm +(alpha'*dK*alpha)/2 -diag(A*dK*A')'*dC 
        %           -trace(A'*diag(lambda)*dK) +trace(A*dK*diag(lambda)*A)
        AdK  = A*dK;
        dnlZ(j) = -(alpha'*dK*(dm-alpha/2) +sum(A.*AdK,2)'*dC                ...
                    +(diag(AdK)'-sum(A'.*AdK,1))*lambda);
    end
end

%% evaluation of current negative log marginal likelihood using the piecewise bound depending on the
%  parameters alpha (al) and lambda (la)
function [nlZ,dnlZ] = margLik_Log_VB(alla,K,y,y_conv,lik, param)
    % extract single parameters
    alpha  = alla(1:end/2,1);
    log_neg_lambda = alla(end/2+1:end,1);
	lambda = -exp(log_neg_lambda);
    % dimensions
    n  = length(y);

    % original variables instead of alpha and la
    VinvK = inv(eye(n)-2*K*diag(lambda));                          % A:=V*inv(K)
    V     = VinvK*K; V=(V+V')/2;                              % enforce symmetry
    v     = abs(diag(V));             % abs prevents numerically negative values
    m     = K*alpha;
    
	%compute the function value and gradient using the piecewise bound 
    [as, dm, dV] = ElogLik('bernLogit', y_conv, m, v, param);
    a = sum(as);
    
    %negative Likelihood
    nlZ = -a -logdet(VinvK)/2 -n/2 +(alpha'*K*alpha)/2 +trace(VinvK)/2;

    if nargout>1 % gradient of Likelihood
        dlZ_alpha  = K*(dm-alpha);
        dlZ_lambda = 2*(V.*V)*dV +v -sum(V.*VinvK,2);   % => fast diag(V*VinvK')
		dlZ_log_neg_lambda = dlZ_lambda .* lambda;

        % stack things together  
        %dnlZ = -[dlZ_alpha; dlZ_lambda];
        dnlZ = -[dlZ_alpha; dlZ_log_neg_lambda];
    end



%% evaluation of current negative log marginal likelihood using the piecewise bound depending on the
%  parameters alpha (al) and lambda (la)
function [nlZ,dnlZ] = margLik_VB(alla,K,y,y_conv,lik, param)
    % extract single parameters
    alpha  = alla(1:end/2,1);
    lambda = alla(end/2+1:end,1);
    % dimensions
    n  = length(y);

    % original variables instead of alpha and la
    VinvK = inv(eye(n)-2*K*diag(lambda));                          % A:=V*inv(K)
    V     = VinvK*K; V=(V+V')/2;                              % enforce symmetry
    v     = abs(diag(V));             % abs prevents numerically negative values
    m     = K*alpha;
    
	%compute the function value and gradient using the piecewise bound 
    [as, dm, dV] = ElogLik('bernLogit', y_conv, m, v, param);
    a = sum(as);
    
    %negative Likelihood
    nlZ = -a -logdet(VinvK)/2 -n/2 +(alpha'*K*alpha)/2 +trace(VinvK)/2;

    if nargout>1 % gradient of Likelihood
        dlZ_alpha  = K*(dm-alpha);
        dlZ_lambda = 2*(V.*V)*dV +v -sum(V.*VinvK,2);   % => fast diag(V*VinvK')

        % stack things together  
        dnlZ = -[dlZ_alpha; dlZ_lambda];
    end


function [alla nlZ] = lbfgs_VB(alla, K, y, y_conv, lik, param)
    optMinFunc = struct('Display', 'FULL',...
    'Method', 'lbfgs',...
    'DerivativeCheck', 'off',...
    'LS_type', 1,...
    'MaxIter', 1000,...
    'LS_interp', 1,...
    'MaxFunEvals', 1000000,...
    'Corr' , 100,...
    'optTol', 1e-15,...
    'progTol', 1e-15);
    [alla, nlZ] = minFunc(@margLik_Log_VB, alla, optMinFunc, K, y, y_conv, lik, param);




%% log(det(A)) for det(A)>0
function y = logdet(A)
    % 1) y=det(A); if det(A)<=0, error('det(A)<=0'), end, y=log(y); 
    %   => naive implementation, not numerically stable
    % 2) U=chol(A); y=2*sum(log(diag(U))); 
    %   => fast, but works for symmetric p.d. matrices only
    % 3) det(A)=det(L)*det(U)=det(L)*prod(diag(U)) 
    %   => logdet(A)=log(sum(log(diag(U)))) if det(A)>0
    [L,U]=lu(A); 
    u=diag(U); 
    if prod(sign(u))~=det(L)
        error('det(A)<=0')
    end
    y=sum(log(abs(u))); % slower, but no symmetry needed 
    % 4) d=eig(A); if prod(sign(d))<1, error('det(A)<=0'), end
    %    y=sum(log(d)); y=real(y); 
    %   => slowest
