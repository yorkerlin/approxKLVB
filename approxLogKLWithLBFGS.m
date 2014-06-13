function [alpha, sW, L, nlZ, dnlZ] = approxLogKLWithLBFGS(hyper, covfunc, lik, x, y)

% Approximation to the posterior Gaussian Process by minimization of the 
% KL-divergence. The function takes a specified covariance function (see 
% covFunction.m) and likelihood function (see likelihoods.m), and is designed to
% be used with binaryGP.m. See also approximations.m.
%
% Written by Hannes Nickisch, 2007-03-29

n = size(x,1);
K = feval(covfunc{:}, hyper.cov, x);                % evaluate the covariance matrix

% a) simply start at zero
%alla_init{1} = zeros(2*n,1);                       % stack alpha/lambda together
alla_init{1} = [zeros(n,1); ones(n,1)*log(0.5)];                       % stack alpha/lambda together

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

for alla_id = 1:length(alla_init)              % iterate over initial conditions

    alla = alla_init{alla_id};

    use_pinv=false; check_cond=true;
    nlZ_old = Inf; nlZ_new = 1e100; it=0;      % make sure the while loop starts

	[alla nlZ_new] = lbfgs(alla, K, y, lik, hyper);  %using L-BFGS to find the opt alla

    % save results
    alla_result{alla_id} = alla;
    nlZ_result( alla_id) = nlZ_new;
end

alla_id = find(nlZ_result==min(nlZ_result)); alla_id = alla_id(1);
alla    = alla_result{alla_id};                            % extract best result
alla

%display the result
nlZ_new = min(nlZ_result)
alla(end/2+1:end,1) = -exp(alla(end/2+1:end,1)); %convert log_neg_lambda to lambda




alpha = alla(1:end/2,1)
W  = -2*alla(end/2+1:end,1)

% recalculate L
sW = sqrt(W);                     
L  = chol(eye(n)+sW*sW'.*K)                             % L'*L=B=eye(n)+sW*K*sW 

% [nlZ_result,alla_id]

% bound on neg log marginal likelihood
nlZ = nlZ_result( alla_id);    

%estimate the hpyer parameter
% do we want derivatives?
if nargout >=4                                     

    dnlZ = zeros(size(hyper.cov));                  % allocate space for derivatives
    % parameters after optimization
    alpha  = alla(      1:end/2,1);
    lambda = alla(end/2+1:end  ,1);
    
    A = inv(  eye(n)-2*K*diag(lambda)  );

	Sigma = A*K

	alpha
	mu = K*alpha

    v=abs(diag(A*K))
	[a,dm,dC] = a_related2(K*alpha,v,y,lik,hyper);

    for j=1:length(hyper.cov)
        dK = feval(covfunc{:},hyper.cov,x,j);
		%           from the paper 
        %           -alpha'*dK*dm +(alpha'*dK*alpha)/2 -diag(A*dK*A')'*dC 
        %           -trace(A'*diag(lambda)*dK) +trace(A*dK*diag(lambda)*A)
		%           Note that lambda == dC
        AdK  = A*dK;
        dnlZ(j) = -(alpha'*dK*(dm-alpha/2) +sum(A.*AdK,2)'*dC                ...
                    +(diag(AdK)'-sum(A'.*AdK,1))*lambda);
    end


	dnlZ = hyper.cov;                                   % allocate space for derivatives

	for j=1:length(hyper.cov)                                    % covariance hypers
		dK = feval(covfunc{:},hyper.cov,x,j)
		%dK = feval(cov{:},hyp.cov,x,[],j);
		AdK = A*dK;
		tmp1=sum(A.*AdK,2)
		tmp2=sum(A'.*AdK,1)

		z = diag(AdK) + sum(A.*AdK,2) - sum(A'.*AdK,1)';
		%dnlZ(j) = alpha'*dK*(alpha/2-df) - z'*dv;
		dnlZ(j) = alpha'*dK*(alpha/2-dm) - z'*dC;
	end


	dnlZ_lik=zeros(size(hyper.lik));
	for j=1:length(hyper.lik)                                    % likelihood hypers
		lp_dhyp = likKL(v,lik,hyper.lik,y,K*alpha,[],[],j);
		dnlZ_lik(j) = -sum(lp_dhyp);
	end
	disp('dnlZ_lik=')
	sprintf('%.15f\n',dnlZ_lik)

  %for j=1:length(hyp.mean)                                         % mean hypers
	%dm_t = feval(mean{:}, hyp.mean, x, j);
	%dnlZ.mean(j) = -alpha'*dm_t;
  %end


  K

end


%% evaluation of current negative log marginal likelihood depending on the
%  parameters alpha (al) and lambda (la)
function [nlZ,dnlZ] = margLik_log(alla,K,y,lik,hpyer)
    % extract single parameters
    alpha  = alla(1:end/2,1);
    log_neg_lambda = alla(end/2+1:end,1);

	lambda = -exp(log_neg_lambda);
    % dimensions
    n  = length(y);
	%K
	%y

    % original variables instead of alpha and la
    VinvK = inv(eye(n)-2*K*diag(lambda));                          % A:=V*inv(K)
    V     = VinvK*K; V=(V+V')/2;                              % enforce symmetry
    v     = abs(diag(V));             % abs prevents numerically negative values
    m     = K*alpha;

    % calculate alpha related terms we need
    if nargout==1
       [a] = a_related2(m,v,y,lik,hpyer);
    else
	%done
       [a,dm,dV] = a_related2(m,v,y,lik,hpyer);
    end

	%res1=trace(VinvK)
	%W = abs(-2*lambda);
    %sW = sqrt(W); L = chol(eye(n)+sW*sW'.*K); 
	%L_inv=L\eye(n);
	%res2=trace(L_inv'*L_inv)
	%Note res1==res2

    %negative Likelihood
    nlZ = -a -logdet(VinvK)/2 -n/2 +(alpha'*K*alpha)/2 +trace(VinvK)/2;

    if nargout>1 % gradient of Likelihood
        dlZ_alpha  = K*(dm-alpha);
        dlZ_lambda = 2*(V.*V)*dV +v -sum(V.*VinvK,2);   % => fast diag(V*VinvK')
		dlZ_log_neg_lambda = dlZ_lambda .* lambda;

        % stack things together  
        dnlZ = -[dlZ_alpha; dlZ_log_neg_lambda];
    end



function [alla2 nlZ] = lbfgs(alla, K, y, lik, hyper)
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
	[alla2, nlZ] = minFunc(@margLik_log, alla, optMinFunc, K, y, lik,hyper);


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

%% compute all terms related to a
% derivatives w.r.t diag(V) and m, 2nd derivatives w.r.t diag(V) and m
function [a,dm,dV,d2m,d2V,dmdV]=a_related(m,v,y,lik)
    N = 20;                                % number of hermite quadrature points
    [f,w]  = gauher(N);            % location and weights for hermite quadrature
    f_dV   = f.^2-1;
    f_dm   = f;
    f_d2V  = f.^4-6*f.^2+3;
    f_dmdV = f.^3-3*f;
    SumLog_lam = zeros(size(f)); % init accumulator
    if nargout>2, dV  = zeros(size(m)); dm   = dV;  end            % init result
    if nargout>5, d2V = zeros(size(m)); dmdV = d2V; end            % init result
    for i=1:length(y)
        [dummy,log_lam] = feval( lik, y(i), (sqrt(v(i))*f+m(i)) );   % squashing 
        SumLog_lam = SumLog_lam+log_lam;                          % accumulation
        if nargout>2                                            % do integration
            dV(i)   = (w'*(log_lam.*f_dV  )) / (2*v(i)      );
            dm(i)   = (w'*(log_lam.*f_dm  )) /    v(i)^(1/2) ;
            if nargout>5
                d2V(i)  = (w'*(log_lam.*f_d2V )) / (4*v(i)^2    );
                dmdV(i) = (w'*(log_lam.*f_dmdV)) / (2*v(i)^(3/2));
            end
        end
    end
    a = w'*SumLog_lam;                                          % do integration
    if nargout>5, d2m= 2*dV; end

function [a,dm,dV,d2m,d2V,dmdV]=a_related2(m,v,y,lik,hyper)
	if nargout<4
		[a,dm,d2m,dV] = likKL(v, lik,hyper.lik,y,m);
		a = sum(a);
	else
		[a,dm,d2m,dV,d2V,dmdV] = likKL(v, lik,hyper.lik,y,m)
		a = sum(a);
	end
%add
function [ll,df,d2f,dv,d2v,dfdv] = likKL(v, lik, varargin)
  N = 20;                                          % number of quadrature points
  [t,w] = gauher(N);      % location and weights for Gaussian-Hermite quadrature
  f = varargin{3};                               % obtain location of evaluation
  sv = sqrt(v);                                                % smoothing width
  ll = 0; df = 0; d2f = 0; dv = 0; d2v = 0; dfdv = 0;    % init return arguments
  for i=1:N                                            % use Gaussian quadrature
    varargin{3} = f + sv*t(i);   % coordinate transform of the quadrature points
    [lp,dlp,d2lp] = feval(lik{:},varargin{1:3},[],'infLaplace',varargin{6:end});
    if nargout>0,     ll  = ll  + w(i)*lp;               % value of the integral
      if nargout>1,   df  = df  + w(i)*dlp;             % derivative w.r.t. mean
        if nargout>2, d2f = d2f + w(i)*d2lp;        % 2nd derivative w.r.t. mean
          if nargout>3                              % derivative w.r.t. variance
            ai = t(i)./(2*sv+eps); dvi = dlp.*ai; dv = dv + w(i)*dvi; % no 0 div
            if nargout>4                        % 2nd derivative w.r.t. variance
              d2v = d2v + w(i)*(d2lp.*(t(i)^2/2)-dvi)./(v+eps)/2;     % no 0 div
              if nargout>5                            % mixed second derivatives
                dfdv = dfdv + w(i)*(ai.*d2lp);
              end
            end
          end
        end
      end
    end
  end
