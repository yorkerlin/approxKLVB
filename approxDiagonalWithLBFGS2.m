function [alpha, sW, L, nlZ, dnlZ] = approxCholeskyWithLBFGS2(hyper, covfunc, lik, x, y)

% Approximation to the posterior Gaussian Process by minimization of the 
% KL-divergence. The function takes a specified covariance function (see 
% covFunction.m) and likelihood function (see likelihoods.m), and is designed to
% be used with binaryGP.m. See also approximations.m.
%
% Written by Hannes Nickisch, 2007-03-29

n = size(x,1);
assert (n==length(y))
K = feval(covfunc{:}, hyper.cov, x);                % evaluate the covariance matrix

alla_init{1} = [zeros(n,1); low_matrix_to_vector(eye(n))];                       % stack alpha/lambda together


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
alla    = alla_result{alla_id}                            % extract best result

%display the result
nlZ_new = min(nlZ_result);
alpha = alla(1:n,1)
C = vector_to_low_matrix(alla(n+1:end,1));

Sigma=C*C'

% bound on neg log marginal likelihood
nlZ = nlZ_result( alla_id);    
sW=[];
L=K\((K\Sigma-eye(n))')


%estimate the hpyer parameter
% do we want derivatives?
if nargout >=4                                     

	dnlZ = zeros(size(hyper.cov));                  % allocate space for derivatives
    C = vector_to_low_matrix(alla(n+1:end,1));
	% parameters after optimization
	m = K*alpha
	%A = (K\(C*C'))'
	%Sigma = A*K
	%v=abs(diag(A*K))
	v=sum(C.*C,2);
	invK_V=K\(C*C');

	%[a,dm,dC] = a_related2(m,v,y,lik,hyper);

	dnlZ = hyper.cov;                                   % allocate space for derivatives

	for j=1:length(hyper.cov)                                    % covariance hypers
		dK = feval(covfunc{:},hyper.cov,x,j)
		tmp1=(invK_V+alpha*m')
		tmp2=(eye(n)- tmp1)'
		tmp3=K\tmp2
		tmp4=dK.*tmp3
		ok=0.5*sum(tmp4(:))
		dnlZ(j)=0.5*sum(sum(dK.*(K\((eye(n)- (invK_V+alpha*m'))')),2),1);
	end
	dnlZ

	dnlZ_lik=zeros(size(hyper.lik));
	for j=1:length(hyper.lik)                                    % likelihood hypers
		lp_dhyp = likKL(v,lik,hyper.lik,y,m,[],[],j);
		dnlZ_lik(j) = -sum(lp_dhyp);
	end
	disp('dnlZ_lik=')
	sprintf('%.15f\n',dnlZ_lik)

  %for j=1:length(hyp.mean)                                         % mean hypers
	%dm_t = feval(mean{:}, hyp.mean, x, j);
	%dnlZ.mean(j) = -alpha'*dm_t;
  %end

end


%% evaluation of current negative log marginal likelihood depending on the
%  parameters alpha (al) and lambda (la)
function [nlZ,dnlZ] = margLik_cholesky(alla,K,y,lik,hpyer)

    % dimensions
    n  = length(y);
    % extract single parameters
    alpha  = alla(1:n,1);
    C = vector_to_low_matrix(alla(n+1:end,1));

	invK_C=K\C;
	VinvK=C*invK_C';                   %V = C*C'; K\V Sigma==V
	v     =sum(C.*C,2);
    m     = K*alpha;


    % calculate alpha related terms we need
    if nargout==1
       [a] = a_related2(m,v,y,lik,hpyer);
    else
	%done
       [a,dm,dV] = a_related2(m,v,y,lik,hpyer);
    end

    %negative Likelihood
    nlZ = -a -logdet(VinvK)/2 -n/2 +(alpha'*K*alpha)/2 +trace(VinvK)/2;

    if nargout>1 % gradient of Likelihood
        dlZ_alpha  = K*(alpha-dm);

		%par1=-2*(alla(n+1:end,1).*convert_dC(dV))
		%par2=low_matrix_to_vector(invK_C)
		dlZ_C=low_matrix_to_vector(invK_C)-convert_diag(1.0./diag(C))-2*(alla(n+1:end,1).*convert_dC(dV));
		dlZ_C=convert_diag(diag(vector_to_low_matrix(dlZ_C)));
        dnlZ = [dlZ_alpha; dlZ_C];
		%par3=dnlZ
    end
	nlZ


function [alla2 nlZ] = lbfgs(alla, K, y, lik, hyper,n)
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
	[alla2, nlZ] = minFunc(@margLik_cholesky, alla, optMinFunc, K, y, lik,hyper);



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
% using GPML 3.4 likelihood files
function [a,dm,dV,d2m,d2V,dmdV]=a_related2(m,v,y,lik,hyper)
	if nargout<4
		[a,dm,d2m,dV] = likKL(v, lik,hyper.lik,y,m);
		a = sum(a);
	else
		[a,dm,d2m,dV,d2V,dmdV] = likKL(v, lik,hyper.lik,y,m)
		a = sum(a);
	end

%came from GPML 3.4/infKL.m which uses likelihood files to compute terms like a_related
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

function res = low_matrix_to_vector(mat)
	assert(size(mat,1)==size(mat,2));
	n = size(mat,1);
	res = mat(find(tril(ones(n))));

function res = vector_to_low_matrix(vet)
	n = floor(sqrt(2*length(vet)));
	res = tril(ones(n));
	res(res==1) = vet;

function res = convert_dC(dv)
	res=dv;
	for i=1:(length(dv)-1)
		res=[res; dv(1+i:end)];
	end

function res = convert_diag(d)
	res =[];
	for i=1:length(d)
		res=[res;[d(i);zeros(length(d)-i,1)]];
	end
