function [post nlZ dnlZ] = infMulLaplace(hyp, mean, cov, lik, x, y)

% Laplace approximation to the posterior Gaussian process.
% The function takes a specified covariance function (see covFunction.m) and
% likelihood function (see likFunction.m), and is designed to be used with
% gp.m. See also infFunctions.m.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2013-05-02.
%
% See also INFMETHODS.M.


inf = 'infLaplace';
n = size(x,1);
assert (size(y,1)==n);
C=max(y);
y_new=zeros(C*n,1);
for i=1:n
	y_new( n*(y(i)-1)+i )=1;
end
reshape(y_new,n,C)
y=y_new

K=feval(cov{:}, hyp.cov, x);
m=feval(mean{:}, hyp.mean, x);

assert (size(m,1)==n)
assert (size(K,2)==n && size(K,1)==n)

hyp.lik.n=n;
likfun = @(f) feval(lik{:},hyp.lik,y,f,[],inf);        % log likelihood function

alpha = zeros(C*n,1);                      % start at mean if sizes do not match

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% switch between optimisation methods
[alpha nlZ]= irls(alpha,m,K,likfun);                         % run optimisation
[Psi_new,dpsi,f,alpha,dlp,dpi,F] = Psi(alpha,m,K,likfun);

last_alpha = alpha;                                     % remember for next call
post.alpha = alpha;                           % return the posterior parameters
dnlZ=[];

alpha
% diagnose optimality
err = @(x,y) norm(x-y)/max([norm(x),norm(y),1]);   % we need to have alpha = dlp
dev = err(alpha,dlp)
if dev>1e-4, warning('Not at optimum %1.2e.',dev), end

E=[];
E_t=zeros(n,n);
%EK=[]
for i=1:C
	from=1+(i-1)*n;
	to=i*n;
	D=dpi(from:to);
	sD = sqrt(D); L = chol(eye(n)+sD*sD'.*K);      
	E_c=sD*sD'.*solve_chol(L,eye(n));
	E=[E;E_c];
	E_t=E_t+E_c;
	%EK=[V;E_c*K];
end
post.E=E;
M=chol(E_t);
post.M=M;

if nargout>2                                           % do we want derivatives?
  dnlZ = hyp;                                   % allocate space for derivatives

  %ok= M\(M'\(eye(n)));
  %ok = (ok+ok')./2;
  U=M'\(E');
  %V=M'\(EK');

  for i=1:length(hyp.cov)                                    % covariance hypers
	dK = feval(cov{:}, hyp.cov, x, [], i);
	dnlZ.cov(i)=0;
	part1=0;
	for j=1:C
		from=1+(j-1)*n;
		to=j*n;
		sub_alpha=alpha(from:to);
		% explicit part
		%dnlZ.cov(i)=dnlZ.cov(i)+sum(sum(E(from:to,:).*dK-E(from:to,:)*ok*E(from:to,:)'.*dK))/2 - sub_alpha'*dK*sub_alpha/2;
		sub_U=U(:,from:to);
		dnlZ.cov(i)=dnlZ.cov(i)+sum(sum(E(from:to,:).*dK-sub_U'*sub_U.*dK))/2 - sub_alpha'*dK*sub_alpha/2;
	end
  end

  %for i=1:length(hyp.lik)                                    % likelihood hypers
	%[lp_dhyp,dlp_dhyp,d2lp_dhyp] = feval(lik{:},hyp.lik,y,f,[],inf,i);
	%dnlZ.lik(i) = -g'*d2lp_dhyp - sum(lp_dhyp);                  % explicit part
  %end

  %ok
  for i=1:length(hyp.mean)                                         % mean hypers
	dm = feval(mean{:}, hyp.mean, x, i);
	dnlZ.mean(i) =0;
	for j=1:C
		from=1+(j-1)*n;
		to=j*n;
		sub_alpha=alpha(from:to);
		dnlZ.mean(i) = dnlZ.mean(i) - sub_alpha'*dm;               % explicit part
	end
  end

end

% Evaluate criterion Psi(alpha) = alpha'*K*alpha + likfun(f), where 
% f = K*alpha+m, and likfun(f) = feval(lik{:},hyp.lik,y,  f,  [],inf).
function [psi,dpsi,f,alpha,dlp,dpi,F] = Psi(alpha,m,K,likfun)
	n=size(m,1);
	assert (size(K,1)==n && size(K,2)==n)
	C=size(alpha,1)/n;
	f=[];
	psi=[];
	for i=1:C
		from=1+(i-1)*n;
		to=i*n;
		sub_alpha=alpha(from:to);
		tmp=K*sub_alpha+m;
		f=[f;tmp];
		psi=[psi;sub_alpha'*(tmp-m)/2];
	end

	[lp,sdlp,sd2lp] = likfun(f);
	psi = sum(psi)-sum(lp);
	F = sd2lp.f;
	dpi=sdlp.pi;
	dlp=sdlp.value;
	if nargout>1,
		dpsi=[];
		tmp=alpha-sdlp.value;
		for i=1:C
			from=1+(i-1)*n;
			to=i*n;
			dpsi=[dpsi; K*tmp(from:to)];
		end
	end



% Run IRLS Newton algorithm to optimise Psi(alpha).
function [alpha nlZ]=irls(alpha, m,K,likfun)
	maxit = 50;
	tol = 1e-12;

	smin_line = 0; smax_line = 2;           % min/max line search steps size range
	nmax_line = 10;                          % maximum number of line search steps
	thr_line = 1e-4;                                       % line search threshold
	Psi_line = @(s,alpha,dalpha) Psi(alpha+s*dalpha, m,K,likfun);    % line search
	pars_line = {smin_line,smax_line,nmax_line,thr_line};  % line seach parameters
	search_line = @(alpha,dalpha) brentmin(pars_line{:},Psi_line,6,alpha,dalpha);

	[Psi_new,dpsi,f,alpha,dlp,dpi,F] = Psi(alpha,m,K,likfun);
	n = size(K,1);
	C=size(alpha,1)/n;

	Psi_old = Inf;  % make sure while loop starts by the largest old objective val
	it = 0;                          % this happens for the Student's t likelihood
	z=0;
	
	while Psi_old - Psi_new > tol && it<maxit                       % begin Newton
		Psi_old = Psi_new; it = it+1;
		z=0;
		E=[];
		E_t=zeros(n,n);

		for i=1:C
			from=1+(i-1)*n;
			to=i*n;
			D=dpi(from:to);
			sD = sqrt(D); L = chol(eye(n)+sD*sD'.*K);      
			E_c=sD*sD'.*solve_chol(L,eye(n));
			E=[E;E_c];
			E_t=E_t+E_c;
			z=z+sum(log(diag(L)));
		end

		M=chol(E_t);
		z=z+sum(log(diag(M)));

		nlZ=z+Psi_new

		%K_2=blkdiag(K,K);
		%H=inv(K_2)+diag(dpi)-F*F';
		%aaaa=0.5*(log(det(H))+log(det(K_2)))
		%bbbb=z

		b = (diag(dpi)-F*F')*(f-repmat(m,C,1))+dlp;
		c=[];
		R=[];


		for i=1:C
			from=1+(i-1)*n;
			to=i*n;
			c=[c;E(from:to,:)*K*b(from:to)]; % EKb
			R=[R;eye(n)];
		end
		dalpha = b - c + E*solve_chol(M,R'*c) - alpha;

		[s_line,Psi_new,n_line,dPsi_new,f,alpha,dlp,dpi,F] = search_line(alpha,dalpha);

	end                                                  % end Newton's iterations



% Compute the log determinant ldA and the inverse iA of a square nxn matrix
% A = eye(n) + K*diag(w) from its LU decomposition; for negative definite A, we 
% return ldA = Inf. We also return mwiA = -diag(w)/A.
function [ldA,iA,mwiA] = logdetA(K,w)
  [m,n] = size(K); if m~=n, error('K has to be nxn'), end
  A = eye(n)+K.*repmat(w',n,1);
  [L,U,P] = lu(A); u = diag(U);           % compute LU decomposition, A = P'*L*U
  signU = prod(sign(u));                                             % sign of U
  detP = 1;                 % compute sign (and det) of the permutation matrix P
  p = P*(1:n)';
  for i=1:n                                                       % swap entries
    if i~=p(i), detP = -detP; j = find(p==i); p([i,j]) = p([j,i]); end
  end
  if signU~=detP  % log becomes complex for negative values, encoded by infinity
    ldA = Inf;
  else            % det(L) = 1 and U triangular => det(A) = det(P)*prod(diag(U))
    ldA = sum(log(abs(u)));
  end 
  if nargout>1, iA = U\(L\P); end               % return the inverse if required
  if nargout>2, mwiA = -repmat(w,1,n).*iA; end
