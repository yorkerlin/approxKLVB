function [varargout] = mgp(hyp, inf, meanf, cov, lik, x, y, xs, ys)
% Gaussian Process inference and prediction. The gp function provides a
% flexible framework for Bayesian inference and prediction with Gaussian
% processes for scalar targets, i.e. both regression and binary
% classification. The prior is Gaussian process, defined through specification
% of its meanf and covariance function. The likelihood function is also
% specified. Both the prior and the likelihood may have hyperparameters
% associated with them.
%
% Two modes are possible: training or prediction: if no test cases are
% supplied, then the negative log marginal likelihood and its partial
% derivatives w.r.t. the hyperparameters is computed; this mode is used to fit
% the hyperparameters. If test cases are given, then the test set predictive
% probabilities are returned. Usage:
%
%   training: [nlZ dnlZ          ] = gp(hyp, inf, meanf, cov, lik, x, y);
% prediction: [ymu ys2 fmu fs2   ] = gp(hyp, inf, meanf, cov, lik, x, y, xs);
%         or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, meanf, cov, lik, x, y, xs, ys);
%
% where:
%
%   hyp      column vector of hyperparameters
%   inf      function specifying the inference method 
%   cov      prior covariance function (see below)
%   meanf     prior meanf function
%   lik      likelihood function
%   x        n by D matrix of training inputs
%   y        column vector of length n of training targets
%   xs       ns by D matrix of test inputs
%   ys       column vector of length nn of test targets
%
%   nlZ      returned value of the negative log marginal likelihood
%   dnlZ     column vector of partial derivatives of the negative
%               log marginal likelihood w.r.t. each hyperparameter
%   ymu      column vector (of length ns) of predictive output meanfs
%   ys2      column vector (of length ns) of predictive output variances
%   fmu      column vector (of length ns) of predictive latent meanfs
%   fs2      column vector (of length ns) of predictive latent variances
%   lp       column vector (of length ns) of log predictive probabilities
%
%   post     struct representation of the (approximate) posterior
%            3rd output in training mode or 6th output in prediction mode
%            can be reused in prediction mode gp(.., cov, lik, x, post, xs,..)
% 
% See also covFunctions.m, infMethods.m, likFunctions.m, meanfFunctions.m.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2014-03-04.
%                                      File automatically generated using noweb.
if nargin<7 || nargin>9
  disp('Usage: [nlZ dnlZ          ] = gp(hyp, inf, meanf, cov, lik, x, y);')
  disp('   or: [ymu ys2 fmu fs2   ] = gp(hyp, inf, meanf, cov, lik, x, y, xs);')
  disp('   or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, meanf, cov, lik, x, y, xs, ys);')
  return
end

if isempty(meanf), meanf = {@meanfZero}; end                     % set default meanf
if ischar(meanf) || isa(meanf, 'function_handle'), meanf = {meanf}; end  % make cell
if isempty(cov), error('Covariance function cannot be empty'); end  % no default
if ischar(cov)  || isa(cov,  'function_handle'), cov  = {cov};  end  % make cell
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if isempty(inf)                                   % set default inference method
  if strcmp(cov1,'covFITC'), inf = @infFITC; else inf = @infExact; end
else
  if iscell(inf), inf = inf{1}; end                      % cell input is allowed
  if ischar(inf), inf = str2func(inf); end        % convert into function handle
end
if strcmp(cov1,'covFITC')                           % only infFITC* are possible
  if isempty(strfind(func2str(inf),'infFITC')==1)
    error('Only infFITC* are possible inference algorithms')
  end
end                            % only one possible class of inference algorithms
if isempty(lik),  lik = {@likGauss}; end                       % set default lik
if ischar(lik)  || isa(lik,  'function_handle'), lik  = {lik};  end  % make cell
if iscell(lik), likstr = lik{1}; else likstr = lik; end
if ~ischar(likstr), likstr = func2str(likstr); end

D = size(x,2);

if ~isfield(hyp,'meanf'), hyp.meanf = []; end        % check the hyp specification
if eval(feval(meanf{:})) ~= numel(hyp.meanf)
  error('Number of meanf function hyperparameters disagree with meanf function')
end
if ~isfield(hyp,'cov'), hyp.cov = []; end
if eval(feval(cov{:})) ~= numel(hyp.cov)
  error('Number of cov function hyperparameters disagree with cov function')
end
if ~isfield(hyp,'lik'), hyp.lik = []; end
if eval(feval(lik{:})) ~= numel(hyp.lik)
  error('Number of lik function hyperparameters disagree with lik function')
end

try                                                  % call the inference method
  if nargin>7   % compute marginal likelihood and its derivatives only if needed
    if isstruct(y)
      post = y;            % reuse a previously computed posterior approximation
    else
	  %prediction
      post = inf(hyp, meanf, cov, lik, x, y);
    end
  else
    if nargout<=1
      [post nlZ] = inf(hyp, meanf, cov, lik, x, y); dnlZ = {};
    else
      [post nlZ dnlZ] = inf(hyp, meanf, cov, lik, x, y);
    end
  end
catch
  msgstr = lasterr;
  if nargin > 7, error('Inference method failed [%s]', msgstr); else 
    warning('Inference method failed [%s] .. attempting to continue',msgstr)
    dnlZ = struct('cov',0*hyp.cov, 'meanf',0*hyp.meanf, 'lik',0*hyp.lik);
    varargout = {NaN, dnlZ}; return                    % continue with a warning
  end
end

if nargin==7                                     % if no test cases are provided
  varargout = {nlZ, dnlZ, post};    % report -log marg lik, derivatives and post
else
  %prediction
  alpha = post.alpha; E = post.E; M = post.M;
  n=size(x,1);
  ns = size(xs,1);                                       % number of data points
  nperbatch = 1;                       % number of data points per mini batch
  nact = 0;                       % number of already processed test data points
  C=max(y);


  nt=size(xs,1);
  yt=ones(nt,C);

  ymu=[];
  ys2=[];
  lp=[];
  fmu=[];
  fs2=[];
  while nact<ns               % process minibatches of test cases to save memory
    id = (nact+1):min(nact+nperbatch,ns);               % data points to process
    kss = feval(cov{:}, hyp.cov, xs(id,:), 'diag');     % self-variance
    Ks  = feval(cov{:}, hyp.cov, x, xs(id,:));          % cross-covariances
    ms = feval(meanf{:}, hyp.meanf, xs(id,:));

	Fmu=[];
	Fs2=zeros(C,C);
	S=10000;
    for i=1:C
	   from=1+(i-1)*n;
	   to=i*n;
	   sub_alpha=alpha(from:to);
	   Fmu = [Fmu; ms + Ks'*sub_alpha];        % conditional meanf fs|f
	   Ei = E(from:to,:);
	   bi = Ei*Ks;
	   c_cav = M\(M'\bi);
	   for j=1:C,
	       fromj=1+(j-1)*n;
	       toj=j*n;
		   Ej = E(fromj:toj,:);
           bj=Ej*Ks;
		   Fs2(j,i)=bj'*c_cav;
	   end

	   Fs2(i,i)=kss+Fs2(i,i)-Ks'*bi;
	end
	fmu=[fmu;Fmu(:)'];
	Fs2 = (Fs2+Fs2')/2;
	fs2=[fs2; Fs2(:)'];

    if nargin>7
		f_star=mvnrnd(Fmu, Fs2, S);
		tmp = exp(f_star);
		tmp = tmp./(sum(tmp, 2)*ones(1,size(tmp,2)));
		pi=mean(tmp);
		ytmp = repmat(yt(id,:),S,1);
		lpy = log(mean(tmp.^(ytmp).*(1-tmp).^(1-ytmp)));
		if nargout > 1
			%Ey = 2*pi-1; %if y is -1 and 1 encoding   bernoulli
			%Vary = 1-(2*pi-1).^2;%if y is -1 and 1 encoding  bernoulli

			Ey = pi; %if y is 0 and 1 encoding  bernoulli
			Vary = pi.*(1.0-pi);%if y is 0 and 1 encoding  bernoulli

			Ey=Ey(:);
			ymu=[ymu; Ey'];
			Vary=Vary(:);
			ys2=[ys2;Vary'];
		end
		lpy=lpy(:);
		lp=[lp;lpy'];
	end
    nact = id(end);          % set counter to index of last processed data point
  end
  varargout = {ymu, ys2, fmu, fs2, lp, post};
end
