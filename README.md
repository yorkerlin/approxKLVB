A SimpleKL method with the variational piecewise bound
========================

The following code is based on the KL method of <a href=http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.8505>Hannes Nickisch , Carlos Guestrin, 2008</a> using the variational piecewise bound obtained from <a href=https://github.com/emtiyaz/VariationalApproxExample> VariationalApproxExample</a>

The following files are created based on the code,approxKL.m.
approxKLWithLBFGS.m
approxPiecesVB.m

You can run the following examples using GPML 2.0 (is included in this repo)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;
x1=[0.8822936
-0.7160792
0.9178174
-0.0135544
-0.5275911];

x2=[-0.9597321
0.0231289
0.8284935
0.0023812
-0.7218931];

x=[x1 x2];

y=[1
-1
1
-1
-1];

hyp.cov = log([2; 2]);
cov = {'covSEiso'};
lik = 'logistic';

%for Laplace method
apx = 'LA';      
[nlZ dnlZ          ] = binaryGP(hyp.cov, ['approx',apx], cov, lik, x, y) 

%for original KL method (which uses Newton method to find opt parameters)
apx = 'KL';      
[nlZ dnlZ          ] = binaryGP(hyp.cov, ['approx',apx], cov, lik, x, y) 


%for KL method with L-BFGS
apx = 'KLWithLBFGS';      
[nlZ dnlZ          ] = binaryGP(hyp.cov, ['approx',apx], cov, lik, x, y) 


%for KL method using piecewise bound with L-BFGS
apx = 'PiecesVB';      
[nlZ dnlZ          ] = binaryGP(hyp.cov, ['approx',apx], cov, lik, x, y) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Output for the Laplace method from GPML 2.0 is
nlZ =

    3.7607


dnlZ =

    0.6037
    0.1888


Output for the original KL method from GPML 2.0 is
nlZ =

    3.6869


dnlZ =

    0.6462
    0.0924
	
Output for the KL method with L-BFGS from GPML 2.0 is
nlZ =

    3.6869


dnlZ =

    0.6462
    0.0924
	

Output for the KL method using piecewise bound with L-BFGS from GPML 2.0 is
nlZ =

    3.6873


dnlZ =

    0.6461
    0.0925
	



The corresponding code of Laplace method for GPML 3.4 is (Note that you need to install the GPML 3.4 package first)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;
meanfunc = @meanZero; hyp.mean=[];
covfunc = @covSEiso; hyp.cov = log([2.0 2.0]);
likfunc = @likLogistic; hyp.lik=[];

x1=[0.8822936
-0.7160792
0.9178174
-0.0135544
-0.5275911];

x2=[-0.9597321
0.0231289
0.8284935
0.0023812
-0.7218931];

x=[x1 x2];

y=[1
-1
1
-1
-1];

inf = @infLaplace;
covfunc = {covfunc};
meanfunc = {meanfunc};
likfunc = {likfunc};
[nlZ dnlZ          ] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, x, y); %training 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Output for the Laplace method from GPML 3.4 is

nlZ =

    3.7607


dnlZ = 

    mean: []
     cov: [0.6037 0.1888]
     lik: []



We use LBFGS implemented in minFunc by Mark Schmidt to estimate the parameters
http://www.di.ens.fr/~mschmidt/Software/minFunc.html

The implementation of variational piecewise bound for logit likelihood is based on the following
<a href=http://www.cs.ubc.ca/~emtiyaz/papers/paper-ICML2011.pdf>ICML paper</a> (also see the 
<a href=http://www.cs.ubc.ca/~emtiyaz/papers/truncatedGaussianMoments.pdf>Appendix</a>
)

The implementation of the KL method is based on the following code from http://hannes.nickisch.org/
<a href=http://hannes.nickisch.org/code/approxXX.tar.gz>Matlab code</a>,
which is used the <a href=http://www.gaussianprocess.org/gpml/code/matlab/release/gpml-matlab-v2.0-2007-06-25.tar.gz>GPML 2.0 package</a> 
