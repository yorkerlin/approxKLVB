A SimpleKL method with the variational piecewise bound
========================

The following code is based on the KL method of <a href=http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.8505>Hannes Nickisch , Carlos Guestrin, 2008</a> using the variational piecewise bound obtained from <a href=https://github.com/emtiyaz/VariationalApproxExample> VariationalApproxExample</a>

The following files are created based on the code, approxKL.m.
<br>
approxKLWithLBFGS.m (modified, based on approxKL.m)
<br>
approxPiecesVB.m (modified, based on approxKL.m)
<br>

You can run the following examples using GPML 2.0 (included in this repo)
<PRE>clear all<B><FONT COLOR="#663300">;</FONT></B> close all<B><FONT COLOR="#663300">;</FONT></B>

x1<B><FONT COLOR="#663300">=[</FONT></B><FONT COLOR="#996600">0.8822936</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#996600">0.7160792
0.9178174</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#996600">0.0135544</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#996600">0.5275911</FONT><B><FONT COLOR="#663300">];</FONT></B>

x2<B><FONT COLOR="#663300">=[-</FONT></B><FONT COLOR="#996600">0.9597321
0.0231289
0.8284935
0.0023812</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#996600">0.7218931</FONT><B><FONT COLOR="#663300">];</FONT></B>

x<B><FONT COLOR="#663300">=[</FONT></B>x1 x2<B><FONT COLOR="#663300">];</FONT></B>

y<B><FONT COLOR="#663300">=[</FONT></B><FONT COLOR="#999900">1</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#999900">1
1</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#999900">1</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#999900">1</FONT><B><FONT COLOR="#663300">];</FONT></B>

hyp<B><FONT COLOR="#663300">.</FONT></B>cov<B><FONT COLOR="#663300"> =</FONT></B> log<B><FONT COLOR="#663300">([</FONT></B><FONT COLOR="#999900">2</FONT><B><FONT COLOR="#663300">;</FONT></B><FONT COLOR="#999900"> 2</FONT><B><FONT COLOR="#663300">]);</FONT></B>
cov<B><FONT COLOR="#663300"> = {</FONT></B><FONT COLOR="#009900">'covSEiso'</FONT><B><FONT COLOR="#663300">};</FONT></B>
lik<B><FONT COLOR="#663300"> =</FONT></B><FONT COLOR="#009900"> 'logistic'</FONT><B><FONT COLOR="#663300">;

%</FONT></B><FONT COLOR="#FF0000">for</FONT> Laplace method
apx<B><FONT COLOR="#663300"> =</FONT></B><FONT COLOR="#009900"> 'LA'</FONT><B><FONT COLOR="#663300">;      
[</FONT></B>nlZ dnlZ<B><FONT COLOR="#663300">          ] =</FONT></B> binaryGP<B><FONT COLOR="#663300">(</FONT></B>hyp<B><FONT COLOR="#663300">.</FONT></B>cov<B><FONT COLOR="#663300">, [</FONT></B><FONT COLOR="#009900">'approx'</FONT><B><FONT COLOR="#663300">,</FONT></B>apx<B><FONT COLOR="#663300">],</FONT></B> cov<B><FONT COLOR="#663300">,</FONT></B> lik<B><FONT COLOR="#663300">,</FONT></B> x<B><FONT COLOR="#663300">,</FONT></B> y<B><FONT COLOR="#663300">) 

%</FONT></B><FONT COLOR="#FF0000">for</FONT> original KL method<B><FONT COLOR="#663300"> (</FONT></B>which uses Newton method to find opt parameters<B><FONT COLOR="#663300">)</FONT></B>
apx<B><FONT COLOR="#663300"> =</FONT></B><FONT COLOR="#009900"> 'KL'</FONT><B><FONT COLOR="#663300">;      
[</FONT></B>nlZ dnlZ<B><FONT COLOR="#663300">          ] =</FONT></B> binaryGP<B><FONT COLOR="#663300">(</FONT></B>hyp<B><FONT COLOR="#663300">.</FONT></B>cov<B><FONT COLOR="#663300">, [</FONT></B><FONT COLOR="#009900">'approx'</FONT><B><FONT COLOR="#663300">,</FONT></B>apx<B><FONT COLOR="#663300">],</FONT></B> cov<B><FONT COLOR="#663300">,</FONT></B> lik<B><FONT COLOR="#663300">,</FONT></B> x<B><FONT COLOR="#663300">,</FONT></B> y<B><FONT COLOR="#663300">) 

%</FONT></B><FONT COLOR="#FF0000">for</FONT> KL method with L<B><FONT COLOR="#663300">-</FONT></B>BFGS
apx<B><FONT COLOR="#663300"> =</FONT></B><FONT COLOR="#009900"> 'KLWithLBFGS'</FONT><B><FONT COLOR="#663300">;      
[</FONT></B>nlZ dnlZ<B><FONT COLOR="#663300">          ] =</FONT></B> binaryGP<B><FONT COLOR="#663300">(</FONT></B>hyp<B><FONT COLOR="#663300">.</FONT></B>cov<B><FONT COLOR="#663300">, [</FONT></B><FONT COLOR="#009900">'approx'</FONT><B><FONT COLOR="#663300">,</FONT></B>apx<B><FONT COLOR="#663300">],</FONT></B> cov<B><FONT COLOR="#663300">,</FONT></B> lik<B><FONT COLOR="#663300">,</FONT></B> x<B><FONT COLOR="#663300">,</FONT></B> y<B><FONT COLOR="#663300">) 

%</FONT></B><FONT COLOR="#FF0000">for</FONT> KL method with L<B><FONT COLOR="#663300">-</FONT></B>BFGS in log domain
apx<B><FONT COLOR="#663300"> =</FONT></B><FONT COLOR="#009900"> 'LogKLWithLBFGS'</FONT><B><FONT COLOR="#663300">;      
[</FONT></B>nlZ dnlZ<B><FONT COLOR="#663300">          ] =</FONT></B> binaryGP<B><FONT COLOR="#663300">(</FONT></B>hyp<B><FONT COLOR="#663300">.</FONT></B>cov<B><FONT COLOR="#663300">, [</FONT></B><FONT COLOR="#009900">'approx'</FONT><B><FONT COLOR="#663300">,</FONT></B>apx<B><FONT COLOR="#663300">],</FONT></B> cov<B><FONT COLOR="#663300">,</FONT></B> lik<B><FONT COLOR="#663300">,</FONT></B> x<B><FONT COLOR="#663300">,</FONT></B> y<B><FONT COLOR="#663300">) 

%</FONT></B><FONT COLOR="#FF0000">for</FONT> KL method<FONT COLOR="#990000"> using</FONT> piecewise bound with L<B><FONT COLOR="#663300">-</FONT></B>BFGS
apx<B><FONT COLOR="#663300"> =</FONT></B><FONT COLOR="#009900"> 'PiecesVB'</FONT><B><FONT COLOR="#663300">;      
[</FONT></B>nlZ dnlZ<B><FONT COLOR="#663300">          ] =</FONT></B> binaryGP<B><FONT COLOR="#663300">(</FONT></B>hyp<B><FONT COLOR="#663300">.</FONT></B>cov<B><FONT COLOR="#663300">, [</FONT></B><FONT COLOR="#009900">'approx'</FONT><B><FONT COLOR="#663300">,</FONT></B>apx<B><FONT COLOR="#663300">],</FONT></B> cov<B><FONT COLOR="#663300">,</FONT></B> lik<B><FONT COLOR="#663300">,</FONT></B> x<B><FONT COLOR="#663300">,</FONT></B> y<B><FONT COLOR="#663300">) 

%</FONT></B><FONT COLOR="#FF0000">for</FONT> KL method<FONT COLOR="#990000"> using</FONT> piecewise bound with L<B><FONT COLOR="#663300">-</FONT></B>BFGS in log domain
apx<B><FONT COLOR="#663300"> =</FONT></B><FONT COLOR="#009900"> 'LogPiecesVB'</FONT><B><FONT COLOR="#663300">;      
[</FONT></B>nlZ dnlZ<B><FONT COLOR="#663300">          ] =</FONT></B> binaryGP<B><FONT COLOR="#663300">(</FONT></B>hyp<B><FONT COLOR="#663300">.</FONT></B>cov<B><FONT COLOR="#663300">, [</FONT></B><FONT COLOR="#009900">'approx'</FONT><B><FONT COLOR="#663300">,</FONT></B>apx<B><FONT COLOR="#663300">],</FONT></B> cov<B><FONT COLOR="#663300">,</FONT></B> lik<B><FONT COLOR="#663300">,</FONT></B> x<B><FONT COLOR="#663300">,</FONT></B> y<B><FONT COLOR="#663300">)</FONT></B> </PRE>


Output for the Laplace method from GPML 2.0 is

nlZ =

    3.7607


dnlZ =

    0.6037
    0.1888

alpha =

   0.4419
   -0.2456
   0.4090
   -0.3608
   -0.2696

W =

    0.2466
    0.1853
    0.2417
    0.2306
    0.1969

Output for the original KL method from GPML 2.0 is

nlZ =

    3.6869

dnlZ =

    0.6462
    0.0924

alpha =

   0.4610
   -0.2721
   0.4321
   -0.3693
   -0.2929

W =

    0.1903
    0.1607
    0.1862
    0.1922
    0.1675

Output for the KL method with L-BFGS from GPML 2.0 is

nlZ =

    3.6869

dnlZ =

    0.6462
    0.0924

alpha =

   0.4610
   -0.2721
   0.4321
   -0.3693
   -0.2929

W =

    0.1903
    0.1607
    0.1862
    0.1922
    0.1675

Output for the KL method with L-BFGS from GPML 2.0 in log domain is

nlZ =

    3.6869

dnlZ =

    0.6462
    0.0924

alpha =

   0.4610
   -0.2721
   0.4321
   -0.3693
   -0.2929

W =

    0.1903
    0.1607
    0.1862
    0.1922
    0.1675

Output for the KL method using piecewise bound with L-BFGS from GPML 2.0 is

nlZ =

    3.6873

dnlZ =

    0.6461
    0.0925

alpha =

   0.4610
   -0.2721
   0.4321
   -0.3693
   -0.2929

W =

    0.1903
    0.1607
    0.1862
    0.1922
    0.1675

Output for the KL method using piecewise bound with L-BFGS from GPML 2.0 in log-domain is

nlZ =

    3.6873

dnlZ =

    0.6461
    0.0925

alpha =

   0.4610
   -0.2721
   0.4321
   -0.3693
   -0.2929

W =

    0.1903
    0.1607
    0.1862
    0.1922
    0.1675

The corresponding code of Laplace method for GPML 3.4 is (Note that you need to install the GPML 3.4 package first)
<PRE>clear all<B><FONT COLOR="#663300">;</FONT></B> close all<B><FONT COLOR="#663300">;</FONT></B>

meanfunc<B><FONT COLOR="#663300"> =</FONT></B> @meanZero<B><FONT COLOR="#663300">;</FONT></B> hyp<B><FONT COLOR="#663300">.</FONT></B>mean<B><FONT COLOR="#663300">=[];</FONT></B>
covfunc<B><FONT COLOR="#663300"> =</FONT></B> @covSEiso<B><FONT COLOR="#663300">;</FONT></B> hyp<B><FONT COLOR="#663300">.</FONT></B>cov<B><FONT COLOR="#663300"> =</FONT></B> log<B><FONT COLOR="#663300">([</FONT></B><FONT COLOR="#996600">2.0 2.0</FONT><B><FONT COLOR="#663300">]);</FONT></B>
likfunc<B><FONT COLOR="#663300"> =</FONT></B> @likLogistic<B><FONT COLOR="#663300">;</FONT></B> hyp<B><FONT COLOR="#663300">.</FONT></B>lik<B><FONT COLOR="#663300">=[];</FONT></B>

x1<B><FONT COLOR="#663300">=[</FONT></B><FONT COLOR="#996600">0.8822936</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#996600">0.7160792
0.9178174</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#996600">0.0135544</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#996600">0.5275911</FONT><B><FONT COLOR="#663300">];</FONT></B>

x2<B><FONT COLOR="#663300">=[-</FONT></B><FONT COLOR="#996600">0.9597321
0.0231289
0.8284935
0.0023812</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#996600">0.7218931</FONT><B><FONT COLOR="#663300">];</FONT></B>

x<B><FONT COLOR="#663300">=[</FONT></B>x1 x2<B><FONT COLOR="#663300">];</FONT></B>

y<B><FONT COLOR="#663300">=[</FONT></B><FONT COLOR="#999900">1</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#999900">1
1</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#999900">1</FONT><B><FONT COLOR="#663300">
-</FONT></B><FONT COLOR="#999900">1</FONT><B><FONT COLOR="#663300">];</FONT></B>

inf<B><FONT COLOR="#663300"> =</FONT></B> @infLaplace<B><FONT COLOR="#663300">;</FONT></B>
covfunc<B><FONT COLOR="#663300"> = {</FONT></B>covfunc<B><FONT COLOR="#663300">};</FONT></B>
meanfunc<B><FONT COLOR="#663300"> = {</FONT></B>meanfunc<B><FONT COLOR="#663300">};</FONT></B>
likfunc<B><FONT COLOR="#663300"> = {</FONT></B>likfunc<B><FONT COLOR="#663300">};

[</FONT></B>nlZ dnlZ<B><FONT COLOR="#663300">          ] =</FONT></B> gp<B><FONT COLOR="#663300">(</FONT></B>hyp<B><FONT COLOR="#663300">,</FONT></B> @infLaplace<B><FONT COLOR="#663300">,</FONT></B> meanfunc<B><FONT COLOR="#663300">,</FONT></B> covfunc<B><FONT COLOR="#663300">,</FONT></B> likfunc<B><FONT COLOR="#663300">,</FONT></B> x<B><FONT COLOR="#663300">,</FONT></B> y<B><FONT COLOR="#663300">)</FONT></B></PRE>

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

The implementation of the KL method is based on the following code from http://hannes.nickisch.org/ <br>
<a href=http://hannes.nickisch.org/code/approxXX.tar.gz>Matlab code</a>,
which is used the <a href=http://www.gaussianprocess.org/gpml/code/matlab/release/gpml-matlab-v2.0-2007-06-25.tar.gz>GPML 2.0 package</a> 
