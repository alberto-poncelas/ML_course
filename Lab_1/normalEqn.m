function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

Xt=transpose(X);
term2=Xt*y;
term1=Xt*X;
theta=inverse(term1)*term2;

end
