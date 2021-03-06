function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
[J2, grad2] = lrCostFunction2(theta, X, y, lambda)
[J, grad] = costFunction(theta, X, y);
regJTerm=(sum(theta.^2)-theta(1).^2)*lambda/(2*m);
J=J+regJTerm;
regGradTerm=(lambda/m)*theta;
regGradTerm(1)=0;
grad=grad+regGradTerm;
end
