function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%Hacemos esto para que no haya log(0)
function tm= myterm(they,thehx)
	thehx2=thehx;
	thehx2(thehx2==0)=10000;%los 0 ponemos 10000
	tm=they.*log(thehx2);
end

HX=transpose(theta)*transpose(X);

HXI=(transpose(HX).-y).^2;

J=sum(sum(HXI))/(2*m)


%Add regularization
regJTerm=(sum(theta.^2)-theta(1).^2)*lambda/(2*m);
J=J+regJTerm;

HXI=(transpose(HX).-y);
grad=transpose(X)*HXI/m;

termGreg=(lambda/m)*theta;
termGreg(1)=0;

grad=grad+termGreg;

grad = grad(:);

end
