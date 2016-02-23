function [J, grad] = lrCostFunction2(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%Hacemos esto para que no haya log(0)
function tm= myterm(they,thehx)
	thehx2=thehx;
	thehx2(thehx2==0)=10000;%los 0 ponemos 10000
	tm=they.*log(thehx2);
end
HX2=transpose(theta)*transpose(X);
HX=sigmoid(HX2);
acc=0;
sumGrad=0;
HX=transpose(HX);
term1=myterm(-y,HX);
term2=myterm((1-y),(1-HX));
acc=sum(term1-term2);
J=acc/m;
%Add regularization
regJTerm=(sum(theta.^2)-theta(1).^2)*lambda/(2*m);
J=J+regJTerm;
HXI=(HX.-y);
grad=transpose(X)*HXI/m;
termGreg=(lambda/m)*theta;
termGreg(1)=0;
grad=grad+termGreg;
grad = grad(:);
end
