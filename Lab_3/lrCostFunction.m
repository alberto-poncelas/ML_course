function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

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
