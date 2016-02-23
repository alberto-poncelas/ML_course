function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

function tm= myterm(they,thehx)
	if (thehx==0)
		thehx=10000;
	end
	tm=they*log(thehx);
end
HX2=transpose(theta)*transpose(X);
HX=sigmoid(HX2);
acc=0;
sumGrad=0;
for t=1:m
	term1=myterm(-y(t),(HX(t)));
	term2=myterm((1-y(t)),(1-HX(t)));
	acc=acc+term1-term2;
end
for iter = 1:10
	theta_new=theta;
	js=length(theta);
	HXI=(transpose(HX).-y);
	for the_j = 1:js		
		xj=X(:,the_j);
		acc2=0;
		for i=1:m
			acc2=acc2+HXI(i)*xj(i);
		end
		sumatorio=acc2;
		term=sumatorio/m;
		grad(the_j)=term;
	end
	
	theta=theta_new;
end
J=acc/m;
end



