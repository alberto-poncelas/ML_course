function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	theta_new=theta;
	js=length(theta);
	for the_j = 1:js
		HX=transpose(theta)*transpose(X);
		HXI=(transpose(HX).-y);		
		xj=X(:,the_j);
		acc=0;
		for i=1:m
			acc=acc+HXI(i)*xj(i);
		end
		sumatorio=acc;	
		term=alpha*sumatorio/m;
		theta_new(the_j)=theta(the_j)-term;

	end
	theta=theta_new;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
