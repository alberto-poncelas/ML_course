function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

function [new_a]= get_a(theta, a)
	sa=size(a,2);
	a=vertcat(ones(1,sa),a);
	new_a=theta*a;
	new_a=sigmoid(new_a);
end

a1=transpose(X);
a2=get_a(Theta1,a1);
a3=get_a(Theta2,a2);

[V p]=max(a3);

p=transpose(p);

p(p==10)=0;

end
