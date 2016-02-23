function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
m=length(y);
HX=transpose(theta)*transpose(X);
HXI=(transpose(HX).-y).^2;
J=sum(sum(HXI))/(2*m)

end
