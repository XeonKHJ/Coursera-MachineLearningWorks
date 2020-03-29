function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


tX = [ones(size(X, 1), 1) X];
h1 = tX * Theta1';
a2 = sigmoid(h1);

a2 = [ones(size(a2,1),1) a2];
h2 = a2 * Theta2';
a3 = sigmoid(h2);

Y = zeros(size(y), num_labels);

for i = 1 : size(y, 1)
  Y(i, y(i)) = 1;
endfor

regTheta1 = Theta1;
regTheta2 = Theta2;
regTheta1(:, 1) = 0;
regTheta2(:, 1) = 0;

J = (-1/m) * sum(sum(((Y .* log(a3) + (1-Y) .* log(1-a3))))) + lambda/(2*m) * (sum((regTheta1.^2)(:)) + sum((regTheta2.^2)(:)));

delta3 = Y - a3;

a = zeros(3, size(X, 2));

for i = 1 : m
  %先向前计算a
  a1 = X(i, :);
  tempa1 = [ones(size(a1,1),1) a1];
  a2 = sigmoid(tempa1 * Theta1');
  
  tempa2 = [ones(size(a2,1),1),a2];
  a3 = sigmoid(tempa2 * Theta2');
  
  %向后传播计算导数
  delta3 = (a3 - Y(i, :))';
  delta2 = Theta2' * delta3 .* (tempa2.*(1-tempa2))';
  %delta1 = Theta1' * (delta2(2:size(delta2, 1))) .* (tempa1.*(1-tempa1))';
  
  Theta2_grad = Theta2_grad + delta3 * tempa2;
  Theta1_grad = Theta1_grad + (delta2(2:size(delta2, 1))) * tempa1;
endfor

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
Theta2_grad = (1/m) * (Theta2_grad + lambda * Theta2);
Theta1_grad = (1/m) * (Theta1_grad + lambda * Theta1);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
