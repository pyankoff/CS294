function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

pred = zeros(1, columns(data));


%3x5   3x4         4x5
z2 = stack{1}.w * data + repmat(stack{1}.b, 1, columns(data));
a2 = sigmoid(z2);

%5x5  5x3        3x5
z3 = stack{2}.w * a2 + repmat(stack{2}.b, 1, columns(a2));
a3 = sigmoid(z3);

%2x5     2x5        5x5
z4 = softmaxTheta * a3;
h = exp(z4);
h  = h./repmat(sum(h, 1), rows(h), 1);

for i=1:size(data, 2)
    pred(i) = find(h(:, i)==max(h(:, i)));
end









% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
