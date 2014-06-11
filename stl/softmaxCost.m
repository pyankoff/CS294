function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;
%cost2 = 0;
thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

h = exp(theta * data);
%size(sum(h,1))
cost = -sum(sum(groundTruth .* ...
    log(h./repmat(sum(h, 1), rows(h), 1)))) ./ ...
    numCases + 0.5 * lambda * sum(sum(theta.^2));

thetagrad = -(groundTruth - ...
    h./repmat(sum(h, 1), rows(h), 1)) * ...
    data' ./ numCases + lambda * theta;

%for i=1:numCases
%    for j=1:numClasses
%        sumterm = 0;
%        for l=1:numClasses
%            sumterm = sumterm + exp(theta(l,:) * data(:, i));
%        end
%        h = exp(theta(j,:) * data(:, i));
%        cost2 = cost2 - (labels(i)==j) * log(h/sumterm) / numCases;
%    end
%end

%for i=1:numCases
%    sumterm = 0;
%    for l=1:numClasses
%        sumterm = sumterm + exp(theta(l,:) * data(:, i));
%    end
%    p = exp(theta*data(:, i))./sumterm;
%
%    thetagrad = thetagrad - (groundTruth(:, i) - p) * data(:,i)' / numCases;
%end





% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

