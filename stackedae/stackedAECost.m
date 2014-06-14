function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

%3x5   3x4         4x5
z2 = stack{1}.w * data + repmat(stack{1}.b, 1, columns(data));
a2 = sigmoid(z2);

%5x5  5x3        3x5
z3 = stack{2}.w * a2 + repmat(stack{2}.b, 1, columns(a2));
a3 = sigmoid(z3);

%2x5     2x5        5x5
z4 = softmaxTheta * a3;
h = exp(z4);

cost = -sum(sum(groundTruth .* ...
    log(h./repmat(sum(h, 1), rows(h), 1)))) ./ M + ...
    0.5 * lambda * sum(sum(softmaxTheta.^2));;


%2x5             2x5      2x5                                     2x5
%delta4 = - (groundTruth - h./repmat(sum(h, 1), rows(h), 1)) .* sigmgrad(z4);

%size(delta4)

%5x5          5x2             2x5     2x5       5x5         
delta4 = -(softmaxTheta' * (groundTruth - ...
         h./repmat(sum(h, 1), rows(h), 1))) .* sigmgrad(z3);

%3x5        3x5!          5x5           3x5      
delta3 = (stack{2}.w' * delta4) .* sigmgrad(z2);

%4x5           4x3!       3x5                4x5
delta2 = (stack{1}.w' * delta3) .* sigmgrad(data);



%3x4!              3x5     5x4
stackgrad{1}.w = delta3 * data' ./ M;

%5x3!              5x5     5x3
stackgrad{2}.w = delta4 * a2' ./ M;

%2x5                 2x5                    5x5
softmaxThetaGrad = -(groundTruth - ...
         h./repmat(sum(h, 1), rows(h), 1)) * a3' ./ M + ...
         lambda * softmaxTheta;



%3x1                  3x5
stackgrad{1}.b = sum(delta3, 2) ./ M;

%5x1                   5x5
stackgrad{2}.b = sum(delta4, 2) ./ M;


% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function sigmg = sigmgrad(x)
    sigmg = sigmoid(x) .* (1 - sigmoid(x));
end

