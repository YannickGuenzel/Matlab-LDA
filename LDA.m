function [score, posteriors, Mdl] = LDA(data, labels, varargin)
%% [score, posteriors, Mdl] = LinearDiscriminantAnalysis(data, labels, varargin)
% Run LDA with automatically adjusted weights for each label, and retrieve
% both the LDA component scores and class posterior probabilities.
%
% INPUT:
% - data   : Input data for which to compute the LDA components, specified
%            as an n-by-p matrix. Rows of X correspond to observations and
%            columns to variables.
% - labels : Class labels, specified as a numeric or categorical vector.
%            Each row of labels corresponds to the classification of the
%            corresponding row of data.
%
% NAME-VALUE PAIRS:
% - 'DiscrimType' : 'linear' (default), 'diaglinear', 'pseudolinear',
%                   'quadratic', 'diagquadratic', or 'pseudoquadratic'
%
% OUTPUTS:
% - score      : LDA component scores, returned as an n-by-(#components)
%                matrix. Rows of score correspond to observations, and
%                columns to components. Only possible if 'DiscrimType' is
%                linear or pseudolinear.
% - posteriors : Posterior probabilities, returned as an n-by-k matrix,
%                where k is the number of classes. posteriors(i,j) is the
%                probability that observation i belongs to class j.
% - Mdl        : The trained ClassificationDiscriminant model object
%
% Version: 25-Jan-2024, R2023a

% Parse name-value input for DiscrimType
p = inputParser;
addParameter(p, 'DiscrimType', 'linear', ...
    @(x) any(strcmpi(x, ...
    {'linear','diaglinear','pseudolinear','quadratic','diagquadratic','pseudoquadratic'})));
parse(p, varargin{:});

% Convert labels to numeric labels
[~, ~, labelsNumeric] = unique(labels(:));
% Get the list of unique labels
label_list = unique(labelsNumeric);

% Compute weights for each label (inverse frequency)
label_w = zeros(size(labelsNumeric));
for iL = 1:numel(label_list)
    these_idx = (labelsNumeric == label_list(iL));
    label_w(these_idx) = 1 / sum(these_idx);
end
label_w = label_w / sum(label_w);  % normalize weights to sum to 1

% Fit the discriminant model
Mdl = fitcdiscr(data, labelsNumeric, ...
    'Weights',      label_w, ...
    'DiscrimType',  p.Results.DiscrimType);

% Project data into the LDA space
if strcmp(p.Results.DiscrimType, 'linear') || strcmp(p.Results.DiscrimType, 'pseudolinear')
    % W are the eigenvectors (discriminant directions),
    % lambda are the corresponding eigenvalues in LAMBDA
    [W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma);
    lambda = diag(LAMBDA);
    % Sort directions by descending eigenvalues
    [~, SortOrder] = sort(lambda, 'descend');
    W = W(:, SortOrder);
    % Compute LDA scores
    score = data * W;
else
    score = [];
end

% Get posterior probabilities for each class using predict
% predict returns [predictedLabels, posteriors, cost].
[~, posteriors] = predict(Mdl, data);