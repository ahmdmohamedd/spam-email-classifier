data = readtable('spambase.data', 'FileType', 'text');

% Split the data into features and labels
X = data{:,1:end-1};  % Features
y = data{:,end};      % Labels (1 = Spam, 0 = Not Spam)

% Split data into training and testing sets (80% train, 20% test)
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
XTrain = X(training(cv), :);
yTrain = y(training(cv), :);
XTest = X(test(cv), :);
yTest = y(test(cv), :);

% Train Naive Bayes Model
nbModel = fitcnb(XTrain, yTrain);

% Make Predictions on Test Set
% Predict the labels and also get posterior probabilities for each class
[yPred, postProbs] = predict(nbModel, XTest);

% Evaluate the Model
confMat = confusionmat(yTest, yPred);
disp('Confusion Matrix:');
disp(confMat);

accuracy = sum(yPred == yTest) / length(yTest);
disp(['Accuracy: ', num2str(accuracy)]);

% Precision, Recall, F1-Score
precision = confMat(2,2) / (confMat(2,2) + confMat(1,2));
recall = confMat(2,2) / (confMat(2,2) + confMat(2,1));
f1Score = 2 * (precision * recall) / (precision + recall);

disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['F1-Score: ', num2str(f1Score)]);

% Visualization - Confusion Matrix Plot
figure;
confusionchart(yTest, yPred);
title('Confusion Matrix for Naive Bayes Spam Classifier');

% Visualization - ROC Curve
% Use the posterior probabilities for the positive class (spam = 1)
[Xroc, Yroc, T, AUC] = perfcurve(yTest, postProbs(:,2), 1); % Compute ROC curve
figure;
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);
grid on;
