clear; close all; clc;
load('data_ps3_2.mat')
%% Set 1 - Linear

svm_test(@Klinear, 1, 1000, set1_train, set1_test);
title('Decision boundary and test errors of SVM with Linear Kernel')
%% Set 2 - Polynomial

svm_test(@Kpoly, 2, 1000, set2_train, set2_test);
title('Decision boundary and test errors of SVM with Polynomial Kernel')

%% Set 3 - Gaussian

svm_test(@Kgaussian, 1, 1000, set3_train, set3_test)
title('Decision boundary and test errors of SVM with Gaussian Kernel')
%% Set 4 - Linear

model = svm_train(set4_train, @Klinear, 1, 1000);
y_est = sign(svm_discrim_func(set4_test.X,model));
errors = find(y_est ~= set4_test.y);

fprintf('TEST RESULTS: %g of test examples were misclassified.\n',...
    length(errors)/length(set4_test.y));
%% Set 4 - Polynomial
model = svm_train(set4_train, @Kpoly, 2, 1000);
y_est = sign(svm_discrim_func(set4_test.X,model));
errors = find(y_est ~= set4_test.y);

fprintf('TEST RESULTS: %g of test examples were misclassified.\n',...
    length(errors)/length(set4_test.y));
%% Set 4 - Gaussian
model = svm_train(set4_train, @Kgaussian, 1.5, 1000);
y_est = sign(svm_discrim_func(set4_test.X,model));
errors = find(y_est ~= set4_test.y);

fprintf('TEST RESULTS: %g of test examples were misclassified.\n',...
    length(errors)/length(set4_test.y));
