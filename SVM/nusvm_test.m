function [] = nusvm_test(kernel,param,nu,train_data,test_data)

figure;
svm = nusvm_train(train_data,kernel,param,nu);

% verify for training data
y_est = sign(svm_discrim_func(train_data.X,svm));
errors = find(y_est ~= train_data.y);

if (errors)
    fprintf('WARNING: %d training examples were misclassified!!!\n',length(errors));
    hold on;
    plot(train_data.X(errors,1),train_data.X(errors,2),'rx');
    hold off;
end

% evaluate against test data
y_est = sign(svm_discrim_func(test_data.X,svm));
errors = find(y_est ~= test_data.y);

fprintf('TEST RESULTS: %g of test examples were misclassified.\n',...
    length(errors)/length(test_data.y));
hold on;
plot(test_data.X(errors,1),test_data.X(errors,2),'k.');

svm_plot(train_data,svm);
hold off;
