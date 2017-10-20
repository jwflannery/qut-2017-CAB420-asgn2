function f = svm_discrim_func(Xnew, svm)

f = (sum(diag(svm.beta)*feval(svm.kernel,svm.XS,Xnew,svm.param)) + svm.w0)';
