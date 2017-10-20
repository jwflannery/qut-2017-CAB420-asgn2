function svm = svm_train(data, kernel, param, C)

y = data.y;
X = data.X;
n = length(y);

% evaluate the kernel matrix
K = feval(kernel,X,X,param); % n x n positive semi-definite matrix
K = (K+K')/2; % should be symmetric. if not, may replace by equiv symm kernel.

% solve dual problem...
D = diag(y); % diagonal matrix with D(i,i) = y(i)
H = D*K*D; % H(i,j) = y(i)*K(i,j)*y(j)
% note, H & K are similar matrices => H is positive semi-definite.
f = -ones(n,1);
A = [];
b = [];
Aeq = y';
beq = 0.0;
LB = zeros(n,1);
UB = C * ones(n,1);
X0 = zeros(n,1);

warning off; % suppress 'Warning: Large-scale method ...'
alpha = quadprog(H+1e-10*eye(n),f,A,b,Aeq,beq,LB,UB,X0)
warning on;

% essentially, we have added a (weak) regularization term to
% the dual problem favoring minimum-norm alpha when solution
% is underdetermined. this is also important numerically
% as any round-off error in computation of H could potentially
% cause dual problem to become ill-posed (minimizer at infinity).
% regularization term forces Hessian to be positive definite.

% select support vectors.
S = find(alpha > eps);
NS = length(S);
beta = alpha(S).*y(S);
XS = X(S,:);

% estimate w0 robustly (bias parameter)
margvecs = find((alpha > 1e-3*max(alpha)) & (alpha < C - 1e-3*max(alpha)));
w0 = mean(y(margvecs) - sum(diag(beta)*K(S,margvecs))');

% store the results
svm.kernel = kernel;
svm.param = param;
svm.NS = NS;
svm.w0 = w0;
svm.beta = beta;
svm.XS = XS;
