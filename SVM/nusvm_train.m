function svm = nusvm_train(data, kernel, param, nu)

y = data.y;
X = data.X;
n = length(y);

% evaluate the kernel matrix
K = feval(kernel,X,X,param); % n x n positive semi-definite matrix
K = (K+K')/2; % should be symmetric. if not, may replace by equiv symm kernel.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For Part 4 of the problem, you must fill in the following section.
% Make sure you undestand the parameters to 'quadprog' (doc quadprog)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = 
f = 
A = 
b = 
Aeq =
beq =
LB = 
UB = 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
dpos = find((y > 0) & (alpha > 0) & (alpha < 1/n));
dneg = find((y < 0) & (alpha > 0) & (alpha < 1/n));
    
margvecs = [dpos ; dneg];
npos = length(dpos);
nneg = length(dneg);

Mpos = reshape(repmat(reshape(K(S,dpos), [NS npos 1]), [1 1 nneg]), [NS npos * nneg]);
Mneg = reshape(repmat(reshape(K(S,dneg), [NS 1 nneg]), [1 npos 1]), [NS npos * nneg]);
rho = mean(0.5*beta'*(Mpos - Mneg));

w0 = median(rho*y(margvecs) - sum(diag(beta)*K(S,margvecs))');

% store the results
svm.kernel = kernel;
svm.NS = NS;
svm.w0 = w0;
svm.beta = beta;
svm.XS = XS;
svm.rho = rho;
svm.param = param;