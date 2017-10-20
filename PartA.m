clear; close all; clc
%% (init)
X = load('data/faces.txt');
%img = reshape(X(2,:),[24, 24]);
%imagesc(img); axis square; colormap gray;

%% (a)
[m, n] = size(X);
zero_mean = (X - mean(X));
[U, S, V] = svds(zero_mean, 50); %W = dot(U, S);
W = U * S;


%fprintf('Program paused. Press enter to continue.\n');
%pause;
%% (b)
for k = 1:10
    approx = (W(:, 1:k) * V(:, 1:k)');
    mse(k) = mean(mean((approx - X).^2));
end

figure;
plot(mse);
title('Plot of Mean Squared Error against K-values')
xlabel('K-value')
ylabel('Mean Squared Error')


fprintf('Program paused. Press enter to continue.\n');
%pause;
%% (c)
j = 50
scale = 2*median(abs(W(:, j)))

pcpos = mean(X) + scale * V(:, j)';
pcneg = mean(X) - scale * V(:, j)';

pos = reshape(pcpos, [24, 24]);
neg = reshape(pcneg, [24, 24]);

figure;
imagesc(pos); axis square; colormap gray;
figure;
imagesc(neg); axis square; colormap gray;

fprintf('Program paused. Press enter to continue.\n');
%pause;
%% (d)
idx = randi([1,50], 1, 25);
figure; hold on; axis ij; colormap gray;
range = max(W(idx, 1:2)) - min(W(idx, 1:2));
scale = [200, 200]./range;
for i=idx, imagesc(W(i, 1)*scale(1), W(i, 2)*scale(2), reshape(X(i, :), 24, 24)); end;

fprintf('Program paused. Press enter to continue.\n');
%pause;
%% (e)
F1idx = randi([1, 50]);
F2idx = randi([1, 50]);
idx = [5, 10, 50];

for k=idx    
    
    scale = 2*median(abs(W(:, k)));
    
    pcF1 = mean(X) + W(F1idx, 1:k) * V(:, 1:k)';
    F1 = reshape(pcF1, [24, 24]);
    
    %pcF2 = mean(X) + W(F2idx, 1:k) * V(:, 1:k)';
    %F2 = reshape(pcF2, [24, 24]);
    
    figure; imagesc(F1); axis square; colormap gray;
    title(['Reconstruction of Face 1 using ' num2str(k) '  principal directions'])
    
    %figure; imagesc(F2); axis square; colormap gray;
    %title(['Reconstruction of Face 2 using ' num2str(k) '  principal directions'])
end;