clear; close all; clc
%% (a)
iris=load('data/iris.txt');
X = iris(:, [1, 2]);

%scatter(X(:,1),X(:,2))

%% (b)

c5 = inf;

figure; [km5r, c5r, d5r] = kmeans(X, 5, 'random'); title('kmeans plot with k = 5 and random initialization');
figure; [km5f, c5f, d5f] = kmeans(X, 5, 'farthest'); title('kmeans plot with k = 5 and furthest initialization');
figure; [km5pp, c5pp, d5pp] = kmeans(X, 5, 'k++'); title('kmeans plot with k = 5 and k++ initialization');

figure; [km20r, c20r, d20r] = kmeans(X, 20, 'random'); title('kmeans plot with k = 20 and random initialization');
figure; [km20f, c20f, d20f] = kmeans(X, 20, 'farthest'); title('kmeans plot with k = 20 and furthest initialization');
figure; [km20pp, c20pp, d20pp] = kmeans(X, 20, 'k++'); title('kmeans plot with k = 20 and k++ initialization');



%% K-Means score




%% (c) Agglomerative - Single Linkage
aggSin5 = agglomCluster(X, 5, 'min')
aggSin20 = agglomCluster(X, 20, 'min')

figure; plotClassify2D([], X, aggSin5); title('Single linkage with k = 5');
figure; plotClassify2D([], X, aggSin20); title('Single linkage with k = 20');

%% (c) Agglomerative - Complete Linkage
aggCom5 = agglomCluster(X, 5, 'max')
aggCom20 = agglomCluster(X, 20, 'max')

figure; plotClassify2D([], X, aggCom5); title('Complete linkage with k = 5');
figure; plotClassify2D([], X, aggCom20); title('Complete linkage with k = 20');

%% (d) EM Guassian Mixture

%emGaus5 = emCluster(X, 5)
emGaus20 = emCluster(X, 20)
