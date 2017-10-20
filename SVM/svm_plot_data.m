
function [] = svm_plot_data(data)

neg = find(data.y < 0); 
pos = find(data.y > 0); 

h=plot(data.X(neg,1),data.X(neg,2),'rv'); hold on;
set(h,'MarkerSize',5);

h=plot(data.X(pos,1),data.X(pos,2),'b^'); hold off;
set(h,'MarkerSize',5);
