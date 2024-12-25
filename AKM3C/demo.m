clear;
clc;
load cifar10_mtv;
% gnd=Y;
C=length(unique(gnd));
N=length(gnd);
M=round(sqrt(C*N));
M_num=[70,300,2000];

% for i=1:6
%     X{i}=X{i}';
% end
for i=1:3
    [result,laKMM, laMM, BiGraph, A,obj] = AKMC(X, C,M_num(i),gnd); 
    result_KMM{i}=result;
end
% for i=1:3
%     W(1,i)=result_KMM{i}(:,1);
%     W(2,i)=result_KMM{i}(:,2);
%     W(3,i)=result_KMM{i}(:,3);
% end
% means=mean(W,2);
% W1=W*100;
% stds=std(W1,0,2); 
% %save('nuswideobj.mat','ls');
