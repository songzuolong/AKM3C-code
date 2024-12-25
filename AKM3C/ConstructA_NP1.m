function [Z, Alpha, Dis, id,distX_updated]= ConstructA_NP1(A, B, k,Wv)
% d*n
% Z is sparse

n = size(A{1},2);  
m= size(B{1},2);   % the number of prototypes
v = length(A);
distXt=zeros(n,m);

for i=1:v
    distX_updated{i} = sqdist(A{i},B{i}); 
    %The Euclidean distance between data points and each prototype
    distX_updated{i}=distX_updated{i}*Wv(i);
    distXt=distXt+distX_updated{i};
end 
Dis=distXt;

di = zeros(n,k+1); 
id = di;  
for i = 1:k+1
    [di(:,i),id(:,i)] = min(distXt, [], 2);  
    temp = (id(:,i)-1)*n+[1:n]';
    distXt(temp) = 1e100;
end

clear distXt temp
id(:,end) = [];
 
Alpha = 0.5*(k*di(:,k+1)-sum(di(:,1:k),2)); 
ver=version;
if(str2double(ver(1:3))>=9.1)
    tmp = (di(:,k+1)-di(:,1:k))./(2*Alpha+eps); % for the newest version(>=9.1) of MATLAB 
else
    tmp =  bsxfun(@rdivide,bsxfun(@minus,di(:,k+1),di(:,1:k)),2*Alpha+eps); % for old version(<9.1) of MATLAB
end
Z = sparse(repmat([1:n],1,k),id(:),tmp(:),n,m);  
return
