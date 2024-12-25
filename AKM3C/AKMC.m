function [result,laKMM, laMM, BiGraph, A,obj] = AKMC(X, c,m,groundtruth)
% Input:
% X:                multi-view data, each view is a data matrix ,the size is nFea x nSmp,
%                   where each column is a sample point
% c:                number of clusters
% m:                number of prototypes
% v:                number of views
% groundtruth��     groundtruth of the data,
% Output:
%       - result: ACC,NMI and Purity.
%       - laKMM: the cluster assignment for each point
%       - laMM: the sub-cluster assignment for each point
%       - BiGraph: the matrix of size nSmp x nMM
%       - A: the multiple means matrix of each view, the size is nFea x nMM
%       - obj:the value of object function
%       - result: ACC,NMI and Purity.
% Requre:
% 		meanInd.m
% 		ConstructA_NP1.m
% 		EProjSimplex_new.m
% 		svd2uv.m
% 		struG2la.m
%       eig1.m
%       sqdist.m
%       gen_nn_distanceA.m
%       ClustringMeasure.m

%----------------------------------------------------------------------------------

k=5;                               % k is the number of adaptive neighbours
v = length(X);
n=size(X{1},2);                    %n is the number of samples
BiGraph = ones(n,m);  
NITER = 30;
Num=20;
zr = 10e-5;
obj=zeros(1,Num);
Wv1=ones(1,v)/v;
Wv2=ones(1,v)/v;


%% =====================  Initialization ===============================

for i=1:v
    StartIndZ{i}=kmeans(X{i}',m);
end

for i=1:v
    A{i} = meanInd(X{i}, StartIndZ{i},m,BiGraph);  
end


%% =====================  updating ====================================
for num=1:Num
[Z, Alpha, distX, id,distX_updated1] =  ConstructA_NP1(X,A,k,Wv1); 
[ZT, AlphaT, distXT, idT,distX_updated2] =  ConstructA_NP1(A,X,k,Wv2);

alpha =1*mean(Alpha);
alphaT =1*mean(AlphaT); 

lambda = (alpha+alphaT)/2;
Z0 = (Z+ZT')/2;                                             %initialize S


[BiGraph, U, V, evc] = svd2uv(Z0, c);                       %initialize F
D1 = 1; D2 = 1;Ater = 0;
dxi = zeros(n,k);
for i = 1:n
    dxi(i,:) = distX(i,id(i,:)); 
end
dxiT = zeros(m,k);
for i = 1:m
    dxiT(i,:) = distXT(i,idT(i,:));
end  

% if sum(evc(1:c)) > c*(1-zr)
%     error('The original graph has more than %d connected component Please set k larger', c);      
% end;


%% ===================== updating  S and F=====================
for iter = 1:NITER
    U1 = D1*U;
    V1 = D2*V;
    dist = sqdist(U1',V1');  
    tmp1 = zeros(n,k); 
    for i = 1:n
        dfi = dist(i,id(i,:));
        ad = -(dxi(i,:)+lambda*dfi)/(2*alpha);   
        tmp1(i,:) = EProjSimplex_new(ad);
    end
    Z = sparse(repmat([1:n],1,k),id(:),tmp1(:),n,m);
    
    tmp2 = zeros(m,k);
    for i = 1:m
        dfiT = dist(idT(i,:),i);
        ad = (dxiT(i,:)-0.5*lambda*dfiT')/(2*alphaT); 
        tmp2(i,:) = EProjSimplex_new(ad);
    end 
    ZT = sparse(repmat([1:m],1,k),idT(:),tmp2(:),m,n);  

    BiGraph = (Z+ZT')/2;
    U_old = U;
    V_old = V;
    [BiGraph, U, V, evc, D1, D2] = svd2uv(BiGraph, c);
    
    fn1 = sum(evc(1:c));
    fn2 = sum(evc(1:c+1));
    
    if fn1 < c-zr % the number of block is less than c
        Ater=0;
        lambda = 2*lambda;
    elseif fn2 > c+1-zr % the number of block is more than c
        Ater = 0;
        lambda = lambda/2;   U = U_old; V = V_old;
    else
        Ater=Ater+1;
        if(Ater==2)
            break;
        end
    end
end

laMM=id(:,1);

[clusternum, laKMM] = struG2la(BiGraph);
result_qwe{num} = ClusteringMeasure(groundtruth, laKMM);


% Update multiple means for each view
for i=1:v
    for j=1:m
        A{i}(:,j)=X{i}(:,:)*BiGraph(:,j)/sum(BiGraph(:,j));
    end
end
% Update the weight for each view
for i=1:v 
    Wv1(i) = 0.5/sqrt(sum(sum( distX_updated1{i}.*BiGraph)));       
    Wv2(i) = 0.5/sqrt(sum(sum( distX_updated2{i}.*BiGraph'))); 
end

obj(num) = loss(distX,BiGraph,alpha,lambda,U,V );
  
fprintf('Iter:%d\n',num);

 if num>1
     te_abs{num}=abs(obj(num-1) - obj(num))/obj(num);
    if(abs(obj(num-1) - obj(num))/obj(num)<0.03)
         break;
    end
 end

end
%% ===================== Result=====================
[clusternum, laKMM] = struG2la(BiGraph);
result = ClusteringMeasure(groundtruth, laKMM);





