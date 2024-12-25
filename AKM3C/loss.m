function obj = loss(distX,Z,alpha,lambda,U,V )
    n = size(Z,1);
    m = size(Z,2);
    a1 = sum(Z,2);
    D1a = spdiags(1./sqrt(a1),0,n,n);  
    a2 = sum(Z,1);
    D2a = spdiags(1./sqrt(a2'),0,m,m);
    st = sum(sum(distX.*Z));
    at = alpha*sum(sum(Z.^2));
    Da = spdiags( [ 1./sqrt(a1) ;1./sqrt(a2')],0,n+m,n+m);
    SS = sparse(n+m,n+m); SS(1:n,n+1:end) = Z; SS(n+1:end,1:n) = Z';
    ft = lambda*trace([U; V]'*(eye(n+m)-Da*SS*Da )*[U; V]);
    obj = st+ at  + ft;
end