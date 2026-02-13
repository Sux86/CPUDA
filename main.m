function [acc]=CPUDA(X_s,Y_s,X_t,Y_tt,P_dim,alpha,gamma)

    [d,ns]=size(X_s);
    [~,nt]=size(X_t);
    
    %Init Y_t
    knn_model = fitcknn((X_s)',Y_s,'NumNeighbors',1);
    Y_t = knn_model.predict((X_t)');
    tmp_y=double(Y_tt)-double(Y_t);
    [result,~]=find(tmp_y==0);
    acc=size(result,1)/size(Y_tt,1);
    fprintf('1NN-acc:%2f \n',acc);
    Y_s=full(ind2vec(double(Y_s)'));
    Y_t=full(ind2vec(double(Y_t)'));

    
    options = [];
    options.ReducedDim = P_dim;
    [P, ~, ~] = pca([X_s X_t]', 'NumComponents', P_dim);
    
    [q,~]=size(Y_s);
    iter=1;
    Objvalue=0;
    best_acc = 0;               % 初始化最佳 acc
    
    X=[X_s X_t];

    % init c^t;
    C_s = (P' * X_s * Y_s') ./ (sum(Y_s, 2)');  % Target class prototype
    C_t = (gamma * C_s + P' * X_t * Y_t') / (gamma * eye(q) + Y_t * Y_t');
    while iter <= 10
        % contruct M
        M=0;
        if ~isempty(Y_t)
            M=conditionalDistribution(ns,nt,Y_s,Y_t,q,0.1);
        end
        % update Y^t
        Y_t = update_y(nt, q, (P' * X_t), C_t);
        acc = get_acc(Y_t, Y_tt);  % 计算当前 acc
      
        B1=(X*M*X' + alpha*eye(size(X,1)) + X_t*X_t');
        B2=X_t*Y_t'*C_t';
        [P, ~] = projected_gradient_descent(P,B1, B2, ...
            'MaxIter', 200, ...
            'Tol', 1e-4, ...
            'Verbose', false);

        % update c^t
        C_s = (P' * X_s * Y_s') ./ (sum(Y_s, 2)');  % Target class prototype
        C_t = (gamma * C_s + P' * X_t * Y_t') / (gamma * eye(q) + Y_t * Y_t');
    end
end


