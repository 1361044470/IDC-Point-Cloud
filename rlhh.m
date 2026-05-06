function [k_M, d_M] = rlhh(dsort)
    % =========================================================================
    % Mutation Point Detection Algorithm
    % Automatically locates the boundary between feature points and outliers.
    % =========================================================================

    % Step 1: Shift the curve to start from the origin
    d1 = dsort(1);
    dsort = dsort - d1; 
    N = length(dsort);
    
    % Step 2: Vectorized calculation to find the optimal k_o
    % This vectorized approach significantly reduces computation time.
    dsort_sq = dsort(:).^2;          
    cum_sq = cumsum(dsort_sq);       
    
    valid_k = ceil(N/2)+1 : N;     
    M = N - ceil(N/2);
    k_primes = (1:M)'; 
    
    % Compute objective function values
    obj_values_vec = dsort_sq(valid_k) - cum_sq(1:M) ./ k_primes;
    
    % Find the index of the minimum value to determine k_o
    [~, min_idx] = min(obj_values_vec);
    k_o = valid_k(min_idx);

    % Step 3: Precompute constraints
    d_k_o = dsort(k_o); 
    d_N = dsort(N);
    Constraint2 = (d_k_o + d_N) / 2.0;

    % Initialize mutation point variables
    k_M = ceil(0.01 * N); 
    minL = Inf;     
    
    % Scale factor 'alpha' determines the starting search index to prevent division by zero, defaulting to 0.01 but empirically lowered (e.g., 0.00001) for real scanned datasets.
    alpha = 0.01;       
    start_idx = max(1, ceil(alpha * N));
    
    % Step 4: Iterative search for the optimal mutation point
    for k = start_idx : N-1   % Loop until N-1 to prevent division by zero
        
        % Calculate relative slopes
        temp1 = dsort(k) / k;
        temp2 = (dsort(N) - dsort(k)) / (N - k);
        
        % Compute the evaluation metric L
        L = temp1 / temp2;     
        d_current = dsort(k);
        
        % Constraint 1 (Empirical coefficient set to 1.7 for optimal performance)
        Constraint1 = 1.7 * k * d_k_o / k_o; 

        % Update mutation point if L is minimized and constraints are satisfied
        if (L <= minL) && (d_current <= min(Constraint1, Constraint2))
            minL = L;
            k_M = k;
        end
    end
    
    % Step 5: Restore the original curve and output values
    dsort = dsort + d1; 
    d_M = dsort(k_M);

end