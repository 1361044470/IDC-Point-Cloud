% =========================================================================
% Iterative Mutation-Point Detection and Completion (IDC)
% For synthetic dataset evaluation (with ground truth labels)
% =========================================================================

% ========== Global Time Initialization ==========
totalTic = tic;  
timingLog = struct();  

%% ========== 1. Read Point Cloud File ==========
fprintf('[Stage 1] Reading point cloud data...\n');
readTic = tic;

ptCloud = pcread('./Data/SyntheticDataset/bunny140k_outliers30% - Cloud.pcd');
points = ptCloud.Location;

labels = load('./Data/SyntheticDataset/bunny140k_outliers30%.outliers.txt'); 
labels = logical(labels); 
assert(size(points,1) == numel(labels), 'Mismatch between points and labels!');

colors = zeros(size(points,1), 3, 'uint8');
colors(labels, 1) = 255;    
colors(~labels, 2) = 255;   
ptCloud.Color = colors;

timingLog.read = toc(readTic);

%% ========== 2. Calculate Average K-Nearest Neighbor Distance ==========
fprintf('[Stage 2] Calculating average KNN distance...\n');
knnTic = tic;
allTic = tic;

K1 = 15; 
kdTree = KDTreeSearcher(points);  
[~, dists] = knnsearch(kdTree, points, 'K', K1+1); 
validDists = dists(:, 2:end);    
avgDistances = mean(validDists, 2);  

timingLog.knn = toc(knnTic);

%% ========== 3. Mutation Point Detection (rlhh) ==========
fprintf('[Stage 3] Detecting initial mutation point...\n');
rlhhTic = tic;

validIdx = ~isnan(avgDistances);    
dsort = sort(avgDistances(validIdx), 'ascend');
[k_M, d_M] = rlhh(dsort);           

timingLog.rlhh = toc(rlhhTic);

%% ========== 4. Initial Point Cloud Separation ==========
fprintf('[Stage 4] Generating incomplete and mixed point clouds...\n');
denoiseTic = tic;

delta1 = round(0.18 * size(points,1)); 
k_Left = max(1, k_M - delta1); 
k_Right = min(length(dsort), k_M + delta1);
d_Left = dsort(k_Left);             
d_Right = dsort(k_Right); 

inliers_incomplete = avgDistances <= d_Left;
denoisedCloud_incomplete = select(ptCloud, inliers_incomplete);  

F_mask = (avgDistances <= d_Right) & (avgDistances > d_Left);    
F_cloud = select(ptCloud, F_mask);

timingLog.denoise = toc(denoiseTic);

%% ========== 5. Iterative Feature Point Completion ==========
fprintf('[Stage 5] Executing iterative feature completion...\n');
completionTic = tic;

maxIterations = 4;      
delta2 = 0.200;         
K2 = 25;                

current_denoisedCloud_incomplete = denoisedCloud_incomplete;  
current_F = F_cloud;                    

for iter = 1:maxIterations
    fprintf('   --- Iteration %d: ', iter);

    incomplete_points = current_denoisedCloud_incomplete.Location;
    kdTree_incomplete = KDTreeSearcher(incomplete_points);

    F_points = current_F.Location;
    [all_idx, ~] = knnsearch(kdTree_incomplete, F_points, 'K', K2);
    fprintf('%d points remaining in the mixed cloud ---\n', size(F_points,1));

    imd_scores = zeros(size(F_points,1),1);
    parfor i = 1:size(F_points,1)
        neighbors = incomplete_points(all_idx(i,:), :);
        cov_mat = cov(neighbors) + eye(3)*1e-6; 
        mu = mean(neighbors);
        beta = F_points(i,:) - mu;
        imd_scores(i) = sqrt(beta / cov_mat * beta');
    end

    imd_sorted = sort(imd_scores(~isinf(imd_scores)));
    [k_th_M, d_th_M] = rlhh(imd_sorted);
    
    deltaRange = round(delta2 * size(F_points,1));
    k_th_Left = max(1, k_th_M - deltaRange);
    k_th_Right = min(length(imd_sorted), k_th_M + deltaRange);

    if iter == maxIterations
        valid_mask = imd_scores <= imd_sorted(k_th_M);
    else
        valid_mask = imd_scores <= imd_sorted(k_th_Left);  
    end
    middle_mask = (imd_scores > imd_sorted(k_th_Left)) & (imd_scores <= imd_sorted(k_th_Right)); 

    valid_F = select(current_F, valid_mask);
    current_denoisedCloud_incomplete = pointCloud(...
        [current_denoisedCloud_incomplete.Location; valid_F.Location],...
        'Color', [current_denoisedCloud_incomplete.Color; valid_F.Color]);

    current_F = select(current_F, middle_mask);

    if current_F.Count == 0
        fprintf('   --- Early termination: Feature cloud is empty.\n');
        break;
    end
end

completed_cloud = current_denoisedCloud_incomplete;
timingLog.completion = toc(completionTic);
timingLog.all = toc(allTic);
fprintf('[Status] Core algorithm finished (Time: %.4fs)\n', timingLog.all);

%% ========== 6. Visualization & Export ==========
fprintf('[Stage 6] Rendering visualizations and exporting results...\n');
vizTic = tic;

% Plot 1: Mutation Point Curve (Methodology Demonstration)
figure('Position', [100 200 600 500], 'Color', 'white', 'Name', 'Adaptive Threshold Detection');
h_curve = plot(dsort, 'Color', [0.2 0.4 0.8], 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 3, 'MarkerFaceColor', [0.6 0.8 1]);
hold on;
Left_threshold_Marking = scatter(k_M-delta1, d_Left, 120, 'Marker', 'd', 'MarkerFaceColor', [0.3 0.6 0.3], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
Threshold_Marking = scatter(k_M, d_M, 120, 'Marker', 'd', 'MarkerFaceColor', [0.9 0.1 0.1], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
Right_threshold_Marking = scatter(k_M+delta1, d_Right, 120, 'Marker', 'd', 'MarkerFaceColor', [0.1 0.4 0.7], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
xline(k_M-delta1, '--', 'LineWidth', 1.2, 'Color', [0.3 0.6 0.3], 'Alpha', 0.7);
xline(k_M+delta1, '--', 'LineWidth', 1.2, 'Color', [0.6 0.3 0.3], 'Alpha', 0.7);
text(k_M + delta1*0.2, d_M*0.55, {sprintf('k_M = %d',k_M), sprintf('(%.4f)', d_M)},...    
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'cap', 'FontSize', 9,...
    'BackgroundColor', [1 1 1 0.9], 'EdgeColor', [0.7 0.7 0.7], 'Margin', 1, 'FontAngle', 'italic');               
ax = gca; ax.FontName = 'Arial'; ax.FontSize = 10; ax.XGrid = 'on'; ax.YGrid = 'on'; ax.GridAlpha = 0.4;
ax.XLabel.String = 'Sorted Point Index'; ax.YLabel.String = 'Average Distance';
ax.XLabel.FontWeight = 'bold'; ax.YLabel.FontWeight = 'bold';
legend([h_curve, Threshold_Marking, Left_threshold_Marking, Right_threshold_Marking],...
    {'Sorted Distance', 'Mutation Point', 'Left Threshold', 'Right Threshold'}, 'Location', 'northwest', 'Box', 'off', 'FontSize', 10);
title({sprintf('Initial Threshold Selection (K=%d)', K1), sprintf('[k-Left: %.4f, k-M: %.4f, k-Right: %.4f]', k_Left, k_M, k_Right)}, 'FontSize', 11, 'FontWeight', 'normal');
xlim([0 length(dsort)*1.05]); box off;

% Plot 2: Before & After Point Cloud Comparison
figure('Position', [750 200 1000 500], 'Color', 'white', 'Name', 'Denoising Result');
subplot(1,2,1); pcshow(ptCloud); title(['Original Noisy Point Cloud: ', num2str(ptCloud.Count),' points']);
subplot(1,2,2); pcshow(completed_cloud); title(['Purified Point Cloud (IDC): ', num2str(completed_cloud.Count),' points']);

timingLog.visualization = toc(vizTic);

pcwrite(completed_cloud, './Data/SyntheticDataset/output_completed_cloud.pcd');

%% ========== 7. Evaluation Metrics Calculation ==========
fprintf('[Stage 7] Calculating quantitative metrics...\n');
evalTic = tic;

calculateMetrics = @(inlier_mask) struct(...
    'ODR', sum(labels & ~inlier_mask) / sum(labels),... 
    'Recall', sum(labels & ~inlier_mask) / (sum(labels & ~inlier_mask) + sum(~labels & ~inlier_mask)),...   
    'Accuracy', (sum(labels & ~inlier_mask) + sum(~labels & inlier_mask)) / numel(labels)...                
);

metrics = struct();

metrics.raw_input = calculateMetrics(true(size(labels)));  
metrics.final_output = calculateMetrics(ismember(points, completed_cloud.Location, 'rows')); 

metricNames = {'ODR', 'Recall', 'Accuracy'};
stageNames = {'raw_input', 'final_output'};
data = zeros(numel(metricNames), numel(stageNames));
for i = 1:numel(stageNames)
    data(:,i) = struct2array(metrics.(stageNames{i}));  
end

metricTable = array2table(data', 'RowNames',stageNames, 'VariableNames',metricNames); 
writetable(metricTable, './Data/SyntheticDataset/quantitative_results.xlsx', 'WriteRowNames', true); 

% Print Final Metrics to Console
fprintf('\n================ FINAL RESULTS ================\n');
disp(metricTable);
fprintf('===============================================\n');

timingLog.evaluation = toc(evalTic);
timingLog.total = toc(totalTic);