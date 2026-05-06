% =========================================================================
% Iterative Mutation-Point Detection and Completion (IDC)
% For real scanned models / indoor scenes (without ground truth labels)
% =========================================================================

% ========== Global Time Initialization ==========
totalTic = tic;  
timingLog = struct();  

%% ========== 1. Read Point Cloud File ==========
fprintf('[Stage 1] Reading real scanned point cloud data...\n');
readTic = tic;

% NOTE: Replace with your specific path (e.g., statue.pcd or indoor scene)
ptCloud = pcread('./Data/ScannedDataset/ModelData/sample20K_pset-statue-acts-unfiltered - Cloud.pcd');
points = ptCloud.Location;  

% Preserve original colors if they exist, otherwise set to white
if isempty(ptCloud.Color)
    colors = ones(size(points,1), 3, 'uint8') * 255;
    ptCloud.Color = colors;
end

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

% Adaptive scale factor for initial separation (Beta 1)
% Note: Real-world scans often exhibit different spatial densities and artifact 
% distributions than synthetic data, thus the scale factor is empirically adapted.
delta1 = round(0.045 * size(points,1));  
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
delta2 = 0.8400; % Threshold range parameter for real data     
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
    
    valid_mask = imd_scores <= imd_sorted(k_th_Left);
    middle_mask = (imd_scores > imd_sorted(k_th_Left)) & (imd_scores <= imd_sorted(k_th_Right)); 
    
    valid_F = select(current_F, valid_mask);
    
    % Merge properties safely (handling colors, normals, intensities dynamically)
    combined_points = [current_denoisedCloud_incomplete.Location; valid_F.Location];
    nvPairs = {'Color', [current_denoisedCloud_incomplete.Color; valid_F.Color]};
    
    if ~isempty(ptCloud.Normal)
        nvPairs = [nvPairs, {'Normal', [current_denoisedCloud_incomplete.Normal; valid_F.Normal]}];
    end
    
    if ~isempty(ptCloud.Intensity)
        nvPairs = [nvPairs, {'Intensity', [current_denoisedCloud_incomplete.Intensity; valid_F.Intensity]}];
    end
    
    current_denoisedCloud_incomplete = pointCloud(combined_points, nvPairs{:});
    
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

%% ========== 6. Result Visualization ==========
fprintf('[Stage 6] Rendering visualizations...\n');
vizTic = tic;

% Plot 1: Mutation Point Curve
figure('Position', [100 200 600 500], 'Color', 'white', 'Name', 'Adaptive Threshold Detection');
h_curve = plot(dsort, 'Color', [0.2 0.4 0.8], 'LineWidth', 2);
hold on;
scatter(k_M, d_M, 120, 'Marker', 'd', 'MarkerFaceColor', [0.9 0.1 0.1], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
scatter(k_M-delta1, d_Left, 120, 'Marker', 'd', 'MarkerFaceColor', [0.3 0.6 0.3], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
scatter(k_M+delta1, d_Right, 120, 'Marker', 'd', 'MarkerFaceColor', [0.1 0.4 0.7], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
xline(k_M-delta1, '--', 'Color', [0.3 0.6 0.3], 'LineWidth', 1.2);
xline(k_M, '--', 'Color', [0.9 0.1 0.1], 'LineWidth', 1.2);
xline(k_M+delta1, '--', 'Color', [0.1 0.4 0.7], 'LineWidth', 1.2);
title('Initial Threshold Selection Curve', 'FontSize', 12);
xlabel('Sorted Point Index', 'FontWeight', 'bold');
ylabel('Average Distance', 'FontWeight', 'bold');
legend('Sorted Distance', 'Mutation Point', 'Left Threshold', 'Right Threshold', 'Location', 'northwest');
grid on; ax = gca; ax.GridAlpha = 0.4; box off;

% Plot 2: Before & After Point Cloud Comparison
figure('Position', [750 200 1000 500], 'Color', 'white', 'Name', 'Denoising Result');
subplot(1,2,1); pcshow(ptCloud); title(['Original Scanned Point Cloud: ', num2str(ptCloud.Count),' points']);
subplot(1,2,2); pcshow(completed_cloud); title(['Purified Point Cloud (IDC): ', num2str(completed_cloud.Count),' points']);

timingLog.visualization = toc(vizTic);

%% ========== 7. Save Point Cloud & High-Res Figure ==========
fprintf('[Stage 7] Exporting results and high-res figures...\n');
saveTic = tic;

% Save Point Cloud Data
outputPath = './Data/ScannedDataset/ModelData/output_statue-acts.pcd';
pcwrite(completed_cloud, outputPath);

% Generate High-Resolution Publication Figure (No White Margins)
fig = figure('Position', [100, 100, 1000, 800], 'Color', 'white', 'Visible', 'off'); 
ax = axes('Parent', fig, 'Position', [0, 0, 1, 1]);  
pcshow(completed_cloud, 'Parent', ax, 'MarkerSize', 20);  
view(ax, 0, 90); % Top view
axis(ax, 'tight'); axis(ax, 'equal'); axis(ax, 'off');
set(ax, 'Color', 'white'); set(fig, 'Color', 'white');
set(fig, 'Renderer', 'opengl');  
drawnow;

outputImagePath = './Data/ScannedDataset/ModelData/statue-acts-figure.png';
exportgraphics(fig, outputImagePath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('   --- Point cloud saved to: %s\n', outputPath);
fprintf('   --- High-res figure saved to: %s\n', outputImagePath);
close(fig); % Close the hidden figure to save memory

timingLog.saving = toc(saveTic);
timingLog.total = toc(totalTic);

%% ========== 8. Execution Time Statistics ==========
fprintf('\n================ FINAL RESULTS ================\n');
fprintf('Original Points       : %d\n', ptCloud.Count);
fprintf('Completed Points      : %d\n', completed_cloud.Count);
fprintf('Removed Outliers      : %d\n', ptCloud.Count - completed_cloud.Count);
fprintf('===============================================\n');

fprintf('\n[Time Analysis]\n');
timeStages = {'1. Read Data'; '2. KNN Search'; '3. Mutation Det.'; '4. Separation'; '5. Iterative Comp.'; '6. Visualization'; '7. Data Export'; 'Total Time'};
timingData = [timingLog.read; timingLog.knn; timingLog.rlhh; timingLog.denoise; timingLog.completion; timingLog.visualization; timingLog.saving; timingLog.total];

fprintf('%-20s %-10s %s\n','Stage','Time (s)','Ratio (%)'); 
fprintf('-------------------------------------------\n');
for i = 1:(numel(timeStages)-1)
    fprintf('%-20s %-10.4f %.2f%%\n', timeStages{i}, timingData(i), timingData(i)/timingLog.total*100);
end