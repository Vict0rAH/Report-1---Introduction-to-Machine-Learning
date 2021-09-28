%% Project assignment 1
clear all
clc
%% Initial Commands
% The following code reads in the abalone dataset.
cdir = fileparts(mfilename('fullpath'));
% Since comma is used in the data, we need to convert these to dots
Data = fileread(fullfile(cdir,'../project/Data_abalone.csv'));
Data = strrep(Data, ',', '.');
FID = fopen('Data_abalone_dot.csv', 'w');
fwrite(FID, Data, 'char');
fclose(FID);
file_path = fullfile(cdir,'../project/Data_abalone_dot.csv');
abalone_table = readtable(file_path);
X = table2array(abalone_table(:, 2:9));
% Table is now stored in Matlab
[NUMERIC, TXT, RAW] = xlsread(fullfile(cdir,'../project/Data_abalone.csv'));
abalone_table = readtable(file_path);
% Extract attribute names from the first column
attributeNames = RAW(1,1:end);
% Extract unique class names from the first row
classLabels = RAW(2:end,1);
classNames = unique(classLabels);
% Extract class labels that match the class names
[y_,y] = ismember(classLabels, classNames); y = y-1;

%% Summary statistics for numerical attributes
Sr = {'Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'};
sexNames = {'Male', 'Female', 'Infant'};
% summary stadistics
% the median will be studied by the percentiles/quantile. The median is the
% quantile .5p. 
stats_names = {'Mean' 'Quantiles' 'Variance' 'Standar deviation' 'range'};

% iniziate vectors
m = zeros(1, 8);
q = zeros(8, 5);
v = zeros(1, 8);
s = zeros(1, 8);
r = zeros(1, 8);

for i = 1:8
    m(1, i) = mean(X(1:end,i));
    q(i, :) = quantile(X(1:end,i),[0 0.25 0.5 0.75 1]);
    v(1, i) = var(X(1:end,i));
    s(1, i) = std(X(1:end,i));
    r(1, i) = range(X(1:end, i));
end

% Create a table with the stadistics summary
Stats = table(m', q, v', s', r', 'VariableNames', stats_names, 'RowNames', Sr');

% study the covariance and correlation between atributes
cov_matrix = cell(8, 8);
correlation = zeros(8, 8);
for i = 1:8
    for j=1:8
        % Calculate the covariance matrix of every attribute 
        covariance = cov(X(1:end, i), X(1:end, j));
        cov_matrix{i, j} = covariance;
        % Calculate the correlation of every attribute
        correlation(i, j) = corr(X(1:end, i), X(1:end, j));
    end
end

%% Deviation of the attributes
figure(1)
mfig('Abalone: Attribute standard deviations'); clf; hold all; 
bar(1:size(Stats), s);
xticks(1:size(Stats))
xticklabels({'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings'});
ylabel('Standard deviation')
xlabel('Attributes')
title('Abalone: attribute standard deviations')

%% Visualization

% Subtract the mean from the data
Y = bsxfun(@minus, X, mean(X));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y);

% Compute variance explained
rho = diag(S).^2./sum(diag(S).^2);
threshold = 0.95;

% Plot variance explained
figure(2)
mfig('Abalone: Var. explained'); clf;
hold on
plot(rho, 'x-');
plot(cumsum(rho), 'o-');
plot([0,length(rho)], [threshold, threshold], 'k--');
legend({'Individual','Cumulative','Threshold'}, ...
        'Location','best');
ylim([0, 1]);
xlim([1, length(rho)]);
grid minor
xlabel('Principal component');
ylabel('Variance explained value');
title('Variance explained by principal components');
hold off

% boxplots of the dimensions and different weights
figure(3)
boxplot(X(1:end,1:3),{'Length','Diameter','Height'})
xlabel('Attribute')
ylabel('Size (mm)')
title('Box plots of shize attributes')
figure(4)
boxplot(X(1:end,4:7),{'Whole weight','Shucked weight','Viscera weight','Shell weight'})
xlabel('Attribute')
ylabel('Weight (g)')
title('Box plots of weight attributes')

%% Investigate how standardization affects PCA
% Subtract the mean from the data
Y1 = bsxfun(@minus, X, mean(X));

% Subtract the mean from the data and divide by the attribute standard
% deviation to obtain a standardized dataset:
Y2 = bsxfun(@minus, X, mean(X));
Y2 = bsxfun(@times, Y2, 1./std(X));
% The formula in the exercise description corresponds to:
%Y2 = (X - ones(size(X,1),1)*mean(X) ) * diag(1./std(X))
% But using bsxfun is a bit cleaner and works better for large X.

% Store the two in a cell, so we can just loop over them:
Ys = {Y1, Y2};
titles = {'Zero-mean', 'Zero-mean and unit variance'};

% Choose two PCs to plot (the projection)
i = 1;
j = 2;

% Make the plot
mfig('Abalone: Effect of standardization'); clf; hold all; 
nrows=3; ncols=2;
for k = 0:1
    % Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    [U, S, V] = svd(Ys{k+1},'econ');
    
    % For visualization purposes, we flip the directionality of the
    % principal directions such that the directions match for Y1 and Y2.
    if k==1; V = -V; U = -U; end;
    
    % Compute variance explained
    rho = diag(S).^2./sum(diag(S).^2);
    
    % Compute the projection onto the principal components
    Z = U*S;
    
    % Plot projection
    subplot(nrows, ncols, 1+k)
        C = length(classNames);
        hold on
        colors = get(gca,'colororder');
        for c = 0:C-1
            scatter(Z(y==c,i), Z(y==c,j), 50, 'o', ...
                    'MarkerFaceColor', colors(c+1,:), ...
                    'MarkerEdgeAlpha', 0, ...
                    'MarkerFaceAlpha', .5);
        end
        xlabel(sprintf('PC %d', i)); 
        ylabel(sprintf('PC %d', j));
        axis equal
        title(sprintf( [titles{k+1}, '\n', 'Projection'] ) )
        % Add a legend to one of the plots (but not both):
        if k; %h = legend(classNames, 'Location', 'best', 'color', 'none'); 
        end;

    % Plot attribute coefficients in principal component space
    subplot(nrows, ncols,  3+k);
        z = zeros(1,size(V,2))';
        quiver(z,z,V(:,i), V(:,j), 1, ...
               'Color', 'k', ...
                'AutoScale','off', ...
               'LineWidth', .1)
        hold on
        for pc=1:length(attributeNames)-1
            text(V(pc,i), V(pc,j),attributeNames{pc}, ...
                 'FontSize', 10)
        end
        xlabel('PC1')
        ylabel('PC2')
        grid; box off; axis equal;
        % Add a unit circle
        plot(cos(0:0.01:2*pi),sin(0:0.01:2*pi));
        title(sprintf( [titles{k+1}, '\n', 'Attribute coefficients'] ) )
        axis tight
        
    % Plot cumulative variance explained
    subplot(nrows, ncols,  5+k);
        plot(cumsum(rho), 'x-');
        ylim([.6, 1]); 
        xlim([1, size(V,2)]);
        grid minor
        xlabel('Principal component');
        ylabel('Cumulative variance explained')
        title(sprintf( [titles{k+1}, '\n', 'Variance explained'] ) )
        grid
        box off
end

%%
% Histogram plots
for i=1:8
   figure(i+4)
   H(i) = histogram(X(:, i));
   xlabel(Sr(i));
   title(['Histogram of the attribute ' Sr(i)])
end

% Plots distinguising the sex (M) or (F) or (I)
% Data attributes to be plotted
i = 1;      % Length
j = 4;      % Whole weight
% Make another more fancy plot that includes legend, class labels, 
% attribute names, and a title
mfig('Abalone: Classes'); clf; hold all; 
C = length(classNames);
% Use a specific color for each class (easy to reuse across plots!):
colors = get(gca, 'colororder'); 
% Here we the standard colours from MATLAB, but you could define you own.
for c = 0:C-1
    h = scatter(X(y==c,i), X(y==c,j), 50, 'o', ...
                'MarkerFaceColor', colors(c+1,:), ...
                'MarkerEdgeAlpha', 0, ...
                'MarkerFaceAlpha', .5);
end
% You can also avoid the loop by using e.g.: (but in this case, do not call legend(classNames) as it will overwrite the legend with wrong entries) 
% gscatter(X(:,i), X(:,j), classLabels)
legend(classNames);
axis tight
xlabel(attributeNames{i});
ylabel(attributeNames{j});
title('Abalone data');

sex = table2array(abalone_table(:, 1));
figure(13)
for i=1:4177
   if(strcmp(sex{i, 1}, 'M'))
       plot(X(i, 1), X(i,4), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b')
       hold on
   elseif (strcmp(sex{i, 1}, 'F'))
       plot(X(i, 1), X(i,4), 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r')
       hold on
   else 
       plot(X(i, 1), X(i,4), 'o', 'MarkerEdgeColor', 'y', 'MarkerFaceColor', 'y')
       hold on
   end
end
legend('Male', 'Female', 'Infant')
xlabel('Length (mm)')
ylabel('Whole weight')
title('Length vs Whole wieght taking into account the sex')
hold off

%% 
% What to do with the nominal?
mode(categorical(table2cell(abalone_table(:,1))))
[GC,GR]=groupcounts(categorical(table2cell(abalone_table(:,1))))