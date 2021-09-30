%% Project assignment 1
clear all
close all
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
X = table2array(abalone_table(1:30:end, 2:9)); % Extracted only some data
% Table is now stored in Matlab
[NUMERIC, TXT, RAW] = xlsread(fullfile(cdir,'../project/Data_abalone.csv'));
abalone_table = readtable(file_path);
% Extract attribute names from the first column
attributeNames = RAW(1,1:end);
% Extract unique class names from the first row
classLabels = RAW(2:30:end,1);
classNames = unique(classLabels);
% Extract class labels that match the class names
[y_,y] = ismember(classLabels, classNames); y = y-1;

%% SUMMARY STATISTICS FOR NUMERICAL ATTRIBUTES
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

%% DEVIATION OF THE ATTRIBUTES
figure(1)
mfig('Abalone: Attribute standard deviations'); clf; hold all; 
bar(1:size(Stats), s);
xticks(1:size(Stats))
xticklabels({'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings'});
ylabel('Standard deviation')
xlabel('Attributes')
title('Abalone: attribute standard deviations')

%% BOXPLOTS
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

%% REMOVING THE OUTLIERS

X = rmoutliers(X);

%% NORMAL DISTRIBUTION ANALYSIS 

% Calculate the statistics without outliers

for i = 1:8
    m(1, i) = mean(X(1:end,i));
    q(i, :) = quantile(X(1:end,i),[0 0.25 0.5 0.75 1]);
    v(1, i) = var(X(1:end,i));
    s(1, i) = std(X(1:end,i));
    r(1, i) = range(X(1:end, i));
end
Samples = 113;
% Number of beans
NBins = 20;

% Plot a histogram
for i=1:8
    figure(5)
    subplot(2, 4, i)
    H(i) = histogram(X(:, i));
    hold on
    [n, BinEdges] = histcounts(X(:, i));
    x = linspace(min(X(:, i)), max(X(:, i)), 1000);
    plot(x, normpdf(x, m(i), s(i)), 'r', 'LineWidth', 5);
    xlabel(Sr(i));
    title(Sr(i))
    xlim([min(x), max(x)]);
    hold off
end

%% STUDY THE COVARIANCE AND THE CORRELATION 

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
for i=1:length(X(:,1))
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

%%
%PCA section
% X=X(1:20:4177,:); %divide dataset
%  X = rmoutliers(X); %remove outliers
rings=X(1:end,8);
X1=X';

mean_rows=mean(X1,2); % mean value for every attribute
number_samples=length(X1); % number of samples in our dataset
mean_matrix=mean_rows*ones(1,number_samples); % expanding mean values into matrix for later substarction
B=X1-mean_matrix;  % new B matrix (X-mean)
% std_deviation=[0.12 0.099 0.042 4.95 0.9 0.11 0.14];
% std_dev=std_deviation';
B=B';  % transposed matrix (because of youtube example)
Bn=B;

for i = 1:8
    s(1, i) = std(X(1:end,i));
end
for i=1:7
    Bn(:,i)=B(:,i).*(1/s(1,i)); % dividing by standard deviation (to normalize)
end

[U,S,V] = svd(Bn/sqrt(number_samples),'econ');  % Find principal components (SVD) normalized by number of samples
%max value of rings(age) is 29
figure(1)
for i=1:number_samples  % ploting first three Principal Components of dataset
    %tried to plot it in sense of different age slots.
    %plotting 1/30 od samples not to let it look to messy
    x=V(:,1)'*X(i,:)';
    y=V(:,2)'*X(i,:)';
    z=V(:,3)'*X(i,:)';
    if(rings(i)<6)
        plot3(x,y,z,'ro','LineWidth',1.5)
    elseif (rings(i)>=6 && rings(i)<10)
        plot3(x,y,z,'bo','LineWidth',1.5)
    elseif (rings(i)>=10 && rings(i)<14)
        plot3(x,y,z,'kx','LineWidth',1.5)
    else
        plot3(x,y,z,'gx','LineWidth',1.5)
    end
    hold on
end
grid minor
set(findall(gca,'-property','FontSize'),'FontSize',20);
xlabel('First principal component');
ylabel('Second principal component');
zlabel('Third principal component');
% legend({'Age < 6','6 < Age < 10','10 < Age < 14','Age > 14'}, ...
%         'Location','best');
title('PCA visualization of first three components');
hold off


figure(2)
for i=1:number_samples  % ploting first three Principal Components of dataset
    %tried to plot it in sense of different age slots.
    %plotting 1/30 od samples not to let it look to messy
    x1=V(:,1)'*X(i,:)';
    y1=V(:,2)'*X(i,:)';
%     z=V(:,3)'*X(i,:)';
    if(rings(i)<6)
        plot(x1,y1,'ro','LineWidth',1.5)
    elseif (rings(i)>=6 && rings(i)<10)
        plot(x1,y1,'bo','LineWidth',1.5)
    elseif (rings(i)>=10 && rings(i)<14)
        plot(x1,y1,'kx','LineWidth',1.5)
    else
        plot(x1,y1,'gx','LineWidth',1.5)
    end
    hold on
end
grid minor
set(findall(gca,'-property','FontSize'),'FontSize',20);
xlabel('First principal component');
ylabel('Second principal component');
% legend
%({'Age < 6','6 < Age < 10','10 < Age < 14','Age > 14'}, ...
       % 'Location','best');
% zlabel('Third principal component');
title('PCA visualization of first two components');
hold off


% Compute variance explained
rho = diag(S).^2./sum(diag(S).^2);
threshold = 0.90;

% Plot variance explained
figure(3)
clf;
hold on
plot(rho, 'x-','LineWidth',1.5);
plot(cumsum(rho), 'o-','LineWidth',1.5);
plot([0,length(rho)], [threshold, threshold], 'k--','LineWidth',1.5);
legend({'Individual','Cumulative','Threshold'}, ...
        'Location','best');
set(findall(gca,'-property','FontSize'),'FontSize',20);
ylim([0, 1]);
xlim([1, length(rho)]);
grid minor
xlabel('Principal component');
ylabel('Variance explained value');
title('Variance explained by principal components');

