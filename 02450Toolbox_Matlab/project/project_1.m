%% Project assignment 1
clear all
% The following code reads in the abalone dataset.
cdir = fileparts(mfilename('fullpath'));
% Since comma is used in the data, we need to convert these to dots
Data = fileread(fullfile(cdir,'../project/Data_abalone.csv'));
Data = strrep(Data, ',', '.');
FID = fopen('Data_abalone_dot.csv', 'w');
fwrite(FID, Data, 'char');
fclose(FID);
file_path = fullfile(cdir,'../project/Data_abalone_dot.csv');
% Table is now stored in Matlab
abalone_table = readtable(file_path);
X = table2array(abalone_table(:, 2:9));

% summary statistics for numerical attributes
Str = {'Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'};
for i = 1:8
    Str(i)
    mean(X(1:end,i))
    quantile(X(1:end,i),[0 0.25 0.5 0.75 1])
    var(X(1:end,i))
    std(X(1:end,i))
end

% Subtract the mean from the data
Y = bsxfun(@minus, X, mean(X));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y);

% Compute variance explained
rho = diag(S).^2./sum(diag(S).^2);
threshold = 0.95;

% Plot variance explained
mfig('NanoNose: Var. explained'); clf;
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

figure(2)
boxplot(X(1:end,1:3),{'Length','Diameter','Height'})
xlabel('Attribute')
ylabel('Size (mm)')
figure(3)
boxplot(X(1:end,4:7),{'Whole weight','Shucked weight','Viscera weight','Shell weight'})
xlabel('Attribute')
ylabel('Weight (g)')

% What to do with the nominal?
mode(categorical(table2cell(abalone_table(:,1))))
% Sample text
