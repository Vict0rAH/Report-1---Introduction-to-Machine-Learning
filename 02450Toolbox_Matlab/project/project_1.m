%% Project assignment 1
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
X = table2array(abalone_table(:, 2:8));

% summary statistics for numerical attributes
S = {'Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'};
for i = 1:8
    S(i)
    mean(X(1:end,i))
    quantile(X(1:end,i),[0 0.25 0.5 0.75 1])
    var(X(1:end,i))
    std(X(1:end,i))
end

% What to do with the nominal?
mode(categorical(table2cell(abalone_table(:,1))))
% Sample text
