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

% Mean values for attributes
% What to do with the nominal?
mean(X(1:end,2))