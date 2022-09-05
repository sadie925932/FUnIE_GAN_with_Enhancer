clear all

% addpath('./UCIQE')

file_path = {'/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/output/AL_NG_OVD/GSoP/'};
file_path = {'/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/test/A/'}

dir_list = dir(string(file_path));
dir_num = length(dir_list);
uciqe = [];

for i = 1%:(dir_num-3)
    if i ~= 114 
        img = imread(strcat(string(file_path),string(i),".jpg"));
        u = UCIQE(img);
        uciqe(i) = u;
    end
end
disp("Mean:");
mean(uciqe)
disp("Std:")
std(uciqe)