clc;clear all;close all;

fid = fopen('/media/wavereid/3048847548843C1A/SPANet-master/testing/real_test_1000.txt','w');
% 
%personlist = dir('E:\classficatorder\'); personlist = personlist(3:end);
%for i=1:length(personlist)
    imglist = dir('/media/wavereid/3048847548843C1A/rain_100H1/speed/*.png');
    for j=1:length(imglist)
        
        fprintf(fid, '%s ', ['/media/wavereid/3048847548843C1A/rain_100H1/speed/' imglist(j).name]);
        fprintf(fid, '%s\r\n',['/media/wavereid/3048847548843C1A/rain_100H1/speed/' imglist(j).name]);
        
       
    end
%end
fclose(fid);