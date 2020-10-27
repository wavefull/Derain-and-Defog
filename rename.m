path='/home/wavereid/linuxsoft/pytorch-CycleGAN-and-pix2pix-master/datasets/222/TestA/';
path2='/home/wavereid/linuxsoft/pytorch-CycleGAN-and-pix2pix-master/datasets/222/TestB/';
file = dir('/home/wavereid/linuxsoft/pytorch-CycleGAN-and-pix2pix-master/datasets/222/TestA/*.png'); 
len = length(file);
for i = 1 : len 
    oldname = strcat(path,file(i).name); 
    newname = strcat(path2,file(i).name(1:end-4),int2str(0),'.png'); 
    command = ['mv' 32 oldname 32 newname];
    status = dos(command);
    if status == 0
        disp([oldname, ' 已被重命名为 ', newname])
    else
        disp([oldname, ' 重命名失败!'])
    end
end
