clc;
clear;

DataFile_train='E:\train\';
%DataFile_test='E:\test\';
path_file_train=dir(DataFile_train);
%path_file_test=dir(DataFile_test);
fileNum_train = length(path_file_train);
%fileNum_test = length(path_file_test);
%pNum=filesnum(DataFile);
ce=cell(fileNum_train-2,642);
%ce_test=cell(fileNum_test-2,1);

for i=3:fileNum_train
    file = [DataFile_train,path_file_train(i).name];  % Read the file name from DataFile
    if (path_file_train(i).isdir == 1)
        x=find_wav(file);
        l=size(x);
        k=0;
        for j=1:l(1)
            [sample,fs]=audioread(x(j,:));
            %[c,fc]=melcepst(sample,fs);
            mf =melcepst(sample,fs,'0dD');             % MFCC extraction
            mfc = cmvn(mf', true);                     % Cepstrum mean-variance normalization
            ce{(i-2),j}=mfc;
        end
    end
end

MFCC=zeros(45*642,39*650);
MFCC_train=zeros(514*45,39*650);
MFCC_test=zeros(128*45,39*650);
for i=1:45
   for j=1:642
      MFCC((i-1)*642+j,:)=reshape((ce{i,j}(:,1:650)),1,39*650);
   end
end

for i=1:45
    MFCC_train((i-1)*514+1:514*i,:)=MFCC((i-1)*642+1:(i-1)*642+514,:);
    MFCC_test((i-1)*128+1:128*i,:)=MFCC((i-1)*642+515:i*642,:);
end
