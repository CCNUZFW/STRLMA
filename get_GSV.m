clc;
clear;
%%
% Step1: Create the universal background model from all the training device data

%disp('Create the universal background model');
%nmix = nMixtures;           % Number of Gaussian mixture components
%final_niter = 10;
%ds_factor = 1;
%ubm = gmm_em(traindeviceData(:), nmix, final_niter, ds_factor, nWorkers);
%ubm=gmm_em(ce_train(:),64,10,1,1);  % Create Universal Background Model

load('ubm64.mat')          % Load the created universal background model
load('MFCC_train.mat')      % Load the extracted MFCC_train
load('MFCC_test.mat')       % Load the extracted MFCC_test

%%
% Step2: Now adapt the UBM to each device to create GMM device model.

disp('Adapt the UBM to each device');
pNum=45;
map_tau = 10.0;
config = 'mwv';
nDevices= pNum;
nChannels= 1;
gmm_train = cell(pNum*514, 1);
gmms_train = cell(pNum*514, 1);
gmms_trains = zeros(514*45,39,64);
gmm_test = cell(pNum*128, 1);
gmms_test = cell(pNum*128, 1);
gmms_tests = zeros(128*45,39,64);
for s=1:nDevices
    disp(['for the ',num2str(s),' device...']);
    for i=1:514
        gmm_train{((s-1)*514+i)} = mapAdapt(MFCC_train(s, i), ubm, map_tau, config);
        gmms_train{((s-1)*514+i)} = mapminmax(gmm_train{((s-1)*514+i)}.mu);
        gmms_trains((s-1)*514+i,:,:) = gmms_train{((s-1)*514+i)};
    end
    for j=1:128
        gmm_test{((s-1)*128+j)} = mapAdapt(MFCC_test(s, j), ubm, map_tau, config);
        gmms_test{((s-1)*128+j)} = mapminmax(gmm_test{((s-1)*128+j)}.mu);
        gmms_tests((s-1)*128+j,:,:) = gmms_test{((s-1)*128+j)};
    end
end



