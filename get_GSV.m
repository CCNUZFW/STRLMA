%% 
% Step0: Set the parameters of the experiment
clc;
clear
%ndevices = 20;
%nMixtures = 32;         % How many mixtures used to generate data
%nChannels = 10;         % Number of channels (sessions) per device
%nFrames = 1000;         % Frames per device (10 seconds assuming 100 Hz)
%nWorkers = 2;           % Number of parfor workers, if available

% Pick random centers for all the mixtures.
%mixtureVariance = .10;
%channelVariance = .05;
%mixtureCenters = randn(nDims, nMixtures, ndevices);
%channelCenters = randn(nDims, nMixtures, ndevices, nChannels)*.1;
%traindeviceData = cell(ndevices, nChannels);
%testdeviceData = cell(ndevices, nChannels);
%deviceID = zeros(ndevices, nChannels);

% Create the random data. Both training and testing data have the same
% layout.
%disp('Create the random data');
%for s=1:ndevices
%    trainSpeechData = zeros(nDims, nMixtures);
%    testSpeechData = zeros(nDims, nMixtures);
%    for c=1:nChannels
%        for m=1:nMixtures
%            % Create data from mixture m for device s
%           frameIndices = m:nMixtures:nFrames;
%           nMixFrames = length(frameIndices);
%            trainSpeechData(:,frameIndices) = ...
%                randn(nDims, nMixFrames)/*sqrt(mixtureVariance) + ...
%                repmat(mixtureCenters(:,m,s),1,nMixFrames) + ...
%                repmat(channelCenters(:,m,s,c),1,nMixFrames);
%            testSpeechData(:,frameIndices) = ...
%                randn(nDims, nMixFrames)*sqrt(mixtureVariance) + ...
%                repmat(mixtureCenters(:,m,s),1,nMixFrames) + ...
%                repmat(channelCenters(:,m,s,c),1,nMixFrames);
%        end
%        traindeviceData{s, c} = trainSpeechData;
%        testdeviceData{s, c} = testSpeechData;
%       deviceID(s,c) = s;                 % Keep track of who this is
%    end
%end

%%
% Step0.5 输入语音提取特征
% disp('准备输入语音')
% y = audiorecorder(16000,16,2);
% disp('Start speaking.');
% recordblocking(y,10);
% disp('End of recording. Playing back ...');

%z = getaudiodata(y);
%filename = 'test.wav';
%audiowrite(filename,z,16000);
%sample = getaudiodata(y);

%[sample,fs]=audioread('test.wav');
%mf =melcepst(sample,fs);
%mfc = cmvn(mf', true);
%ce_test{1,1}=mfc;
%ce_test{2,1}=mfc;
%ce_test{3,1}=mfc;

%%
% Step1: Create the universal background model from all the training device data

%disp('Create the universal background model');
%nmix = nMixtures;           % In this case, we know the # of mixtures needed
%final_niter = 10;
%ds_factor = 1;
%ubm = gmm_em(traindeviceData(:), nmix, final_niter, ds_factor, nWorkers);
%ubm=gmm_em(ce_train(:),64,10,1,1);
% load ('UBM1.mat')
load ('TIM256ubm.mat')
pNum=45;
load('D:\研究生\特征\MFCC\ce.mat')
ce_train=ce(:,1:514);                                                                                                                    
ce_test=ce(:,515:642);

%%
% Step2: Now adapt the UBM to each device to create GMM device model.
% disp('Adapt the UBM to each device');
% map_tau = 10.0;
% config = 'mwv';
% ndevices=fileNum_train-2;
% nChannels= 192;
% gmm = cell(ndevices, 1);
% for s=1:ndevices
%     disp(['for the ',num2str(s),' device...']);
%     gmm{s} = mapAdapt(ce_train(s, :), ubm, map_tau, config);
% end
% 
% 

disp('Adapt the UBM to each speaker');
map_tau = 10.0;
config = 'mwv';
nSpeakers= pNum;
nChannels= 1;
gmm_train = cell(pNum*514, 1);
gmms_train = cell(pNum*514, 1);
gmms_trains = zeros(514*45,39,256);
gmm_test = cell(pNum*128, 1);
gmms_test = cell(pNum*128, 1);
gmms_tests = zeros(128*45,39,256);
for s=1:nSpeakers
    disp(['for the ',num2str(s),' speaker...']);
    for i=1:514
        gmm_train{((s-1)*514+i)} = mapAdapt(ce_train(s, i), ubm, map_tau, config);
        gmms_train{((s-1)*514+i)} = mapminmax(gmm_train{((s-1)*514+i)}.mu);
        gmms_trains((s-1)*514+i,:,:) = gmms_train{((s-1)*514+i)};
    end
    for j=1:128
        gmm_test{((s-1)*128+j)} = mapAdapt(ce_test(s, j), ubm, map_tau, config);
        gmms_test{((s-1)*128+j)} = mapminmax(gmm_test{((s-1)*128+j)}.mu);
        gmms_tests((s-1)*128+j,:,:) = gmms_test{((s-1)*128+j)};
    end
end



