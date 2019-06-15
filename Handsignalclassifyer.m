%https://www.kaggle.com/koryakinp/fingers/version/2 data was taken from here
%C:\Users\jr08368\Desktop\HandSignals
digitDatasetPath = fullfile('C:\','Users','Justin','Desktop','HandSignals');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm = randperm(18000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
labelCount = countEachLabel(imds)
img = readimage(imds,1);
size(img)
% labelCount = countEachLabel(imds)
% img = readimage(imds,1);
% size(img)
numTrainFiles = 1125;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    imageInputLayer([128 128 1])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(12)
    softmaxLayer
    classificationLayer];     
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','parallel',...
       'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)
figure (2)
 C = confusionmat(YValidation,YPred);
 labels=labelCount(:,1);
 a=table2array(labels);
 cm=confusionchart(C,a);
 cm.Title='Hand Gesture Recognition Training Confusion Matrix';