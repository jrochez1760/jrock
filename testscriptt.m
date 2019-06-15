%C:\Users\Justin\Desktop\HandSignalstestdata
 load('handsignalrec.mat')

digitDatasetPath1 = fullfile('C:\','Users','Justin','Desktop','HandSignalstestdata');
imds2 = imageDatastore(digitDatasetPath1, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure (1);
perm = randperm(3600,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds2.Files{perm(i)});
end
labelCount = countEachLabel(imds2);
YPred2 = classify(net,imds2);
YTest2 = imds2.Labels;
% YPred = classify(net,imdsValidation);
% YTest = imds2.Labels;
accuracy = sum(YPred2 == YTest2)/numel(YTest2)
figure (1)
 C = confusionmat(YTest2,YPred2);
 labels=labelCount(:,1);
 a=table2array(labels)
 cm=confusionchart(C,a);
 cm.Title='Hand Gesture Recognition';