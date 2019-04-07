%Code d'experimentation des CNN

%Ouvrir les donnees
location = 'C:\Users\Antoine\Documents\MATLAB\pfe\cropped_uiuc_255'; %Utilisation d'une version rognee du dataset pour diminuer un peu le temps de calcul
ds = imageDatastore(location,"IncludeSubfolders",true,'LabelSource','foldernames');

%Afficher quelques exemples
% figure;
% perm = randperm(240,20);
% for i = 1:12
%     subplot(3,4,i);
%     imshow(ds.Files{perm(i)});
% end

%Construction des sets
numTrainFiles = 30; %nbr de samples/classe utilises pour l'entrainement
[dsTrain,dsValidation] = splitEachLabel(ds,numTrainFiles,'randomize');

% On definit le reseau
layers = [
    imageInputLayer([256 256 1])
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','pool_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','pool_3')
    
    convolution2dLayer(3,64,'Padding','same','Name','conv_4')
    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','relu_4')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_4')
    
    convolution2dLayer(3,128,'Padding','same','Name','conv_5')
    batchNormalizationLayer('Name','BN_5')
    reluLayer('Name','relu_5')
    
    fullyConnectedLayer(25,'Name','fully_connected')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','classif_layer')];

% Options d'entrainement
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',dsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','parallel',...
    'MiniBatchSize', 16);

%Entrainement
net = trainNetwork(dsTrain,layers,options);

%Vraiment bien: on peut afficher tout le detail du reseau avec la commande
%analyzeNetwork(net)
%Test
YPred = classify(net,dsValidation);
YValidation = dsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

%Matrice de confusion
figure
plotconfusion(YValidation,YPred);
set(findall(gcf,'-property','FontSize'),'FontSize',5)


%Autres reseaux
% layers = [
%     imageInputLayer([256 256 1])
%     convolution2dLayer(3,16,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3,64,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     fullyConnectedLayer(25)
%     softmaxLayer
%     classificationLayer];
