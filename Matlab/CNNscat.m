%Experimentation reseau diffusant

%Ouvrir les donnees
%Pour rappel, la Deep Learning Toolbox de Matlab n'implemente pas de layer
%de scattering transform. La ST est donc faite avant avec la fonction
%scatter_dataset et ce sont les matrices obtenues avec cette fonction qui
%sont chargees ici. Il est donc necessaire de definir une "Read function"
%(ici custom_reader.m) pour charger correctement le dataset

location = 'C:\Users\Antoine\Documents\MATLAB\pfe\scattered_uiuc_M1_J5_L5'; 
ds = imageDatastore(location,"IncludeSubfolders",true,'LabelSource','foldernames','FileExtensions','.mat','ReadFcn',@custom_reader);

%Construction des sets
numTrainFiles = 30;
[dsTrain,dsValidation] = splitEachLabel(ds,numTrainFiles,'randomize');

%On definit le reseau
layers2 = [
    imageInputLayer([15 20 26],'Name','scattered_input') %il faut bien adapter la taille de l'entree par rapport a la ST (pour J=5, L=5, M=1 480x640x1 devient 15x20x26
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','pool_2')
    
    convolution2dLayer(3,64,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(25,'Name','fully_connected')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','classif_layer')];
%all cnn reseau commun

%Options d'entrainement
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ... 
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',dsValidation, ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','parallel',...
    'MiniBatchSize', 16);

%Entrainement
net2 = trainNetwork(dsTrain,layers2,options);

%Test
YPred = classify(net2,dsValidation);
YValidation = dsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

%Matrice de confusion
figure
plotconfusion(YValidation,YPred);
set(findall(gcf,'-property','FontSize'),'FontSize',5)