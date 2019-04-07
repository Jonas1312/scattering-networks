%Script permettant de faire de la ST sur un dataset, l'enregistrement des
%matrices se fait ensuite a la main depuis le workspace. Ce script sera
%converti en fonction sous peu. La difference avec scatter_dataset est que
%l'on conserve l'entierete du jeu de donnees sous la forme d'une seule
%martrice pour les features et d'une matrice pour les labels.

src = uiuc_src('C:\Users\Antoine\Documents\MATLAB\pfe\uiuc_texture_dataset'); %creation de la source, derouler pour voir composition

filt_opt.J = 6;
filt_opt.L = 4;
scat_opt.M = 1;
scat_opt.oversampling = 0;
[Wop, filters] = wavelet_factory_2d([480, 640], filt_opt, scat_opt);
features0{1} = @(x)(sum(sum(format_scat(scat(x,Wop)),2),3));

db = prepare_database(src, features0);

% proportion of training example
prop = 1;
% split between training and testing
[train_set, test_set] = create_partition(src, prop);
% Options de la SVM
params = ['-q -c 0.001'];
%Pour la reproductibilite
% rng(1);
%Preparation des features

ind_features = [];
feature_class = [];

for k = 1:length(train_set)
    ind = db.indices{train_set(k)};
    ind_features = [ind_features ind];
    feature_class = [feature_class ...
        db.src.objects(train_set(k)).class*ones(1,length(ind))];
end

features = db.features(:,ind_features);