function scatter_dataset(set_path, name_new_folder, image_size, J, L, M)
%Fonction: Transforme un data set complet avec la scattering transform et enregistre
%le resultat de facon structuree (un dossier par classe)
%Exemple: scatter_dataset('C:\Users\Antoine\Documents\MATLAB\pfe\uiuc_texture_dataset', 'scattered_uiuc_M1_J5_L5', [480, 640],5, 5, 1)

ds = imageDatastore(set_path,"IncludeSubfolders",true,'LabelSource','foldernames');

filt_opt.J = J;
filt_opt.L = L;
scat_opt.M = M;
scat_opt.oversampling = 0;

Wop = wavelet_factory_2d(image_size, filt_opt, scat_opt);

mkdir(strcat('C:\Users\Antoine\Documents\MATLAB\pfe\',name_new_folder));
existing_dirs = {};

c=1; %des compteurs pour generer les noms de fichier
d=1;

while hasdata(ds) %pour tout le data set
    [I, info] = read(ds); %info pour obtenir le label
    label = char(info.Label);

    I = im2double(I);
    %Si label jamais rencontre, on cree le dossier
    if ~any(strcmp(existing_dirs,label)) %any regarde dans le vecteur logique s'il y a au moins un 1 (ismember renvoit un vecteur...)
       existing_dirs{d} = label;
       disp(strcat('C:\Users\Antoine\Documents\MATLAB\pfe\',name_new_folder,'\',label,' cree'))
       mkdir(strcat('C:\Users\Antoine\Documents\MATLAB\pfe\',name_new_folder,'\',label));
       d = d+1;
    end
    %On calcule la scattering transform 
    scattered_image = format_scat(scat(I,Wop));
    filename = strcat(int2str(c),'.mat');
    c = c+1;
    
    %On enregistre la scattering transform du label sous forme de matrice
    %dans le bon dossier
    save(strcat('C:\Users\Antoine\Documents\MATLAB\pfe\',name_new_folder,'\',label,'\',filename),'scattered_image');
    
end
end

