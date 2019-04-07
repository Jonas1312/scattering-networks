src = uiuc_src('C:\Users\Antoine\Documents\MATLAB\pfe\uiuc_texture_dataset'); %creation de la source, derouler pour voir composition

n_files = size(src.files);
n_files = n_files(2);

current_folder = '1';
mkdir('C:\Users\Antoine\Documents\MATLAB\pfe\cropped_uiuc_255\1');

for k=1:n_files
    
    disp(k);
    
    I = data_read(src.files{k});
    cropped_image = imcrop(I,[100,100,255,255]);
    
    filename = strcat(int2str(k),'.jpg');
    imwrite(cropped_image,strcat('C:\Users\Antoine\Documents\MATLAB\pfe\cropped_uiuc_255\',current_folder,'\',filename))
    
    if mod(k,40)==0
        current_folder = int2str(fix(k/40)+1);
        mkdir(strcat('C:\Users\Antoine\Documents\MATLAB\pfe\cropped_uiuc_255\', current_folder));
    end
end
