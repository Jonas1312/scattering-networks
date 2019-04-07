function data = custom_reader(filename)
%Fonction: Expliquee dans CNNscat.m
%Exemple: //
data = load(filename);
data = data(1);
data = data.scattered_image;
data = reshape(data,15,20,26);

end