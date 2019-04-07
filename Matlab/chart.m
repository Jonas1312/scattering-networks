function chart(filename,n_scale,n_rot)
%Fonction: Creer la representation similaire a celle de Bruna et Mallat (et
%qui sert de deuxieme systeme de features dans le rapport), s'applique
%forcement a une image carree
%Exemple: chart("C:\Users\Antoine\Documents\MATLAB\pfe\cropped_uiuc\22\851.jpg",7,5)

    I = imread(filename);
    % I = rgb2gray(I);
    I = im2double(I);

    filt_opt.J = n_scale; %J va de 0 a 6
    filt_opt.L = n_rot; %L va bien de 1 a 5 (voir doc pour angle)
    scat_opt.M = 1;
    scat_opt.oversampling = 0;

    [Wop, filters] = wavelet_factory_2d(size(I), filt_opt, scat_opt);
    Sx = scat(I, Wop);

    color_maps = cell(1,n_scale);

    %labels et labels vides
    angles = (0:n_rot-1)  * pi / n_rot;
    labels = ["0 rad"];
    empty_labels = [""];

    for theta=2:numel(angles) %2 pour ne pas ajouter l'angle nul qui fait exception pour le formatage du label
        label =  num2str(angles(theta));
        label = label(1:3) + " rad";
        labels = [labels label];
        empty_labels = [empty_labels ""];
        labels = cellstr(labels);
        empty_labels = cellstr(empty_labels);
    end
    
    %Trouvons le coeff pour normaliser les couleurs
    tarte = 0;
    for p=1:n_scale*n_rot
      coef = Sx{2}.signal{p};
      tarte = max(tarte,coef);
    end     

    for j=0:n_scale-1

        %On cherche les coefficients correspondant a chaque rotation
        pieces_of_pie = [];
        for r=1:n_rot
            p = find(Sx{2}.meta.j(1,:) == j & ...
                Sx{2}.meta.theta(1,:) == r);
            pieces_of_pie = [pieces_of_pie Sx{2}.signal{p}];
        end

        %On normalise
        pieces_of_pie = pieces_of_pie./tarte;
        pieces_of_pie = double(pieces_of_pie);

        %On determine la color_map
        color_map = ones(n_rot,3);
        for i=1:n_rot
            color_map(i,:) = color_map(i,:).*pieces_of_pie(i);
        end
        color_maps{j+1} = color_map;

    end

    %Plot tout
    figure 
    for j=1:n_scale
        ax = subplot(1,n_scale+1,j+1); %ax est necessaire ici parce que sinon une colormap est valable pour une figure complete
        if j==1
            p = pie(ones(1,n_rot),labels);
        else
            p = pie(ones(1,n_rot),empty_labels);
        end
        colormap(ax,color_maps{j})
        set(findall(gcf,'-property','FontSize'),'FontSize',6) 
    end

    for j=1:n_scale %ajout des titres 
        ax = subplot(1,n_scale+1,j+1); 
        title(strcat('Scale=',int2str(2^(j-1))),'FontSize',10)
    end
    
    ax = subplot(1,n_scale+1,1); %ajout de l'image originale
    imshow(I)
    title("Original image",'FontSize',10)
end


