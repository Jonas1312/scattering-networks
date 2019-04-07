clear
x = imread('roto_trans_test.jpg');
x = im2double(x);
x = x(128:128+255, 128:128+255);

y = imread('roto_trans_test.jpg');
y = im2double(y);
y = y(128:128+255, 128:128+255);

% scat_opt.M = 2; %ordre max
% filt_opt.J = 2;
% filt_opt.L = 8;
filt_opt = struct;
filt_rot_opt = struct;
% filt_rot_opt.J = 1;
% filt_rot_opt.L = 1;
% oversampling must be set to infty
% so that scattering of original and rotated
% image will be sampled at exactly the same points
 scat_opt.oversampling = 10;
 
[Wop, filters] = wavelet_factory_3d(size(x), filt_opt, filt_rot_opt, scat_opt);
[Wop2, filters2] = wavelet_factory_2d(size(y), filt_opt, scat_opt); %cellule regroupant les filtres, filters bien pour etudier les filtres

Sx = scat(x, Wop);
Sy = scat(y, Wop2);

sx_raw = format_scat(Sx);
sy_raw = format_scat(Sy);

% rotated roto-trans scattering
x_rot = rot90(x,1);
Sx_rot = scat(x_rot, Wop);
sx_rot_raw = format_scat(Sx_rot);

% rotated scattering
y_rot = rot90(y,1);
Sy_rot = scat(y_rot, Wop2);
sy_rot_raw = format_scat(Sy_rot);


% rotate back
for p = 1:size(sx_rot_raw,1)
     tmp = rot90(squeeze(sx_rot_raw(p,:,:)), -1);
     sx_rot_raw_back(p,:,:) = permute(tmp, [3, 1, 2]);
end

for p = 1:size(sy_rot_raw,1) 
     tmp2 = rot90(squeeze(sy_rot_raw(p,:,:)), -1);
     sy_rot_raw_back(p,:,:) = permute(tmp2, [3, 1, 2]);
end

 
% compute norm ratio
norm_diff = sqrt(sum((sx_rot_raw_back(:)-sx_raw(:)).^2));
norm_s = sqrt(sum((sx_raw(:)).^2));
norm_ratio = norm_diff/norm_s

u = sy_rot_raw_back(:)-sy_raw(:);
v = u.^2;
norm_diff2 = sqrt(sum(v));
norm_s2 = sqrt(sum((sy_raw(:)).^2));
norm_ratio2 = norm_diff2/norm_s2


