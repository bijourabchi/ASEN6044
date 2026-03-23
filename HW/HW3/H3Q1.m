clear; clc; close all

rng(500)
%% Define grid space + paramters

dx = 0.05;
x = 1:dx:14;
z = dx:dx:30;
a = 2;
b = 2;

Nx = length(x);
Nz = length(z);
%% Define prior grid dist

px = zeros(1,Nx);
px(x >= 1 & x <= 11) = 1/(11-1);

pz = (1/(b^a * gamma(a))) .* (z.^(a-1)) .* exp(-z./b);

P = zeros(Nx,Nz);

for i = 1:Nx
    for j = 1:Nz
        P(i,j) = px(i) * pz(j);
    end
end

P = P / sum(P(:));

[X,Z] = meshgrid(z,x);

figure
surf(X,Z,P)
xlabel('z')
ylabel('x')
zlabel('p(x,z)')
title('Joint Prior Distribution p(x,z)')
shading interp

%% Posterior Bayes Update

y = [91.56, 70.43, 108.67];
Ny = 3;

L = likelihood(y(1),x,z);

figure
surf(X,Z,L)
xlabel('z')
ylabel('x')
zlabel('p(y|x,z)')
title('Likelihood Dist p(y|x,z)')
shading interp

for i = 1:Ny
    Posterior = BayesUpdate(y(i),x,z,P, true,i);
    P = Posterior;
    MAP = getMAP(P,x,z);
    MMSE = getMMSE(P,x,z);
end

%% IS

px = zeros(1,Nx);
px(x >= 1 & x <= 11) = 1/(11-1);

pz = (1/(b^a * gamma(a))) .* (z.^(a-1)) .* exp(-z./b);

P = zeros(Nx,Nz);

for i = 1:Nx
    for j = 1:Nz
        P(i,j) = px(i) * pz(j);
    end
end

Ns = 500; % Sample Count
% Since x/z are independant, we can sample from individual dist
xi = 1 + 10*rand(Ns,1);
zi = gamrnd(a,b,Ns,1);
w = zeros(1,500);

for i = 1:Ns

w(i) = likelihood(y(1),xi(i),zi(i))*likelihood(y(2),xi(i),zi(i))*likelihood(y(3),xi(i),zi(i));

end

w = w./sum(w); % Normalize

% MMSE estimate
IS_MMSE = [0;0];
for i = 1:Ns
    IS_MMSE = IS_MMSE + w(i)*[xi(i);zi(i)];
end

figure; hold on
scatter3(zi, xi, w, 40, w, 'filled')
plot3([IS_MMSE(1) IS_MMSE(1)],[IS_MMSE(2) IS_MMSE(2)],[0 max(w)],'k--')
xlim([1 14])
ylim([0 30])
xlabel('z')
ylabel('x')
zlabel('Importance Weight')
title('Importance Sampling Weighted Particles')

colorbar
colormap turbo
grid on

%% Helper Functions

function [L] = likelihood(y,x,z)

Nx = length(x);
Nz = length(z);
L = zeros(Nx,Nz);

for i = 1:Nx
    for j = 1:Nz
        % Out of bounds
        if y < x(i) || y > (x(i)+z(j))^2
            L(i,j) = 0;
            continue
        end

        L(i,j) = 1/((x(i)+z(j))^2-x(i));
    end
end
    
    
end

function [Posterior] = BayesUpdate(y,x,z,Prior, PLOT,i)

L = likelihood(y,x,z);

Posterior = Prior.*L;
fprintf("Value of Marginal P(y_1:N) = %d for meas 1:%i \n",sum(Posterior(:)),i)
Posterior = Posterior./(sum(Posterior(:)));



if PLOT
    MAP = getMAP(Posterior,x,z);
    MMSE = getMMSE(Posterior,x,z);
    [X,Z] = meshgrid(z,x);
    figure
    hold on
    surf(X,Z,Posterior)
    scatter3(MAP(1),MAP(2),MAP(3),'rx')
    plot3([MMSE(1) MMSE(1)],[MMSE(2) MMSE(2)],[0 MAP(3)],'k--')
    xlabel('z')
    ylabel('x')
    zlabel('p(x,z|y)')
    title(sprintf('p(x,z | y_1...y_%d)',i))    
    shading interp
end
end

function [MAP] = getMAP(P,x,z)

    [peak,idx] = max(P(:));
    [row, col] = ind2sub(size(P), idx);
    [X,Z] = meshgrid(z,x);
    peakx = X(row,col);
    peakz = Z(row,col);

    MAP = [peakx;peakz;peak];

end

function [MMSE] = getMMSE(P,x,z)
[X,Z] = meshgrid(z,x);

x_mmse = sum(sum(P .* X));
z_mmse = sum(sum(P .* Z));

[~, ix] = min(abs(x - x_mmse));
[~, iz] = min(abs(z - z_mmse));

MMSE = [x_mmse; z_mmse; P(ix,iz)];

end