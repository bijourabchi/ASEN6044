%%% ASEN 6044 HW2 Q3e

%%% Load Datafile
clear; clc; close all
load("hw2_missileprob_data.mat")

%%% Constants
N = size(ystacked,1)/2;
np = 2;
L = 80000;
dt = 5; % sec
t = dt:dt:N*dt;

fun = @(x) Liklihood(x, ystacked,t);
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations', 30000, 'MaxIterations', 10000);

x0 = [50;300;20;100];

[valid,err] = checkGradients(fun,x0,Display="on");

[x,fval] = fminunc(fun,x0,options);


%%% Plot Results
t = [0, t];
results(x0true,x,ystacked,t,L)

function [Like,G] = Liklihood(x,Y,t)

% Unpack
xi = x(1);
xid = x(2);
a = x(3);
ad = x(4);
rho = Y(1:2:end);
theta = Y(2:2:end);

% Constants
nu = 1/2;
sigma2 = 0.005;
L = 80000;
g = 9.81;

% Dynamics
xit = xi + xid*t;
at = a+ad*t-0.5*g*t.^2;

% Force col vector
xit = xit(:);
at = at(:);

% P(rho_k|x0)

% Constant terms
C = gamma(((nu+1)/2))/gamma(nu/2) * (1/(sqrt(nu*pi)));

Pr = C*((1 + ((rho - sqrt((L - xit).^2+(at).^2)).^2)/nu)).^(-(nu+1)/2);

% P(theta_k|x0)

% Constant term
C = 1/(sqrt(sigma2*2*pi));

Pb = C*exp((-(theta-atan2(at,(L-xit))).^2)/(2*sigma2));

Like = -sum(log(Pr.*Pb));

%% Gradient Vector
if nargout > 1
    t = t(:);
    % First Vector
    C1 = (nu+1)/2;

    G11 = (2*(rho - sqrt((L - xit).^2+(at).^2)).*(L-xit))./((((rho - sqrt((L - xit).^2+(at).^2)).^2) + nu).*(sqrt((L - xit).^2+(at).^2)));
    G12 = G11.*t;
    G13 = (-2*(rho - sqrt((L - xit).^2+(at).^2)).*at)./((((rho - sqrt((L - xit).^2+(at).^2)).^2) + nu).*(sqrt((L - xit).^2+(at).^2)));
    G14 = G13.*t;

    G11 = sum(G11);
    G12 = sum(G12);
    G13 = sum(G13);
    G14 = sum(G14);

    C2 = 1/(2*sigma2);
    G21 = (-2*(theta-atan2(at,(L-xit))).*at)./(((at./(L-xit)).^2 + 1).*(L-xit).^2);
    G22 = G21.*t;
    G23 = (-2*(theta-atan2(at,(L-xit))))./(((at./(L-xit)).^2 + 1).*(L-xit));
    G24 = G23.*t;

    G21 = sum(G21);
    G22 = sum(G22);
    G23 = sum(G23);
    G24 = sum(G24);

    G = C1*[G11;G12;G13;G14]+ C2*[G21;G22;G23;G24];
end
end

%%% Plotting Functions
function [] = results(x0t,x0_hat,Y,tvec,L)

    N = length(tvec);

    % Get Truth/Estimated Traj

    trajT = zeros(2,N);
    trajE = zeros(2,N);
    
    
    for k = 1:N
        [xi,a] = f(x0t,tvec(k));
        trajT(:,k) = [xi;a];
        [xi,a] = f(x0_hat,tvec(k));
        trajE(:,k) = [xi;a];
    end

    % Get measurment in x,y coords
    for i = 1:2:2*N
        if i > 80 break; end

        aM(i) = Y(i)*sin(Y(i+1));
        eM(i) = L - Y(i)*cos(Y(i+1));
    end

    
    figure(2); hold on; grid on
    scatter(trajE(1,:),trajE(2,:), 's', 'MarkerFaceColor', [0.5 0 0.8], 'MarkerEdgeColor', [0.5 0 0.8])
    scatter(trajT(1,:),trajT(2,:),100,'x','Color', '#40E0D0')
    scatter(eM,aM,'ro')
    legend('ML Estimate','Ground Truth','Measurments')
    xlabel("Easting (m)")
    ylabel("Altitude (m)")
    title("ML Missile Tracking")
    

end

function [y] = h(x0,t, L)

    
    vk = [0 0];

    [xik,ak] = f(x0,t);

    rho = sqrt((L - xik)^2 + ak^2) + vk(1);
    theta = atan2(ak,(L-xik)) + vk(2);

    y = [rho;theta];
end

%%% Dynamics function
function [xik,ak] = f(x0,t)

    xi0 = x0(1);
    xid0 = x0(2);
    a0 = x0(3);
    ad0 = x0(4);

    g = 9.81;

    xik = xi0 + xid0*t;
    ak = a0 + ad0*t - 0.5*g*t^2;
end