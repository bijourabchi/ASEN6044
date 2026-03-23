%%% Bijan Jourabchi
%%% ASEN 6044 HW2 Q3

%%% Load Datafile
clear; clc; close all
load("hw2_missileprob_data.mat")

%%% Get Noise Covar matrix

N = size(ystacked,1)/2;
np = 2;
L = 80000;
dt = 5; % sec
t = dt:dt:N*dt;

R = [70 0; 0 0.005];
Rb = kron(eye(N), R);

%%% Whitening transformation

S = chol(Rb);
ya = S*ystacked;

%%% NLS
x0g = [10;30;230;100];
%x0g = [235;-150;200;2000];
%x0g = x0true;

% NLS
itr = 0;
maxitr = 100;
maxalph = 10;
converged = false;

while itr < maxitr && ~converged
    % Setup NLS
    yc = zeros(np,N);
    H = zeros(np*N,4);
    
    for k = 1:N
    
        yc(:,k) = h(x0g,t(k),R, L,true);
    
        %[xi,a] = f(x0g,t(k));
        H(2*k-1 : 2*k, :) = jcb(x0g,L,t(k));
    end
    yc = S*yc(:);
    H = S*H;

    % Cost fxn
    
    residuals = ya - yc;
    Jcurr = residuals'/Rb*residuals;
    dx = inv(H'*inv(Rb)*H)*H'*inv(Rb)*(ya - yc);
    alph = 1;
    alphsplts = 0;
    while alphsplts < maxalph

        xn0 = x0g + alph*dx;
        
        

        yc = zeros(np,N);
        for k = 1:N
            yc(:,k) = h(xn0,t(k),R, L,true);
        end
        yc = S*yc(:);

        Jn = (ya - yc)'/Rb*(ya - yc);

        if abs(Jcurr - Jn) < 50
            converged = true;
            break;
        end

        if Jn > Jcurr
            alph = alph/2;
        else
            x0g = xn0;
            break;
        end
        alphsplts = alphsplts + 1;
    end
    itr = itr + 1;
end

err = x0true - x0g %% ERROR

for k = 1:N

        H(2*k-1 : 2*k, :) = jcb(x0g,80000,t(k));
end

P = inv(H'*inv(Rb)*H);
plot_NLS(x0true,x0g,ystacked,t,Rb,L)


%%% Measurment function
function [y] = h(x0,t,R,L,noise)

    if noise
        vk = mvnrnd([0,0],R)';
    else 
        vk = [0 0];
    end

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

%%% Jacobian fxn 
function [H] = jcb(x,L,t)
    
    xi = x(1);
    xid = x(2);
    a = x(3);
    ad = x(4);
    g = 9.81;

    H11 = 2*(xid*t+xi-L)/sqrt((2*a-g*t^2+2*ad*t)^2 + 4*(xid*t+xi-L)^2);
    H12 = H11 * t;
    H13 = (2*a - g*t^2 + 2*ad*t) / sqrt((2*a-g*t^2+2*ad*t)^2 + 4*(xid*t+xi-L)^2);
    H14 = H13*t;

    H21 = 2*(2*a - g*t^2 + 2*ad*t)/((2*a - g*t^2 +2*ad*t)^2 + 4*(xid*t+xi-L)^2);
    H22 = H21*t;
    H23 = 4*(-xid*t-xi+L)/((2*a - g*t^2 +2*ad*t)^2 + 4*(xid*t+xi-L)^2);
    H24 = -t*H23;

    H = [H11 H12 H13 H14;H21 H22 H23 H24];
end

%%% Plotting Functions
function [] = plot_NLS(x0t,x0NLS,Y,tvec,R,L)
    
    N = length(tvec);

    % Get Truth/Estimated Traj

    trajT = zeros(2,N);
    trajE = zeros(2,N);
    
    
    for k = 1:N
        [xi,a] = f(x0t,tvec(k));
        trajT(:,k) = [xi;a];
        [xi,a] = f(x0NLS,tvec(k));
        trajE(:,k) = [xi;a];
    end

    % Get measurment in x,y coords
    rho = Y(1:2:end);
    theta = Y(2:2:end);
    for i = 1:N
        aM(i) = rho(i)*sin(theta(i));
        eM(i) = L - rho(i)*cos(theta(i));

        y_est = h(x0NLS,tvec(i),eye(2),L,false);
        rho_est(i) = y_est(1);
        theta_est(i) = y_est(2);
    end

    
    figure(4); hold on; grid on
    scatter(trajE(1,:),trajE(2,:), 's', 'MarkerFaceColor', [0.5 0 0.8], 'MarkerEdgeColor', [0.5 0 0.8])
    scatter(trajT(1,:),trajT(2,:),100,'x','Color', '#40E0D0')
    scatter(eM,aM,'ro')
    legend('NLS Estimate','Ground Truth','Measurments')
    xlabel("Easting (m)")
    ylabel("Altitude (m)")
    title("NLS Missile Tracking")

    fname = fullfile('Figures',['NLS_estimate' '.png']);

    % Set figure properties for saving
    fig = gcf;
    set(fig,'Color','w','PaperPositionMode','auto');
    
    % Save as PNG (use -r300 for high resolution)
    print(fig,'-dpng','-r300',fname);

     % Residual Plot
    figure(6); hold on; grid on
    tiledlayout(2,1)
    nexttile;
    plot(tvec,rho_est(:) - rho,'-o','Color','b')
    ylabel('\rho residuals')
    ylim([-2500 1500])
    nexttile;
    plot(tvec,theta_est(:) - theta,'-o','Color','r')
    xlabel("Time (s)")
    ylabel("\theta residuals")

    fname = fullfile('Figures',['NLS_estimate_Res' '.png']);

    % Set figure properties for saving
    fig = gcf;
    set(fig,'Color','w','PaperPositionMode','auto');
    
    % Save as PNG (use -r300 for high resolution)
    print(fig,'-dpng','-r300',fname);

    figure(67); hold on; grid on
    plot(tvec,rho_est(:) - rho,'-o','Color','b')
    ylabel('\rho residuals')
    xlabel("Time (s)")
    ylim([-2500 1500])

    fname = fullfile('Figures',['NLS_Rho_Res' '.png']);

    % Set figure properties for saving
    fig = gcf;
    set(fig,'Color','w','PaperPositionMode','auto');
    
    % Save as PNG (use -r300 for high resolution)
    print(fig,'-dpng','-r300',fname);
    

end
