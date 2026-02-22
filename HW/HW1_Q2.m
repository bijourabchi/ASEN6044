%%% Bijan Jourabchi
%%% ASEN 6044 HW1 Q2

clear; clc; close all
rng(100)
%%% Part a) _ITALIC TEXT_ 

x0 = 0.75;
N = 5;
bernoulliParameterEstimationSim(x0, N, true);

N = 10;
bernoulliParameterEstimationSim(x0, N, true);

N = 50;
bernoulliParameterEstimationSim(x0, N, true);

%%% Part b)
x_hat_vec = zeros(1,5000);

for j = 1:5000
    
    x_hat_vec(j) = bernoulliParameterEstimationSim(x0, 10, false);

end

avg = mean(x_hat_vec)
variance = var(x_hat_vec)

%%% Part c)
x0 = [2,3];
unknownScalarMeas(x0, 5, true);
unknownScalarMeas(x0, 10, true);
unknownScalarMeas(x0, 50, true);

%%% Part d)
x_hat_vec = zeros(2,5000);
for j = 1:5000
    x_hat_vec(:,j) = unknownScalarMeas(x0, 10, false);
end


avg = mean(x_hat_vec(1,:))
variance = var(x_hat_vec(1,:))

function [x_hat] = bernoulliParameterEstimationSim(x0, N, PLOT)

sample = rand(N,1);

Y = sample <= x0;

% results
m = sum(Y);

% MLE Result
x_hat = m/N;
% Likelihood curve

x = linspace(0,1,100);

L = -m*log(x) - (N-m)*log(1-x);

if PLOT
    fprintf("x_MLE = %.2f for %d samples \n",x_hat,N)
    figure(N); hold on; grid on
    
    plot(x,L,'k',LineWidth=1.5)
    xline(x_hat,'r--',LineWidth=2)
    xline(x0,'b--',LineWidth=2)
    
    title("Negative Log-Likelihood Function vs x_0 for " + N + " samples")
    xlabel("x_0")
    ylabel("Negative Log-Likelihood")
    
    legend(["Negative Log-Likelihood", "MLE Estimate", "True x_0"])
end
end

function [x_hat] = unknownScalarMeas(x0, N, PLOT)

% unpack
mu = x0(1);
sigmasqr = x0(2);

% Sample
Y = normrnd(mu,sqrt(sigmasqr),1,N);

% MLW results
mu_hat = sum(Y)/N;
temp = (Y - mu_hat).^2;
sigmasqr_hat = sum(temp) / N;

x_hat = [mu_hat; sigmasqr_hat];

% Negative Log Liklihood
sigmasqrRange = linspace(0.5,8,100);
muRange = linspace(-3,3,100);

L = zeros(100,100);

for i = 1:100
    for j = 1:100
        L(j,i) = -log((1/((2*pi*sigmasqrRange(j))^(N/2))) * exp((-1/(2*sigmasqrRange(j)) * sum((Y-muRange(i)).^2))));
    end
end

if PLOT

    figure(N+1); hold on; grid on
    contourf(muRange, sigmasqrRange, L)
    plot(mu_hat, sigmasqr_hat, 'r*', 'MarkerSize', 15, 'LineWidth', 2)
    title("Negative Log-Likelihood Function vs x_0 for " + N + " samples")

end

end

