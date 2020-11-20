%% Design Project IV, EL2700: Model Predictive Control
% ======================================================================= %
% Script Summary:
% ---------------
% In this script, we will compute N-step controllable set for a given 
% dynamical system with external inputs and a given target set
% We will use MPT3 toolbox to compute N-step controllable set. To install
% MPT3 toolbox, check out this link:
% https://www.mpt3.org/Main/Installation
% ======================================================================= %
%% Parameters for inverted pendulum
% ======================================================================= %
clear all;
load('pendulum_parameters.mat', 'params')
m   = params.m;          % Mass of the inverted pendulum
l   = params.l;          % length to the pendulum center of mass
I   = params.I;          % moment of inertia of the pendulum
bp  = params.b;          % coefficient of friction for pendulum
g   = params.g;          % acceleration due to gravity
h   = params.h;          % Sampline time
% ======================================================================= %
%% Discrete-time linear model of the inverted pendulum
% ======================================================================= %

% Continous time A and B matrix
Ac = [[0,               1];
      [m*g*l/(I + m*l^2), -bp/(I + m*l^2)]];
Bc = [0; m*l/(I + m*l^2)];
 
[nx, nu] = size(Bc);            % State and input dimension
[A, B]   = c2d(Ac, Bc, h);      % Discrete-time model

% Penalty matrices
Q    = diag([1/((0.05)^2) 1/(3^2)]);% diag([1 0]);
R    = 1/(10^2);%1;

% Define the system as a MPT toolbox LTISystem
% The control constraints will be defined later. 
sys = LTISystem('A', A, 'B', B);
sys.x.min =  [-pi/2; -pi/2];
sys.x.max =  [ pi/2;  pi/2];
sys.x.penalty = QuadFunction(Q);
sys.u.penalty = QuadFunction(R);

% ======================================================================= %
%% State constraint
% Here you will construct the state constraints Polyhedron X_constraint. 
% This polyhedron is needed when computing the backwards reachable sets. 
% ======================================================================= %

Hx = [-eye(nx); eye(nx)];                  
hx = [-sys.x.min; sys.x.max];
X_constraint  = Polyhedron( 'A', Hx, 'b', hx);

% ======================================================================= %
%% Q1. Influence of control horizon on  N-step controllable set
% ======================================================================= %
% invariant_set_type = Type of invariant set 
% invariant_set_type = 0 => Xf is zero terminal constraint set
% invariant_set_type = 1 => Xf is the invariant set for the closed-loop
%                        dynamics under the infinite-horizon LQR control

invariant_set_type = 'lqr';
N    = [5, 10, 20];
sys.u.min = -20;
sys.u.max = 20;
figure('Name','Varying control horizon', 'Position', [150 150 900 600])
hold on
for i = 1:length(N)
    % Compute the invariant set depending on the selected type
     switch (invariant_set_type)
         case 'zero'
            Xf = Polyhedron([-eye(nx); eye(nx)], zeros(nx*2, 1));
         case 'lqr'
            Xf = sys.LQRSet();
         otherwise
            msg = "Please set the variable invariant_set_type to either 'lqr' or 'zero'.";
            close; error(msg);
     end
     
    % Iterate N(i) steps backwards to get KNS
    temp_reach_set = Xf;
    for j = 1 : N(end+1-i)
        backwards_step = sys.reachableSet('X', temp_reach_set, ...
                                          'direction', 'backward');
        temp_reach_set = intersect(backwards_step, X_constraint);
    end
    
    KNS(i) = temp_reach_set;
end
plot(KNS);

xlabel('Angle [rad]')
ylabel('Angular velocity [rad/s]')
title(['Varying control horizon,  terminal set = ', invariant_set_type, ' set' ])
xlim([-1.75 1.75])
ylim([-1.75 1.75])
legend([repmat('N = ',length(N),1),num2str(fliplr(N)')])

% ======================================================================= %
% Comment on how the contol horizon and invariant_set_type 
% the influence N-step controllable set 
% Answer:
%
%
%
% ======================================================================= %
%% Q2. Influence of control constraint on N-step controllable set with N = 5
% ======================================================================= %
% invariant_set_type = 'zero' => Xf is zero terminal constraint set
% invariant_set_type = 'lqr'  => Xf is the invariant set for the closed-loop
%                         dynamics under the infinite-horizon LQR control

invariant_set_type = 'lqr';
N = 5;
ulims = [[-5, 5]
               [-10, 10]
               [-15, 15]];
figure('Name','Varying control constraints', 'Position', [150 150 900 600])
hold on
for i = 1:length(ulims)
    % Set the input constraints
    sys.u.min = ulims(end+1-i,1);
    sys.u.max = ulims(end+1-i,2);
    
    % Compute the invariant set depending on the selected type
     switch (invariant_set_type)
         case 'zero'
            Xf = Polyhedron([-eye(nx); eye(nx)], zeros(nx*2, 1));
         case 'lqr'
         	Xf = sys.LQRSet();
         otherwise
         	msg = "Please set the variable invariant_set_type to either 'lqr' or 'zero'.";
            close; error(msg);
     end
    
     % Iterate N steps backwards to get KNS
    temp_reach_set = Xf;
    for j = 1 : N
        backwards_step = sys.reachableSet('X', temp_reach_set, ...
                                          'direction', 'backward');
        temp_reach_set = intersect(backwards_step, X_constraint);
    end
    
    KNS(i) = temp_reach_set;
    %KNS = temp_reach_set;
end
plot(KNS);


xlabel('Angle [rad]')
ylabel('Angular velocity [rad/s]')
title(['Varying input constraints,  terminal set = ', invariant_set_type, ' set'  ])
xlim([-1.75 1.75]);
ylim([-1.75 1.75]);
legend([repmat('|u_t| <= ',length(ulims),1),num2str(flipud(ulims(:,2)))])

% ======================================================================= %
% Comment on how the contol constraint and invariant_set_type 
% the influence N-step controllable set 
% Answer:
%
%
% ======================================================================= %


