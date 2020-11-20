m = 0.2;
g = 9.81;
l = 0.3;
I = 0.006;
b = 0.012;

a_0 = -(m*g*l)/(I+m*l^2)
a_1 = b/(I+m*l^2)
b_0 = m*l/(I+m*l^2)
s_1 = (-a_1+sqrt(a_1^2-4*a_0))/2
s_2 = (-a_1-sqrt(a_1^2-4*a_0))/2

h = sym('h');
s = sym('s');
%a_0 = sym('a_0');
%a_1 = sym('a_1');

A_c = [0 1; -a_0 -a_1];
B_c = [0; b_0];

A = simplify(ilaplace(inv(eye(2)*s - A_c)))
B = simplify(int(A*B_c, sym('t'), 0, h))
C = [1 0];
D = 0;

% h_num_arr =  [0.001:0.001:.1];
% 
% P_cont = pole(ss(A_c,B_c,C,D));
% 
% P = [];
% P_exp = [];
% for h_num = h_num_arr
% A_it = eval(subs(A, 't', h_num));
% B_it = eval(subs(B, 'h', h_num));
% sys = ss(A_it,B_it,C,D);
% P = [P pole(sys)];
% P_exp = [P_exp exp(P_cont*h_num)];
% end
% plot(h_num_arr,P)
% hold on
% plot(h_num_arr,P_exp, '--')
% legend({'z_1', 'z_2', 'e^{s_1h}', 'e^{s_2h}'})
% xlabel('h')

% figure();
% P_cont = [];
% for h_num = h_num_arr
% %sys = c2d(ss(A_c,B_c,C,D), h_num);
% sys = ss(A_c,B_c,C,D);
% P_cont = [P_cont pole(sys)];
% end
% plot(h_num_arr,P_cont)
