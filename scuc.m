clear;
clc;
close all;
P1=sdpvar(25,1);
P2=sdpvar(25,1);
P3=sdpvar(25,1); 
P1(1)=0;
P2(1)=0;
P3(1)=0; 
Fc1=sdpvar(25,1);
Fc2=sdpvar(25,1);
Fc3=sdpvar(25,1);
I=[1;1;2;2;4;5;3];
J=[2;4;3;4;5;6;6];
G=digraph(I,J);
IN=-incidence(G);
Pmax=[220;100;20];
Pmin=[90;10;10];
SU=[100;300;0];
UR=[50;40;15];
DR=[50;40;15];
Ton=[4;2;1];
Toff=[-4;-3;-1];
I1t=binvar(25,1);
I2t=binvar(25,1);
I3t=binvar(25,1);
I1t(1)=0;
I2t(1)=0;
I3t(1)=0;
p=sdpvar(7,1);
Pflow=[200;100;10000;100;10000;100;10000];
theta= sdpvar(6,1);
X=[0.17;0.258;0.037;0.197;0.037;0.14;0.018];
z=0;
for m=2:25
Fc1(m)=(176.9+13.5*P1(m)+0.0004*P1(m) *P1(m))*1.2469;
Fc2(m)=(129.9+32.6*P2(m)+0.001*P2(m) *P2(m))*1.2461;
Fc3(m)=(137.4+17.6*P3(m)+0.005*P3(m) *P3(m))*1.2462;
  z= Fc1(m)* I1t(m)+Fc2(m)* I2t(m)+Fc3(m)* I3t(m)+z;
if value(I1t(m)- I1t(m-1))==1
   z=z+SU(1);
  end
  if value(I2t(m)- I2t(m-1))==1
   z=z+SU(2);
  end
  if value(I3t(m)- I3t(m-1))==1
   z=z+SU(3);
  end
end
Pd=[219.19,253.35,234.67,236.73,239.06,244.48,273.39,290.40,283.56,281.20,328.61,328.10,326.18,323.60,326.86,287.79,260,246.74,255.97,237.35,243.31,283.14,283.05,248.75];
Pw=[44,70.2,76,82,84,84,100,100,78,64,100,92,84,80,78,32,4,8,10,5,6,56,82,52];
for m=2:25
Cons=[P1(m)* I1t(m)+ P2(m)* I2t(m)+ P3(m)* I3t(m)+Pw(m-1)==Pd(m-1),P1(m)- P1(m-1)<=(1- I1t(m)*(1-I1t(m-1)))*UR(1)+ I1t(m)* (1-I1t(m-1))*Pmin(1), P2(m)- P2(m-1)<=(1- I2t(m)*( 1- I2t(m-1)))*UR(2)+ I2t(m)* (1- I2t(m-1))*Pmin(2), P3(m)- P3(m-1)<=(1- I3t(m)*( 1- I3t(m-1)))*UR(3)+ I3t(m)* (1- I3t(m-1))*Pmin(3),P1(m-1)- P1(m)<=(1- I1t(m-1)*( 1- I1t(m)))*DR(1)+ I1t(m-1)* (1-I1t(m))*Pmin(1), P2(m-1)- P2(m)<=(1- I2t(m-1)*( 1- I2t(m)))*DR(2)+ I2t(m-1)* ( 1- I2t(m))*Pmin(2), P3(m-1)- P3(m)<=(1- I3t(m-1)*( 1- I3t(m)))*DR(3)+ I3t(m-1)* ( 1- I3t(m))*Pmin(3),Pmin(1) * I1t(m)<= P1(m), P1(m)<= Pmax(1) * I1t(m), Pmin(2) * I2t(m)<= P2(m), P2(m)<= Pmax(2) * I2t(m), Pmin(3) * I3t(m)<= P3(m), P3(m)<= Pmax(3) * I3t(m),IN*p==[ P1(m); P2(m);0;Pw(m-1)- Pd(m-1);0; P3(m)], -Pflow <= p, p <=Pflow];
end
ops=sdpsettings('solver','bnb');
sol=optimize(Cons,z,ops); 
s_P1=value(P1)
s_P2=value(P2)
s_P3=value(P3)
s_z=value(z)
s_I1t=value(I1t)
s_I2t=value(I2t)
s_I3t=value(I3t)
