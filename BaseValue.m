function [Sb,Ub,Ib,Zb,filename,Imax,vmax,vmin] = BaseValue(NetType)
switch NetType
    case 1
        Sb = 1e6;
        Ub = 12.35e3;
        Ib = Sb/(sqrt(3)*Ub);
        Pmax = 12e6;
        Zb = Ub/Ib/sqrt(3);
        filename = 'SCE47';
    case 2
        Sb = 1e6;
        Ub = 12e3;
        Ib = Sb/(sqrt(3)*Ub);
        Pmax = 12e6;
        Zb = 144;
        filename = 'SCE56';
end
Imax = Pmax/(sqrt(3)*Ub*Ib);
vmax = 1.1^2;
vmin = 0.9^2;
end