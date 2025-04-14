function [su_cforecast,sc_cforecast]=cf_var_doan(data,num_maxlag,cf,var_cons)
%%conditional forecast calculation are based on Doan litterman and
%Sims(1984)

%%%%%%% IMPORTANT: Befor running this function,  path of IRIS package must be
% added. for example this command: addpath C:\Users...\IRIS-R2019b-and-Newer

%%% data :Matrix nn*T nn: number of endog. variables . T: number of periods
%%% num_maxlag: number of lag of variables in the model

%%% cf :  matrix num_step*nn which specify the senario for conditional
% forecast. num_step: number of steps ahead forcast. cells of this Matix
% fill by "NaN" except the senario set specific value for that cell. 

%%% var_cons: nn*nn matrix which specify restrictions on VAR coefficients.
% array of (n,m) in this matrix represents coefficient of  mth variables in
% the equation of nth variable. cells of this Matix % fill by "NaN" except 
% we put a value as a constraint for that coefficient. when we set array of 
% (n,m) equal =0 it means all lags of mth variable do not affect on n
% variable through the coefficients. 


%%


q=qq(1969,1):qq(2100,1);
q=q';
[T,nn]=size(data);
[t2,nn3]=size(cf);
if nargin<4
  var_cons=  NaN(nn,nn);
end 

[nn2,nn1]=size(var_cons);

if nn~=nn1||nn~=nn2||nn~=nn3
   error('number of columns in cf, rows and columns in vat_cons and columns in data must be equal') 
end

for i=1:nn
  name(1,i)=string(sprintf( 'V_%d', i )); 
 end
 for i=1:nn
    %name=convertCharsToStrings(sprintf( 'Vd%d',i));
    x1.(name(1,i))=tseries(q(1:T,1),data(1:T,i));
end
startHist = get(x1.(name(1,1)),'start');
endHist = get(x1.(name(1,1)),'end');
%%
%%%simple VAR
v = VAR(name); %?emptyVAR?
p = num_maxlag;
[v,vd] = estimate(v,x1,startHist:endHist, ...
    'order=',p,'const=',true, ...
    'covParameters=',true);  

%% restricted SVAR
   
constrA = NaN(nn,nn,p);
for o=1:p
constrA(:,:,o) = var_cons;
end

[rv1,rvd1] = estimate(v,x1,startHist:endHist, ...
    'order=',p,'const=',true, ...
    'covParameters=',true,'A=',constrA); 
%%
%  SVAR_chol %%%'method=','chol'. it could be 'qr', 'svd'  or 'householder'
[s0,sd0] = SVAR(v,vd,'method=','chol');
B0=get(s0,'B');

  [s1,sd1] = SVAR(rv1,rvd1,'method=','chol');
B1=get(s1,'B');

  
%%
% unconditional forecast
startFcast = endHist + 1;
endFcast = endHist + t2;
 u0 = forecast(s0,x1,startFcast:endFcast);
 u0.mean = dboverlay(x1,u0.mean);
 u1 = forecast(s1,x1,startFcast:endFcast);
 u1.mean = dboverlay(x1,u1.mean);
   
 %%
 % conditional forecast
%  senario
j1 = struct();
cff=sum(isnan(cf),1);
for i=1:nn
    if cff(1,i)<t2
   j1.(name(1,i))=tseries();
j1.(name(1,i))(startFcast:endFcast)= cf(:,i);
    end
end
  
cf01 = forecast(s0,x1,startFcast:endFcast,j1);
cf01.mean=dboverlay(x1,cf01.mean);
su_cforecast=cf;
for i=1:nn
su_cforecast(:,i)=cf01.mean.(name(1,i)).Data(end-t2+1:end,1) ;
end
cf11 = forecast(s1,x1,startFcast:endFcast,j1);
cf11.mean=dboverlay(x1,cf11.mean);
 sc_cforecast=cf;
for i=1:nn
sc_cforecast(:,i)=cf11.mean.(name(1,i)).Data(end-t2+1:end,1)  ;
end