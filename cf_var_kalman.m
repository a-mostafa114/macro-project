function [u_cforecast,s_cforecast]=cf_var_kalman(data,num_maxlag,cf)
[tt,nn1]=size(data);
[tf,nn2]=size(cf);
if nn2~=nn1
   error('number of columns in cf and data do not equal') 
end
 T=tt+tf-num_maxlag;
  tff=isoutlier(data,'gesd');
  data(tff==1)=NaN;
data1=[data;cf];
for i=0:num_maxlag
lag_data(:,:,i+1)=data1(tt+tf+1-i-T:tt+tf-i,:);
end
zero_m=zeros(nn1,nn1);
identity_m=eye(nn1,nn1);
C=eye((num_maxlag+1)*nn1,(num_maxlag+1)*nn1);
a1=NaN(nn1,(num_maxlag+1)*nn1);
a3=zeros(nn1,(num_maxlag+1)*nn1);
a2=eye((num_maxlag-1)*nn1,(num_maxlag+1)*nn1);
A=[a1;a2;a3];
b1=zeros((num_maxlag)*nn1,1);
b2=ones(nn1,1);
B=[b1;b2];
mmd1=ssm(A,B,C);
%% initial value for parameter
md0=varm(nn1,num_maxlag);
estmd0=estimate(md0,lag_data(:,:,1));
AR_C=[];
for j=1:num_maxlag
    AR_C=[AR_C estmd0.AR{j}];
end
AR_C=[AR_C estmd0.Covariance];
AR_C1=AR_C(:);
%% set estimation ssm and forecast
y=lag_data(:,:,1);
for o=1:num_maxlag-1
    y=[y lag_data(:,:,o+1)];
end
epsilon=NaN(T,nn1);
y=[y epsilon];
 output=refine(mmd1,y,AR_C1)
 logL=cell2mat({output.LogLikelihood})';
 [~,maxlog]=max(logL);
 refinedparam1=output(maxlog).Parameters

[EstMdl,estParams,~,~,~] = estimate(mmd1,y,refinedparam1);

X = smooth(EstMdl,y)
isnany=isnan(y);
yy=y;
for i=1:size(X,1)
    for j=1:size(X,2)
        if isnany(i,j)==1
            yy(i,j)=X(i,j);
        end
    end
end
yyfir=yy(2:end,1:end-nn1);
yyres=yy(1:end-1,end-nn1+1:end);

yy=[yyfir yyres];

ff=reshape(estParams,nn1,size(estParams,1)/nn1)
 A2=[ff;a2;a3];
state_t=yy*A2;
c_f=state_t(end-tf+1:end,1:nn2);
u_cforecast=cf;
for k=1:size(u_cforecast,1)
index=isnan(u_cforecast(k,:));
u_cforecast(k,index)=c_f(k,index);
end
%%       SVAR
a_R=NaN(nn1,nn1);
a12=tril(a_R ,0);
a11=NaN(nn1,num_maxlag*nn1);
a1_s=[a11 a12];
A_s=[a1_s;a2;a3];
mmd2=ssm(A_s,B,C);
%paprm0
AR_C11=AR_C1(1:num_maxlag*nn1*nn1,1);
AR_C21=tril(estmd0.Covariance);
AR_C21=AR_C21(:);
AR_C21(AR_C21==0)=[];
AR_C2=[AR_C11;AR_C21];
%% estimate SVAR
 output2=refine(mmd2,y,AR_C2);
 logL2=cell2mat({output2.LogLikelihood})';
 [~,maxlog2]=max(logL2);
 refinedparam2=output2(maxlog2).Parameters;

[EstMd2,estParams2,~,~,~] = estimate(mmd2,y,refinedparam2);
X2 = smooth(EstMd2,y)
isnany=isnan(y);
yy2=y;
for i=1:size(X2,1)
    for j=1:size(X2,2)
        if isnany(i,j)==1
            yy2(i,j)=X2(i,j);
        end
    end
end
 yyfir2=yy2(2:end,1:end-nn1);
yyres2=yy2(1:end-1,end-nn1+1:end);

yy2=[yyfir2 yyres2];
A21_s=reshape(estParams2(1:num_maxlag*nn1*nn1,1),nn1,num_maxlag*nn1);
a22_st=estParams2(num_maxlag*nn1*nn1+1:end,1)
%convet vector to tril matrix
az=1;
bz=0;
for i=1:nn1
az=bz+1;
bz=bz+nn1-i+1;
a22{1,i}=zeros(i,1);
a22{2,i}=a22_st(az:bz,1);
a22_s(:,i)=[cell2mat(a22(1,i));cell2mat(a22(2,i))];
end
a22_s=a22_s(2:end,:);

A2_s=[A21_s a22_s];
A2_s=[A2_s;a2;a3];
state_t2=yy2*A2_s;
c_f2=state_t2(end-nn2:end,1:nn2);
s_cforecast=cf;
for k=1:size(s_cforecast,1)
index=isnan(s_cforecast(k,:));
s_cforecast(k,index)=c_f2(k,index);
end