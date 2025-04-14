function[Target_adj, Exp_Var_adj, CPI_Cs_adj, PPI_Cs_adj]=initial_dprocess(file_address,xls_name, ...
         Target_sheet,seasonal_adjust,Explanatory_sheet,CPI_comp,PPI_comp)
%%% xls_file_address : Address for Excel data file. the excel file can contain 4 sheet:
%  CPI (or Target ) variable, All expalnatory Variables,twelve CPI Components and PPI Components

%%% CPI_sheet: name of sheet that contains CPI index 

%%% CPI_comp: name of sheet that contains  12 CPI sub-indexes 

%%% PPI_comp: name of sheet that contains  8 PPI sub-indexes 

%%% seasonal_adjust==1 all variables seasonally adjust by X12 ,otherwise
% there is no seasonal adjustment

%%%%%%%%%%%%%%%% % Data structure in all sheets should follow below principles:
% - first column of each sheet display date( or time) variable
% - First row of the excel sheets include variable names
% - second row of  the excel sheets include variables Transformation format 
%- transformation format is a scaler number for each variable as below:
% [0 1 2 3 4 5]=[no change, Ln, Diff, Double Diff, Diff_Ln, Double Diff_Ln]
% - Data in each colum of excel begin from third row 
% - First sheet include only one variable
% - Starting date and last period of all variables (include components,
% Target ,Exp.) must be same.
% if transformation==1 , all variables transformation -- Ln(.) format 
% if transformation==2 , all variables transformation -- first difference(.) format
% if transformation==3 , all variables transformation -- Second difference(.) format
% if transformation==4 , all variables transformation -- Second difference(LN(.))
% format (growth)
% if transformation==5 , all variables transformation -- Second difference(Ln(.)) format
%if transformation==0 (and otherwise) no change in the variables 

if nargin<4
error('Error:insufficient input')
end
if nargin==4
 Explanatory_sheet=[];   
  CPI_comp=  []; 
  PPI_comp=  [];  
end
if nargin==5
  CPI_comp=  []; 
  PPI_comp=  [];  
end
if nargin==6
  PPI_comp=  [];  
end
if isempty(file_address)
    error('Error:file address')
end
%

if isempty(xls_name)
    error('Error:xls_name')
end
%
if isempty(Target_sheet)
    error('Error:Target Variable is empty')
else
file_address2 = strcat(file_address,xls_name);
Target=xlsread(file_address2,Target_sheet);
Tr=[Target(1,2)];
Target=Target(2:end,2:end);
data=[Target];
[T1,N1]=size(Target);
end
%Explanatory
if isempty(Explanatory_sheet)
 Exp_Var='without Explanatory Variables';
 N2=0;
 T2=T1;
else
Exp_Var=xlsread(file_address2 , Explanatory_sheet);
Tr=[Tr Exp_Var(1,2:end)];
Exp_Var=Exp_Var(2:end,2:end);
[T2,N2]=size(Exp_Var);
data=[data Exp_Var];
end
%CPI_COMp
if isempty(CPI_comp)
 CPI_Cs='without CPI Component';
 N3=0;
 T3=T1;
else
CPI_Cs=xlsread(file_address2 , CPI_comp);
Tr=[Tr CPI_Cs(1,2:end)];
CPI_Cs=CPI_Cs(2:end,2:end);
[T3,N3]=size(CPI_Cs);
data=[data CPI_Cs];
end
%
if isempty(PPI_comp)
PPI_Cs='without PPI Component';
 N4=0;
 T4=T1;
else
PPI_Cs=xlsread(file_address2 , PPI_comp);
Tr=[Tr PPI_Cs(1,2:end)];
PPI_Cs=PPI_Cs(2:end,2:end);
[T4,N4]=size(PPI_Cs);
data=[data PPI_Cs];
end
 

if T1==T2 && T1==T3 && T1==T4 
 if seasonal_adjust==1   
   file_address3 = strcat(file_address,'IRIS_Tbx') ;

     addpath(file_address3)
 irisstartup

q=qq(1900,1):qq(2000,1);
q=q';
[T,N]=size(data);
data_Adj=NaN([T,N]);

for i=1:N
    x1=tseries(q(1:T,1),data(1:T,i));
    x2=x12(x1);
    x3=x2.data;
    ns=size(x3,1);
    if ns~=T 
        error('Error : there are missing data in the variables')
    end
    data_Adj(1:ns,i)=x3;
end

 
[data_base_trans]=transformation2(data_Adj ,Tr);
else
[data_base_trans]=transformation2(data ,Tr);
 end
else
    error('Error: mismach time dimensions') 
 end
%%%%
Target_adj=data_base_trans(:,N1);
if N2~=0
Exp_Var_adj=data_base_trans(:,N1+1:N1+N2);
else
   Exp_Var_adj=Exp_Var;
end
%
if N3~=0
CPI_Cs_adj=data_base_trans(:,N1+N2+1:N1+N2+N3);
else
   CPI_Cs_adj=CPI_Cs;
end

if N4~=0
PPI_Cs_adj=data_base_trans(:,N1+N2+N3+1:end);
else
   PPI_Cs_adj=PPI_Cs;
end
 