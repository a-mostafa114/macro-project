function[data_trans]=transformation2(Dataset ,Tr)
%[0 1 2 3 4 5]=[no change, Ln, Diff, Double Diff, Diff_Ln, Double Diff_Ln]


data_adj=Dataset;

[T,N]=size(data_adj);
data_trans=NaN(T,N);

for i=1:N
    if Tr(1,i)==0
        data_trans(1:T,i)=data_adj(1:T,i);
    elseif Tr(1,i)==1
        data_trans(1:T,i)=log(data_adj(1:T,i));
    elseif Tr(1,i)==2
        data_trans(2:T,i)=diff(data_adj(1:T,i),1) ./ data_adj(1:T-1,i);
    elseif Tr(1,i)==3
        data_trans(3:T,i)=diff(data_adj(1:T,i),2);
    elseif Tr(1,i)==4
        data_trans(2:T,i)=diff(log(data_adj(1:T,i)),1);
    elseif Tr(1,i)==5
        data_trans(3:T,i)=diff(log(data_adj(1:T,i)),2);
    end
end
s_data=zeros(1,N);
for j=1:N
    if Tr(1,j)==3 || Tr(1,j)==5
        s_data(1,j)=2;
    elseif Tr(1,j)==2 || Tr(1,j)==4
        s_data(1,j)=1;
    else
        s_data(1,j)=0;
    end
end
data_trans=data_trans(max(max(s_data),max(s_data))+1:end,:);


