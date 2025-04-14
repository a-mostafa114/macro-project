function [u_cf_rmse,s_cf_rmse]=cf_kalman_rmse(Data,num_maxlag,scf,num_steps)

ins=10;
[T,nn]=size(Data);
 
 u_cf_rmse=zeros(num_steps,nn);
s_cf_rmse=zeros(num_steps,nn);
cf_r=NaN(num_steps,nn);
for n=num_steps:ins+num_steps   
  var_temp=Data(1:end-n,:);
for i=1:nn
    if scf(1,i)==1
       cf_r(:,i)=Data(end-n+1:end-n+num_steps,i);
    end
end
 [u_cforecast,s_cforecast]=cf_var_kalman(var_temp,num_maxlag,cf_r);
 u_cf_rmse = u_cf_rmse+(u_cforecast-Data(end-n+1:end-n+num_steps,:)).^2;
 s_cf_rmse = s_cf_rmse+(s_cforecast-Data(end-n+1:end-n+num_steps,:)).^2;

end
 u_cf_rmse=(u_cf_rmse./ins).^0.5;
 s_cf_rmse=(s_cf_rmse./ins).^0.5;
