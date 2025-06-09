%% random forest
clc
clear
[data,txt,raw]=xlsread('.\rf_dataset06_spatial_newparameter.xlsx');
i_index_all=data(:,1);
uav_lon_all=data(:,2);
uav_lat_all=data(:,3);
tb_v_all=data(:,4);%
tb_h_all=data(:,5);%
original_uav_soilmoisture_all=data(:,6);
uav_soilmoisture_all=data(:,7);
hydrago_soilmoisture_all=data(:,8);
soil_temperature_all_all=data(:,9);
polra_name=txt(:,10);
flightcsv=txt(:,11);
landuse=txt(:,12);
landuse=landuse(2:end);
lat=data(:,13);
lon=data(:,14);
Demo=txt(:,15);
elevall=data(:,16);
flighthight=data(:,17);
UTC=data(:,18);
t2=data(:,19);
skt=data(:,20);
st=data(:,21);%
swv=data(:,22);
slope60m=data(:,23);
aspect60m=data(:,24);
srtm_60_05=data(:,25);
slope_30m=data(:,26);
aspect_30m=data(:,27);
SRTM_30m=data(:,28);
bulk_densi=data(:,29);
clay100=data(:,30);
sand100=data(:,31);
NDVI=data(:,32);
alt=data(:,33);
new_clay=data(:,34);
new_sand=data(:,35);
%%  import
res = data;
%%  20%，80%
seed=10;
rng(seed,"philox")
s = rng;
temp = randperm(length(res));

%% ten fold
[M,N]=size(res);
indices=crossvalind('Kfold',res(1:M,N),10);
error1_all=[];
error2_all=[];
for k=1:10
    test = (indices == k); 
    train = ~test;
    P_train=res(train,[4:5,21,33,29,34,35,32])';
    T_train=res(train,8)';%
    M = size(P_train, 2);
    P_test=res(test,[4:5,21,33,29,34,35,32])';
    T_test=res(test,8)';
    N = size(P_test, 2);
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input);
     
    [t_train, ps_output] = mapminmax(T_train, 0, 1);
    t_test = mapminmax('apply', T_test, ps_output);
    p_train = p_train'; p_test = p_test';
    t_train = t_train'; t_test = t_test';
     
    %%  train
    trees = 30;                                      
    leaf  = 5;                                        
    OOBPrediction = 'on';                             
    OOBPredictorImportance = 'on';                    
    Method = 'regression';                            
    net = TreeBagger(trees, p_train, t_train, 'OOBPredictorImportance', OOBPredictorImportance,...
          'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
    importance = net.OOBPermutedPredictorDeltaError;  
     
    %%  test
    t_sim1 = predict(net, p_train);
    t_sim2 = predict(net, p_test );
     
    T_sim1 = mapminmax('reverse', t_sim1, ps_output);
    T_sim2 = mapminmax('reverse', t_sim2, ps_output);
     
    %%  RMSE
    RMSE1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
    RMSE2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
   
    % pearson
    pearson1 = corrcoef(T_train,T_sim1');
    pearson2 = corrcoef(T_test,T_sim2');
    
    % R2
    R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
    R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;
     
    % MAE
    mae1 = sum(abs(T_sim1' - T_train)) ./ M;
    mae2 = sum(abs(T_sim2' - T_test )) ./ N;

    error1_all=[error1_all;RMSE1,pearson1(2),R1,mae1];
    error2_all=[error2_all;RMSE2,pearson2(2),R2,mae2];
end
disp(nanmean(error1_all)) 
disp(nanmean(error2_all)) 

P_train = res(temp(1: floor(length(res)*0.8)), [4:5,21,33,29,34,35,32])';
T_train = res(temp(1: floor(length(res)*0.8)), 8)';
M = size(P_train, 2);

P_test = res(temp((floor(length(res)*0.8)+1): end), [4:5,21,33,29,34,35,32])';
T_test = res(temp((floor(length(res)*0.8)+1): end), 8)';
N = size(P_test, 2);
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
 
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

leaf = [5 10 20 50 100];
col = 'rbcmy';
figure
for i=1:length(leaf)
    b = TreeBagger(30, p_train, t_train,'Method','R','OOBPrediction','On',...
            'MinLeafSize',leaf(i));
    plot(oobError(b),col(i))
    hold on
end
xlabel('Number of Grown Trees')
ylabel('Mean Squared Error (m^3/m^3)') 
legend({'5' '10' '20' '50' '100'},'Location','NorthEast')
hold off

trees = 30;  %100                                    
leaf  = 5;                                       
OOBPrediction = 'on';                             
OOBPredictorImportance = 'on';                    
Method = 'regression';                            
net = TreeBagger(trees, p_train, t_train, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;   
save('rf_net_newtexture_newparameter.mat','net');
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );
 
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
 
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
 

figure
plot(1: trees, oobError(net), 'b-', 'LineWidth', 1)
legend('Error')
xlabel('Trees')
ylabel('Error')
xlim([1, trees])
grid

% pearson
pearson1 = corrcoef(T_train,T_sim1');

% R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
 
% MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M;
 
% RMSE
RMSE1 = sqrt(mean((T_sim1' - T_train).^2));

%%  scatter 
density_2D = ksdensity([T_sim2,T_test'],[T_sim2,T_test']);
figure,
scatter(T_sim2,T_test',35,density_2D,'filled');
colorbar, caxis([0 50]),colormap(slanCL(536));
my_handle=colorbar;
my_handle.Title.String = 'number';
my_handle.Title.FontSize = 12;
grid on,box on,hold on
x=0:0.1:1;
y=x;
plot(y,x,'r--');
xlabel('Prediction');
ylabel('Validation');
xlim([0 0.6])
ylim([0 0.6])
title('RF model')

index1=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'shrublands'));
index2=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'tree'));
index3=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'grass'));
index4=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'crop'));
index5=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'cropandnatural'));
index6=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'bareland'));
Bias_landcover_all=[];
MAE_landcover_all=[];
R2_landcover_all=[];
MSE_landcover_all=[];
RMSE_landcover_all=[];
ubRMSE_landcover_all=[];
r_landcover_all=[];
p_landcover_all=[];
erros_landcover_all=[];
figure,
for landcover_index=[1,3,4,6]
    if landcover_index==1
        color=slanCL(687,1);
        landcover_uav=T_sim2(index1);
        landcover_hydrago=T_test(index1)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
     elseif landcover_index==2
        color=slanCL(687,2);
        landcover_uav=T_sim2(index2);
        landcover_hydrago=T_test(index2)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==3
        color=slanCL(687,3);
        landcover_uav=T_sim2(index3);
        landcover_hydrago=T_test(index3)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==4
        color=slanCL(687,4);
        landcover_uav=T_sim2(index4);
        landcover_hydrago=T_test(index4)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==6
        color=slanCL(687,6);
        landcover_uav=T_sim2(index6);
        landcover_hydrago=T_test(index6)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    end       
    % Bias
    Bias_landcover=mean(landcover_uav(~isnan(landcover_uav)) -  landcover_hydrago(~isnan(landcover_uav)));
    % MAE
    MAE_landcover=mean(abs(landcover_uav(~isnan(landcover_uav)) -  landcover_hydrago(~isnan(landcover_uav))));
    % R-squared
    R2_landcover = 1 - norm(landcover_hydrago(~isnan(landcover_uav)) - landcover_uav(~isnan(landcover_uav))) ^ 2 / norm(landcover_hydrago(~isnan(landcover_uav)) - mean(landcover_hydrago(~isnan(landcover_uav)))) ^ 2;
    % (Mean Squared Error, MSE)
    MSE_landcover = mean((landcover_uav(~isnan(landcover_uav)) - landcover_hydrago(~isnan(landcover_uav))).^2);
    % RMSE
    RMSE_landcover = sqrt(mean((landcover_uav(~isnan(landcover_uav)) -  landcover_hydrago(~isnan(landcover_uav))).^2));
    % ubRMSE
    smpre=landcover_uav(~isnan(landcover_uav));
    smobs=landcover_hydrago(~isnan(landcover_uav));
    ubRMSE_landcover=sqrt(mean(power(smpre-mean(smpre)-smobs+mean(smobs),2)));
    % r
    [r_raw_landcover,p]=corrcoef(landcover_uav(~isnan(landcover_uav)),landcover_hydrago(~isnan(landcover_uav)));
    if size(r_raw_landcover,1)>1
    r_landcover=r_raw_landcover(2);
    p_landcover=p(2);
    else
        r_landcover=nan;
        p_landcover=nan;
    end

    Bias_landcover_all=[Bias_landcover_all;Bias_landcover];
    MAE_landcover_all=[MAE_landcover_all;MAE_landcover];
    R2_landcover_all=[R2_landcover_all;R2_landcover];
    MSE_landcover_all=[MSE_landcover_all;MSE_landcover];
    RMSE_landcover_all=[RMSE_landcover_all;RMSE_landcover];
    ubRMSE_landcover_all=[ubRMSE_landcover_all;ubRMSE_landcover];
    r_landcover_all=[r_landcover_all;r_landcover];
    p_landcover_all=[p_landcover_all;p_landcover];
    erros_landcover_all=[MAE_landcover_all,RMSE_landcover_all,ubRMSE_landcover_all];
end
x=0:0.1:1;
y=x;
plot(y,x,'r--');
xlim([0 0.6])
ylim([0 0.6])
hold off
title("RF")
legend({'Shrubland','Grassland','Farmland','Bareland'},'Location','Northwest');
set(gca, 'LooseInset', [0,0,0,0]);

fig=figure;
Xbar=erros_landcover_all;
ax=gca;
hold on;
box on
yline(0.06,'--r','0.06');
bHdl=bar(Xbar,'LineWidth',.8);
CList=slanCL(955,1:4);
bHdl(1).FaceColor=CList(1,:);
bHdl(2).FaceColor=CList(3,:);
bHdl(3).FaceColor=CList(4,:);
ax.FontName='Times New Roman';
ax.LineWidth=.8;
ax.FontSize=12;
ax.YGrid='on';
ax.GridLineStyle='-.';
ax.XTick=1:23;
title('RF model');
ylabel("Error(m^3/m^3)");
legend('Standard','MAE','RMSE','ubRMSE','Orientation','horizontal','Location','Northeast');
set(ax,'xticklabels',{'Shrubland','Grassland','Farmland','Bareland'});
ylim([0 0.14]);
fig.Units = 'pixels';
fig.Position = [0, 0, 800, 600]; 
fig.Color = [1, 1, 1]; 
hold off
set(gca, 'LooseInset', [0,0,0,0]);
