clc
clear
[data,txt,raw]=xlsread('.\rf_dataset06_spatial_newparameter.xlsx');
i_index_all=data(:,1);
uav_lon_all=data(:,2);
uav_lat_all=data(:,3);
tb_v_all=data(:,4);
tb_h_all=data(:,5);
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
st=data(:,21);
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

%%  import
res = data;

%%  20%，80%
seed=10;
rng(seed,"philox")
s = rng;
temp = randperm(length(res));
%% train test
P_train = res(temp(1: floor(length(res)*0.8)), [4:5,21,33,29:32])';
T_train = res(temp(1: floor(length(res)*0.8)), 8)';
M = size(P_train, 2);

P_test = res(temp((floor(length(res)*0.8)+1): end), [4:5,21,33,29:32])';
T_test = res(temp((floor(length(res)*0.8)+1): end), 8)';
N = size(P_test, 2);
%% GLR
n=size(data,2);
x=data(:,[2:5,21,28:32]);  
y=data(:,8);
x1=x;
X=ones(size(x,1),1);
X=[X,x];
myalpha=0.05;
[b,bint,r,rint,stats] = regress(y(temp(1:floor(length(res)*0.8)),:),X(temp(1:floor(length(res)*0.8)),:),myalpha);
y_n1=b(1)+X(temp(1:floor(length(res)*0.8)),2:end)*b(2:end);
y_n2=b(1)+X(temp((floor(length(res)*0.8)+1):end),2:end)*b(2:end);

T_train=y(temp(1:floor(length(res)*0.8)));
T_sim1=y_n1;
T_test=y(temp((floor(length(res)*0.8)+1):end));
T_sim2=y_n2;
% pearson
pearson1 = corrcoef(T_train,T_sim1);
pearson2 = corrcoef(T_test,T_sim2);


% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;


% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M;
mae2 = sum(abs(T_sim2 - T_test )) ./ N;


% RMSE
RMSE1 = sqrt(mean((T_sim1 - T_train).^2));
RMSE2 = sqrt(mean((T_sim2 - T_test).^2));

%% landcover
index1=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'shrublands'));
index2=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'tree'));
index3=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'grass'));
index4=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'crop'));
index5=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'cropandnatural'));
index6=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'bareland'));

T_sim1=y_n2';
T_train=y(temp((floor(length(res)*0.8)+1):end),:)';

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
        landcover_uav=T_sim1(index1);
        landcover_hydrago=T_train(index1);
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==2
        color=slanCL(687,2);
        landcover_uav=T_sim1(index2);
        landcover_hydrago=T_train(index2);
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==3
        color=slanCL(687,3);
        landcover_uav=T_sim1(index3);
        landcover_hydrago=T_train(index3);
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==4
        color=slanCL(687,4);
        landcover_uav=T_sim1(index4);
        landcover_hydrago=T_train(index4);
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==6
        color=slanCL(687,6);
        landcover_uav=T_sim1(index6);
        landcover_hydrago=T_train(index6);
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
xxx=0:0.1:1;
yyy=xxx;
plot(yyy,xxx,'r--');
xlim([0 0.6])
ylim([0 0.6])
hold off
title("GLR")
legend({'Shrubland','Woodland','Grassland','Farmland','Bareland'},'Location','Northwest');
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
title('GLR model');
ylabel("Error(m^3/m^3)");
legend('Standard','MAE','RMSE','ubRMSE','Orientation','horizontal','Location','Northeast');
set(ax,'xticklabels',{'Shrubland','Woodland','Grassland','Farmland','Bareland'});
ylim([0 0.14]);
fig.Units = 'pixels';
fig.Position = [0, 0, 800, 600];
fig.Color = [1, 1, 1]; 
hold off
set(gca, 'LooseInset', [0,0,0,0]);
