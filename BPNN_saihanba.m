clc
clear
%%  20%，80%
seed=10;
rng(seed,"philox")
s = rng;

%% 2.import
[data,txt,raw]=xlsread('.\rf_dataset06_spatial_newparameter.xlsx');
res=data;
temp = randperm(length(res));
format long 
landuse=txt(:,12);
landuse=landuse(2:end);

input=data(:,[4:5,21,33,29:32]);   
output=data(:,8);       
%% 3.train test
input_train = input(temp(1:floor(length(res)*0.8)),:)';                   
output_train =output(temp(1:floor(length(res)*0.8)))';                   
input_test =input(temp((floor(length(res)*0.8)+1):end),:)';    
output_test =output(temp((floor(length(res)*0.8)+1):end))';    
%% 4.data normalization
[inputn,inputps]=mapminmax(input_train,0,1);         
[outputn,outputps]=mapminmax(output_train);          
inputn_test=mapminmax('apply',input_test,inputps);   
%% 5.Find the optimal hidden layer
inputnum=size(input,2);   
outputnum=size(output,2);
disp(['Number of input layer nodes：',num2str(inputnum),',  Number of output layer nodes：',num2str(outputnum)])
disp(['The range of hidden layer nodes is ',num2str(fix(sqrt(inputnum+outputnum))+1),' to ',num2str(fix(sqrt(inputnum+outputnum))+10)])
disp(' ')
disp('optimal hidden layer...')
 
MSE=1e+5;                             
transform_func={'tansig','purelin'};  
train_func='trainlm';                 
draw_mse0=[];
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    net=newff(inputn,outputn,hiddennum,transform_func,train_func); 
    
    % set parameters
    net.trainParam.epochs=1000;       
    net.trainParam.lr=0.01;           
    net.trainParam.goal=0.000001;    
    
    % train
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);     
    mse0=mse(outputn,an0);   
    disp(['When the number of hidden layer nodes is',num2str(hiddennum),' the mean square error of the training set is：',num2str(mse0)])
    
    %
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
    draw_mse0=[draw_mse0;mse0];
end
disp(['The optimal number of hidden layer nodes is：',num2str(hiddennum_best),'，MSE：',num2str(MSE)])
figure,
plot(fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10,draw_mse0);

%% 6.Constructing the BP neural network with the optimal hidden layer
net=newff(inputn,outputn,hiddennum_best,transform_func,train_func);

net.trainParam.epochs=1000;        
net.trainParam.lr=0.01;            
net.trainParam.goal=0.000001;      
%% 7.Train
net=train(net,inputn,outputn);     
%% 8.Test
an=sim(net,inputn_test);                     
test_simu=mapminmax('reverse',an,outputps); 
error=test_simu-output_test;                 

% 
W1 = net.iw{1, 1}; 
B1 = net.b{1};      
W2 = net.lw{2,1};   
B2 = net.b{2};      
save('BPNN.mat','net');


[~,len]=size(output_test);            
SSE1=sum(error.^2);                   
MAE1=sum(abs(error))/len;             
MSE1=error*error'/len;                
RMSE1=MSE1^(1/2);                     
MAPE1=mean(abs(error./output_test));  
r=corrcoef(output_test,test_simu);    
R1=r(1,2);    

T_test=output_test;
T_sim2=test_simu;
% R2
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;

disp(['SSE：',num2str(SSE1)])
disp(['MAE：',num2str(MAE1)])
disp(['MSE：',num2str(MSE1)])
disp(['RMSE：',num2str(RMSE1)])
disp(['MAPE：',num2str(MAPE1*100),'%'])
disp(['r： ',num2str(R1)])
disp(['R2：', num2str(R2)])


%% landcover
index1=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'shrublands'));
index2=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'tree'));
index3=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'grass'));
index4=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'crop'));
index5=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'cropandnatural'));
index6=find(strcmp(landuse(temp((floor(length(res)*0.8)+1):end)),'bareland'));

T_sim1=test_simu;
T_train=output_test;

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
        landcover_uav=T_sim1(index1)';
        landcover_hydrago=T_train(index1)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==2
        color=slanCL(687,2);
        landcover_uav=T_sim1(index2)';
        landcover_hydrago=T_train(index2)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==3
        color=slanCL(687,3);
        landcover_uav=T_sim1(index3)';
        landcover_hydrago=T_train(index3)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==4
        color=slanCL(687,4);
        landcover_uav=T_sim1(index4)';
        landcover_hydrago=T_train(index4)';
        scatter(landcover_uav,landcover_hydrago,35,color,'filled');
        grid on,box on,hold on
        xlabel('Prediction (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Validation (soil moisture in m^3/m^3)', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'LooseInset', [0,0,0,0]);
    elseif landcover_index==6
        color=slanCL(687,6);
        landcover_uav=T_sim1(index6)';
        landcover_hydrago=T_train(index6)';
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
title("ANN")
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
title('BPNN model');
ylabel("Error(m^3/m^3)");
legend('Standard','MAE','RMSE','ubRMSE','Orientation','horizontal','Location','Northeast');
set(ax,'xticklabels',{'Shrubland','Woodland','Grassland','Farmland','Bareland'});
ylim([0 0.14]);
fig.Units = 'pixels';
fig.Position = [0, 0, 800, 600]; 
fig.Color = [1, 1, 1]; 
hold off
set(gca, 'LooseInset', [0,0,0,0]);
