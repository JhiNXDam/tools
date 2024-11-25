clear all; close all; clc;

%SaliencyMap Path
SalMapPath = '../SalMap/';
%Models = {'LHM','DESM','CDB','ACSD','GP','LBE','CDCP','SE','MDSF','DF','CTMF','PDNet','PCF'};%'TPF'
%Models = {'PCF','PDNet','CTMF','DF','MDSF','SE','CDCP','LBE','GP','ACSD','CDB','DESM','LHM'};
%Models = {'SSAV'};
Models = {'MSAAFNet_54', 'SARNet', 'DAFNet', 'MJRBM', 'ERPNet', 'RRNet', 'EMFINet', 'MSCNet', 'SeaNet', 'MEANet'}; % 'AEINet', 'LVNet', 'SARNet', 'DAFNet', 'AGNet', 'RRNet', 'ACCoNet', 'MCCNet', 'MJRBM', 'ERPNet', 'EMFINet'
%Deep Model
%DeepModels = {'DF','PDNet','CTMF','PCF'};%TPF-missing, MDSF-svm,

modelNum = length(Models);
groupNum = floor(modelNum/3)+1;

%Datasets
DataPath = '../Dataset/';
%Datasets = {'GIT'};
%Datasets = {'STERE','DES','NLPR','SSD','LFSD','GIT'};%'NJU2K',
%Datasets = {'GIT','SSD','DES'};
%Datasets = {'SSD','DES','LFSD','STERE','NJU2K','GIT','NLPR','SIP'};
% Datasets = {'SSD','DES','LFSD','STERE'};
Datasets = {'ORSSD'};

%Results
ResDir = 'E:/MatlabEvaluationTools-main/Result_overall/';

method_colors = linspecer(groupNum);

% colors
%str=['r','r','r','g','g','g','b','b','b','c','c','c','m','m','m','y','y','y','k','k','k','g','g','b','b','m','m','k','k','r','r','b','b','c','c','m','m'];

str=['r','g','b','y','c','m','k','r','g','b','y','c','m','k'];
%str=['r','g','b','y','c','m','k','r','g','b','y','c','m','k'];


datasetNum = length(Datasets);
for d = 1:datasetNum
    close all;
    dataset = Datasets{d};
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    matPath = [ResDir dataset '-mat/'];
    plotMetrics     = gather_the_results(modelNum, matPath, Models);
    
    %% plot the PR curves
    figure(1);
    hold on;
    grid on;
    axis([0 1 0.89 0.97]);
    title(dataset);
    xlabel('Recall');
    ylabel('Precision');
    
    for i = 1 : length(plotMetrics.Alg_names)
        if i==1
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), 'Color', 'r', 'LineWidth', 3);   
        elseif i==2
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '--','Color', 'm', 'LineWidth', 2); 
        elseif i==3
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '--','Color', 'b', 'LineWidth', 2);
        elseif i==4
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i),'--','Color', 'y', 'LineWidth', 2);
        elseif i==5
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '--','Color', 'c', 'LineWidth', 2);
        elseif i==6
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '--','Color', 'g', 'LineWidth', 2);
        elseif i==7
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '--','Color', 'k', 'LineWidth', 2);
        elseif i==8
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '--','Color', [0 0.4470 0.7410], 'LineWidth', 2);
        elseif i==9
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '-.', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2); 
        elseif i==10
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '-.', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2);
        end
        [~,max_idx] = max(plotMetrics.Fmeasure_Curve(:,i));
        h1 = plot(plotMetrics.Recall(max_idx,i), plotMetrics.Pre(max_idx,i));
        set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    end
    
%     for i = 1 : length(plotMetrics.Alg_names)
%         if mod(i,2)==0
%             plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '--', 'Color', str(i), 'LineWidth', 2);   
%         elseif mod(i,2)==1
%             plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), 'Color', str(i), 'LineWidth', 2);
%         end
%         [~,max_idx] = max(plotMetrics.Fmeasure_Curve(:,i));
%         h1 = plot(plotMetrics.Recall(max_idx,i), plotMetrics.Pre(max_idx,i), '.', 'Color', str(i),'markersize',20);
%         set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
%     end
    legend(plotMetrics.Alg_names,'Location','SouthWest');
    set(gcf,'position',[0 600 560 420]);
    
    figPath = [ResDir 'Curve/'];
    if ~exist(figPath,'dir')
        mkdir(figPath);
    end
    saveas(gcf, [figPath dataset '_PRCurve.svg'] );
    saveas(gcf, [figPath dataset '_PRCurve.png'] );
    
    %% plot the F-measure curves
    figure(2);
    hold on;
    grid on;
    axis([0 255 0.7 0.95]);
    title(dataset);
    xlabel('Threshold');
    ylabel('F-measure');
    x = [255:-1:0]';
    for i = 1 : length(plotMetrics.Alg_names)
        if i==1
            plot(x, plotMetrics.Fmeasure_Curve(:,i), 'Color', 'r', 'LineWidth', 3);
        elseif i==2
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '--','Color', 'm', 'LineWidth', 2);
        elseif i==3
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '--','Color', 'b', 'LineWidth', 2); 
        elseif i==4
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '--','Color', 'y', 'LineWidth', 2);
        elseif i==5
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '--','Color', 'c', 'LineWidth', 2);
        elseif i==6
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '--','Color', 'g', 'LineWidth', 2);
        elseif i==7
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '--','Color', 'k', 'LineWidth', 2);
        elseif i==8
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '--','Color', [0 0.4470 0.7410], 'LineWidth', 2);
        elseif i==9
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '-.', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2);
        elseif i==10
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '-.', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2);
        end
        
        [maxF,max_idx] = max(plotMetrics.Fmeasure_Curve(:,i));
        h2 = plot(255-max_idx, maxF);
        set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    end
    legend(plotMetrics.Alg_names,'Location','SouthWest');
    %legend(plotMetrics.Alg_names);
    set(gcf,'position',[1160 600 560 420]);
    saveas(gcf, [figPath dataset '_FmCurve.svg']);
    saveas(gcf, [figPath dataset '_FmCurve.png'] );
end

