% % % % % % % % % % clear all; close all; clc;

%SaliencyMap Path
% SalMapPath = '../SalMap/Image_SOD/';  %The saliency map results can be downloaded from our webpage: http://dpfan.net/d3netbenchmark/
SalMapPath = '../SalMap/';  

%Evaluated Models
Models = {'MyNet'};  % SEINet-EfficientNetB7, SEINet-VGG16-attributes-based, SEINet-ResNet50-attributes-based

%Datasets
DataPath = 'D:/RSI_Evaluation/Dataset/';
Datasets = {'ORSSD'};

%%

%Evaluated Score Results
ResDir = '../Result_overall/';

%Initial paramters setting
Thresholds = 1:-1/255:0;
datasetNum = length(Datasets);
modelNum = length(Models);

for d = 1:datasetNum
    
    tic;
    dataset = Datasets{d};
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    ResPath = [ResDir dataset '-mat/'];
    if ~exist(ResPath,'dir')
        mkdir(ResPath);
    end
    resTxt = [ResDir dataset '_result-overall.txt'];
    fileID = fopen(resTxt,'w');
    
    for m = 1:modelNum
        model = Models{m};
        
        gtPath = [DataPath dataset '/GT/'];
                
        salPath = [SalMapPath model '/' dataset '/'];
        
        imgFiles = dir([salPath '*.png']);
        imgNUM = length(imgFiles);
        
        [threshold_Fmeasure, threshold_Emeasure] = deal(zeros(imgNUM,length(Thresholds)));
        
        [threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));
        
        [Smeasure, adpFmeasure, adpEmeasure, MAE, weiFm] =deal(zeros(1,imgNUM));
        
        parfor i = 1:imgNUM  %parfor i = 1:imgNUM  You may also need the parallel strategy. 
            
            fprintf('Evaluating(%s Dataset,%s Model): %d/%d\n',dataset, model, i,imgNUM);
            name =  imgFiles(i).name;
            
            %load gt
            gt = imread([gtPath name]);
            
            if (ndims(gt)>2)
                gt = rgb2gray(gt);
            end
            
            if ~islogical(gt)
                gt = gt(:,:,1) > 128;
            end
            
            %load salency
            sal  = imread([salPath name]);
            
            %check size
            if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
                sal = imresize(sal,size(gt));
                imwrite(sal,[salPath name]);
                fprintf('Error occurs in the path: %s!!!\n', [salPath name]); %check whether the size of the salmap is equal the gt map.
            end
            
            sal = im2double(sal(:,:,1));
            
            %normalize sal to [0, 1]
            sal = reshape(mapminmax(sal(:)',0,1),size(sal));
            Sscore = StructureMeasure(sal,logical(gt));
            Smeasure(i) = Sscore;

%             [weiFm(i)]= WFb(sal,gt);
            [weiFm(i)]= 0;
            
            % Using the 2 times of average of sal map as the adaptive threshold.
            threshold =  2* mean(sal(:)) ;
            [~,~,adpFmeasure(i)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);
            
            
            Bi_sal = zeros(size(sal));
            Bi_sal(sal>threshold)=1;
            adpEmeasure(i) = Enhancedmeasure(Bi_sal,gt);
            
            [threshold_F, threshold_E]  = deal(zeros(1,length(Thresholds)));
            [threshold_Pr, threshold_Rec]  = deal(zeros(1,length(Thresholds)));
            
            for t = 1:length(Thresholds)
                threshold = Thresholds(t);
                [threshold_Pr(t), threshold_Rec(t), threshold_F(t)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);
                
                Bi_sal = zeros(size(sal));
                Bi_sal(sal>threshold)=1;
                threshold_E(t) = Enhancedmeasure(Bi_sal,gt);
            end
            
            threshold_Fmeasure(i,:) = threshold_F;
            threshold_Emeasure(i,:) = threshold_E;
            threshold_Precion(i,:) = threshold_Pr;
            threshold_Recall(i,:) = threshold_Rec;
            
            MAE(i) = mean2(abs(double(logical(gt)) - sal));
            
        end
        
        %Precision and Recall 
        column_Pr = mean(threshold_Precion,1);
        column_Rec = mean(threshold_Recall,1);
        
        %Mean, Max F-measure score
        column_F = mean(threshold_Fmeasure,1);
        meanFm = mean(column_F);
        maxFm = max(column_F);
        
        %Mean, Max E-measure score
        column_E = mean(threshold_Emeasure,1);
        meanEm = mean(column_E);
        maxEm = max(column_E);
        
        %Adaptive threshold for F-measure and E-measure score
        adpFm = mean2(adpFmeasure);
        adpEm = mean2(adpEmeasure);
        
        %Smeasure score
        Smeasure = mean2(Smeasure);
        
        %MAE score
        mae = mean2(MAE);

        %WeiFm score
        WeiFm = mean2(weiFm);
        
        %Save the mat file so that you can reload the mat file and plot the PR Curve
        save([ResPath model],'Smeasure', 'mae', 'column_Pr', 'column_Rec', 'column_F', 'adpFm', 'meanFm', 'maxFm', 'column_E', 'adpEm', 'meanEm', 'maxEm');
       
        fprintf(fileID, '(Dataset:%s; Model:%s) Smeasure:%.4f; MAE:%.4f; adpEm:%.4f; meanEm:%.4f; maxEm:%.4f; adpFm:%.4f; meanFm:%.4f; maxFm:%.4f.\n',dataset,model,Smeasure, mae, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm);   
    end
    toc;
    
end


