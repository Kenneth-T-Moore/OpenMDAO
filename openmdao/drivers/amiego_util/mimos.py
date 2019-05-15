from __future__ import print_function

import numpy as np

from openmdao.core.driver import Driver
from openmdao.drivers.genetic_algorithm_driver import GeneticAlgorithm


class MIMOS(Driver):
    """
    Class definition for the MIMOS psuedo-driver.

    Implements Multiple Infills via a Multi-Objective Strategy (MIMOS) as a plugin for
    the minlp slot on the AMIEGO driver.

    Attributes
    ----------
    dvs : list
        Cache of integer design variable names.
    """


"""
%% This function samples Multiple Infills via a Multi-Objective Strategy (MIMOS)
function [x_new, ei_min, eflag] = MIMOS3(lb,ub,ModelInfo_obj,intcon,prob_data)
    %% Find the non-dominated set using a MOEA
    [y_NDpt, x_NDpt] = findNDset(lb, ub, ModelInfo_obj, intcon);
    if ~isempty(y_NDpt)
        ei_min = min(y_NDpt(:,1));
        num_NDpt = size(y_NDpt,1);
        actual_pt2sam = min(prob_data.req_sam,num_NDpt);
%         fprintf('\n%s%d','Number ND points: ',num_NDpt)
        %% Create and select samples from clusters
        [x_new,eflag] = create_cluster(y_NDpt,x_NDpt,actual_pt2sam, ModelInfo_obj.X_org, ModelInfo_obj.y_org);
    else
        x_new=[];ei_min=[];eflag=-1;
    end

% keyboard
%% Sub functions
function [y_NDpt, x_NDpt] = findNDset(lb, ub, ModelInfo_obj, intcon)
    if length(intcon) < length(lb)
        %Continuous (Keep this option open for Bret's problem)
        lb_con=lb(length(intcon)+1:end)'; %Lower bounds for the continuous design variables
        ub_con=ub(length(intcon)+1:end)'; %Upper bound for the continuous design variables
        res = 0.1;
        bits_con=ceil(log2((ub_con-lb_con)/res + 1)); %Bits for continuous variable
    else
        lb_con=[]; ub_con=[]; bits_con=[];
    end
    %Discrete
    xI_lb = lb(intcon);
    xI_ub = ub(intcon);
    bits_dis=ceil(log2(xI_ub-xI_lb+1))';
    lb_dis = xI_lb';
    ub_dis = (xI_lb' + (2.^bits_dis - 1)); %% Needs additional work!! Artificially increase the bounds of the design var to accommodate discrete resolution in GA
    problem.ub_org = xI_ub;
    problem.ModelInfo_obj = ModelInfo_obj;

    [x_NDpt_raw,y_NDpt_raw] = NBGA3([],lb_con,ub_con,bits_con,...
        lb_dis,ub_dis,bits_dis,problem);

    % Excludes any point that violates the constraints (due to binary
    % coding)
    cc=[];
    for ii = 1:size(y_NDpt_raw,1)
        if ~isempty(find((problem.ub_org' - x_NDpt_raw(ii,intcon))<0,1))
            cc = [cc;ii];
        end
    end
    y_NDpt_raw(cc,:) = [];
    x_NDpt_raw(cc,:) = [];

    %% Check for no duplicacy within integer ND design space
    [~,~,ib] = union(ModelInfo_obj.X_org(:,intcon),x_NDpt_raw(:,intcon),'rows');
    ind_up = sort(ib);
    x_NDpt = x_NDpt_raw(ind_up,:);
    y_NDpt = y_NDpt_raw(ind_up,:);

function [x_new,eflag] = create_cluster(y_NDpt,x_NDpt,actual_pt2sam, exist_pt_x, exist_pt_y)
    %% Normalize the ND points
    [num_NDpt, num_obj] = size(y_NDpt);
    if num_NDpt<=1
        [~,dis] = knnsearch(exist_pt_x, x_NDpt, 'k',1');
        if dis>0
            eflag = 1;
            x_new = x_NDpt;
        else
            fprintf('\n%s','No new sample found!')
            eflag = 0;
        end
    else
        eflag = 1;
        ideal_pt = zeros(1,num_obj);
        worst_pt = zeros(1,num_obj);
        for ii = 1:num_obj
            ideal_pt(1,ii) = min(y_NDpt(:,ii));
            worst_pt(1,ii) = max(y_NDpt(:,ii));
        end
        norm_NDpt = (y_NDpt - repmat(ideal_pt,[num_NDpt,1]))./...
            (repmat(worst_pt,[num_NDpt,1]) - repmat(ideal_pt,[num_NDpt,1]));

        [clus, clus_cen] = kmeans(norm_NDpt,actual_pt2sam);
%         % Plot the clusters if needed
%         % % str = {'bd','go','rx','c+','m*','ks'};
%         str = {'b','g','r','c','m','k'};
%         for ii = 1:actual_pt2sam
%             aa = find(clus==ii);
%             switch num_obj
%                 case 2
%                     plot(norm_NDpt(aa,1),norm_NDpt(aa,2),[str{ii},'o']);hold on
%                     plot(clus_cen(:,1),clus_cen(:,2),[str{mod(ii,6)+1},'h'],'MarkerSize',15)
%                 case 3
%                     plot3(norm_NDpt(aa,1),norm_NDpt(aa,2),norm_NDpt(aa,3),[str{mod(ii,6)+1},'o']);hold on
%                     plot3(clus_cen(:,1),clus_cen(:,2),clus_cen(:,3),[str{mod(ii,6)+1},'h'],'MarkerSize',15)
%             end
%         end

        clus_cen_sorted = zeros(actual_pt2sam,num_obj,num_obj);
        Ii = zeros(actual_pt2sam,num_obj);
        for ii = 1:num_obj
            [~,Ii(:,ii)] = sort(clus_cen(:,ii),1);
            clus_cen_sorted(:,:,ii) = clus_cen(Ii(:,ii),:);
        end

        Ii_temp = Ii;
        num_sam = 0;
        clus_picked = zeros(actual_pt2sam,1);
        while num_sam < actual_pt2sam
            obj_picked = mod(num_sam,num_obj)+1;
            clus_picked(num_sam+1) = Ii_temp(1,obj_picked);
            Ii_new = [];
            for ii = 1:num_obj
                Ii_temp2 = Ii_temp(:,ii);
                Ii_temp2(Ii_temp2 == clus_picked(num_sam+1))=[];
                Ii_new = [Ii_new,Ii_temp2];
            end
            num_sam = num_sam+1;
            Ii_temp = Ii_new;
        end

        x_new = zeros(length(clus_picked), size(x_NDpt,2));
        ind=[];
        for  ii = 1:length(clus_picked)
           obj_picked = mod(ii-1,num_obj)+1;
           x_clus = x_NDpt(clus == clus_picked(ii),:);
           y_clus = y_NDpt(clus == clus_picked(ii),:);
           [x_new(ii,:),flg] = pickfromcluster1(x_clus, y_clus, obj_picked, exist_pt_x, exist_pt_y);
           if flg == 1
               ind = [ind;ii];
           end
        end
        x_new(ind,:) = [];
    end

function [x_new, flag] = pickfromcluster1(x_clus, y_clus, obj_picked, exist_pt_x, exist_pt_y)
    flag = 0;
    [~,dis] = knnsearch(exist_pt_x, x_clus, 'k',1');
    ind_posi_dis = find(dis>0);
    posi_dis = dis(ind_posi_dis);
    if obj_picked == 1 % Expected Improvement (Strategy: Balance)
        % Picks a point that satisfies: [min(EI) and dis>0]
        [~,ind] = min(y_clus(ind_posi_dis,1));
        x_new = x_clus(ind_posi_dis(ind),:);
    elseif obj_picked == 2 % Reduce void spaces (Strategy: pure exploration)
        % Picks a point that satisfies: max(dis)
        [~,ind] = max(posi_dis);
        x_new = x_clus(ind_posi_dis(ind),:);
    elseif obj_picked == 3 % Search around the best solution (Strategy: pure exploitation)
        % Picks a point that satisfies: closest to the best existing point
        [~,ind_ymin] = min(exist_pt_y);
        [~,dis2] = knnsearch(exist_pt_x(ind_ymin,:),x_clus,'k',1);
        ind_posi_dis2 = find(dis2>0);
        posi_dis2 = dis2(ind_posi_dis2);
        [~,ind2] = min(posi_dis2);
        x_new = x_clus(ind_posi_dis2(ind2),:);
    end
    if isempty(x_new)
        flag = 1;
        x_new=0*x_clus;
    end

%     %% Check distance from the best existing point
%     [~,ind_ymin] = min(exist_pt_y);
%    eucl_dis = norm(x_new - exist_pt_x(ind_ymin,:));
%    fprintf('\n%s%d%s%0.3f','Objective/distance: ',obj_picked,', ', eucl_dis)

function [x_new, flag] = pickfromcluster2(x_clus, ~, ~, exist_pt_x, ~)
    flag = 0;
    [~,dis] = knnsearch(exist_pt_x, x_clus, 'k',1');
    ind_posi_dis = find(dis>0);
    posi_dis = dis(ind_posi_dis);

    % Strategy: Picks point farthest for existing points
    [~,ind] = max(posi_dis);
    x_new = x_clus(ind_posi_dis(ind),:);

    if isempty(x_new)
        flag = 1;
        x_new=0*x_clus;
    end

function [x_new, flag] = pickfromcluster3(x_clus, y_clus, obj_picked, exist_pt_x, exist_pt_y)
    flag = 0;
    [idx_exist_pt,dis] = knnsearch(exist_pt_x, x_clus, 'k',1');
    ind_posi_dis = find(dis>0);
    posi_dis = dis(ind_posi_dis);
    if obj_picked == 1 || obj_picked == 2
        % Picks a point that satisfies: max(dis)
        [~,ind] = max(posi_dis);
        x_new = x_clus(ind_posi_dis(ind),:);
    elseif obj_picked == 3 % Search around the best solution (Strategy: pure exploitation)
        % Picks a point that satisfies: closest to the best existing point
        [~,ind_clus_ymin] = min(exist_pt_y(idx_exist_pt));
        [~,dis2] = knnsearch(exist_pt_x(idx_exist_pt(ind_clus_ymin),:),x_clus,'k',1);
        ind_posi_dis2 = find(dis2>0);
        posi_dis2 = dis2(ind_posi_dis2);
        [~,ind2] = min(posi_dis2);
        x_new = x_clus(ind_posi_dis2(ind2),:);
    end
    if isempty(x_new)
        flag = 1;
        x_new=0*x_clus;
    end
"""