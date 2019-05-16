"""
Class definition for the MIMOS psuedo-driver.

Implements Multiple Infills via a Multi-Objective Strategy (MIMOS) as a plugin for
the minlp slot on the AMIEGO driver.

Developed by Satadru Roy
School of Aeronautics & Astronautics
Purdue University, West Lafayette, IN 47906
2018~9
Implemented in OpenMDAO, May 2019, Kenneth T. Moore
"""
from __future__ import print_function
from six import iteritems

from math import factorial
import os

import numpy as np

from openmdao.core.driver import Driver
from openmdao.drivers.genetic_algorithm_driver import GeneticAlgorithm


class MIMOS(Driver):
    """
    Class definition for the MIMOS psuedo-driver.

    Implements Multiple Infills via a Multi-Objective Strategy (MIMOS) as a plugin for
    the minlp slot on the AMIEGO driver.

    This will not work as a standalone driver.

    Attributes
    ----------
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    dvs : list
        Cache of integer design variable names.
    eflag_MINLPBB : bool
        This is set to True when we find a local minimum.
    fopt : ndarray
        Objective value with the maximum expected improvement.
    xI_lb : ndarray
        Lower bound of the integer design variables.
    xI_ub : ndarray
        Upper bound of the integer design variables.
    xopt : ndarray
        List of new infill points from optimizing exploitation and exploration.
    _randomstate : np.random.RandomState, int
         Random state (or seed-number) which controls the seed and random draws.
    """

    def __init__(self):
        """
        Initialize the MIMOS driver.
        """
        super(MIMOS, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['integer_design_vars'] = True
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = False

        # Options
        self.options.declare('required_samples', 3,
                             desc='Number of infill points.')
        self.options.declare('bits', default={}, types=(dict),
                             desc='Number of bits of resolution. Default is an empty dict, where '
                             'every unspecified variable is assumed to be integer, and the number '
                             'of bits is calculated automatically. If you have a continuous var, '
                             'you should set a bits value as a key in this dictionary.')
        self.options.declare('disp', True,
                             desc='Set to False to prevent printing of iteration '
                             'messages.')
        self.options.declare('max_gen', default=100,
                             desc='Number of generations before termination.')
        self.options.declare('pop_size', default=0,
                             desc='Number of points in the GA. Set to 0 and it will be computed '
                             'as four times the number of bits.')
        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute the points in a generation in parallel.')
        self.options.declare('Pc', default=0.5, lower=0., upper=1.,
                             desc='Crossover rate.')
        self.options.declare('Pm',
                             desc='Mutation rate.', default=None, lower=0., upper=1.,
                             allow_none=True)

        self.dvs = []
        self.idx_cache = {}
        self._ga = None

        # We will set this to True if we have found a minimum.
        self.eflag_MINLPBB = False

        # random state can be set for predictability during testing
        if 'SimpleGADriver_seed' in os.environ:
            self._randomstate = int(os.environ['SimpleGADriver_seed'])
        else:
            self._randomstate = None

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super(MIMOS, self)._setup_driver(problem)

        # Size our design variables.
        j = 0
        for name, val in iteritems(self.get_design_var_values()):
            self.dvs.append(name)
            if name in self._designvars_discrete:
                if np.isscalar(val):
                    size = 1
                else:
                    size = len(val)
            else:
                size = len(val)
            self.idx_cache[name] = (j, j + size)
            j += size

        # Lower and Upper bounds
        self.xI_lb = np.empty((j))
        self.xI_ub = np.empty((j))
        dv_dict = self._designvars
        for var in self.dvs:
            i, j = self.idx_cache[var]
            self.xI_lb[i:j] = dv_dict[var]['lower']
            self.xI_ub[i:j] = dv_dict[var]['upper']

        model_mpi = None
        comm = self._problem.comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options['run_parallel']:
            comm = None

        self._ga = GeneticAlgorithm(self.objective_callback, comm=comm, model_mpi=model_mpi)

    def run(self):
        """
        Execute the MIMOS method.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False if successful.
        """
        self.eflag_MINLPBB = False

        x_nd, y_nd = self.find_nondominated_set()

        if len(x_nd) > 0:
            ei_min = min(y_nd[:, 0])
            num_nd = y_nd.shape[0]

            actual_pt2sam = min(self.options['required_samples'], num_nd)

            # Create and select samples from clusters.
            x_new, eflag = self.create_cluster(y_nd, x_nd, actual_pt2sam)

        else:
            x_new = []
            ei_min = None
            eflag = True

        # Save the new infill points for AMIEGO to retrieve.
        self.xopt = x_new
        self.fopt = ei_min
        self.eflag_MINLPBB = eflag

        return False

    def find_nondominated_set(self):
        """
        Compute a non-dominated set of candidate points.

        This is computed via a genetic algorithm that returns a set of points that are pareto
        optimal for maximum expected improvement (exploration) and maximum distance from existing
        points.

        Returns
        -------
        ndarray
            Non dominated design points.
        ndarray
            Objective values at non dominated points.
        """
        model = self._problem.model
        ga = self._ga
        ga.nobj = 2

        pop_size = self.options['pop_size']
        max_gen = self.options['max_gen']
        user_bits = self.options['bits']
        Pm = self.options['Pm']  # if None, it will be calculated in execute_ga()
        Pc = self.options['Pc']

        count = self.xI_lb.shape[0]
        lower_bound = self.xI_lb.copy()
        upper_bound = self.xI_ub.copy()
        outer_bound = np.full((count, ), np.inf)

        bits = np.empty((count, ), dtype=np.int)

        # Figure out initial design vars.
        desvars = self._designvars
        desvar_vals = self.get_design_var_values()
        x0 = np.empty(count)
        for name, meta in iteritems(desvars):
            i, j = self.idx_cache[name]
            x0[i:j] = desvar_vals[name]

        # Bits of resolution
        abs2prom = model._var_abs2prom['output']

        for name, meta in iteritems(desvars):
            i, j = self.idx_cache[name]

            if name in self._designvars_discrete:
                prom_name = name
            else:
                prom_name = abs2prom[name]

            if name in user_bits:
                val = user_bits[name]

            elif prom_name in user_bits:
                val = user_bits[prom_name]

            else:
                # If the user does not declare a bits for this variable, we assume they want it to
                # be encoded as an integer. Encoding requires a power of 2 in the range, so we need
                # to pad additional values above the upper range, and adjust accordingly. Design
                # points with values above the upper bound will be discarded by the GA.
                log_range = np.log2(upper_bound[i:j] - lower_bound[i:j] + 1)
                val = log_range  # default case -- no padding required
                mask = log_range % 2 > 0  # mask for vars requiring padding
                val[mask] = np.ceil(log_range[mask])
                outer_bound[i:j][mask] = upper_bound[i:j][mask]
                upper_bound[i:j][mask] = 2**np.ceil(log_range[mask]) - 1 + lower_bound[i:j][mask]

            bits[i:j] = val

        # Automatic population size.
        if pop_size == 0:
            pop_size = 4 * np.sum(bits)

        desvar_new, opt, nfit = ga.execute_ga(x0, lower_bound, upper_bound, outer_bound,
                                              bits, pop_size, max_gen,
                                              self._randomstate, Pm, Pc)

        return desvar_new, opt

    def objective_callback(self, x, icase):
        """
        Evaluate problem objectives at the requested point.

        Parameters
        ----------
        x : ndarray
            Value of design variables.
        icase : int
            Case number, used for identification when run in parallel.

        Returns
        -------
        ndarray
            Objective values
        bool
            Success flag, True if successful
        int
            Case number, used for identification when run in parallel.
        """
        obj_surrogate = self.obj_surrogate

        # Objective 1: Expected Improvement
        # (Normalized as per the convention in openmdao_Alpha:Kriging.)
        xval = (x - obj_surrogate.X_mean) / obj_surrogate.X_std

        NegEI1 = calc_genEI_norm(xval, obj_surrogate, 1)

        #function f = objfunc_MOGA(x_con,x_dis,problem)
        #x = [x_dis,x_con]; % 1 x k
        #% if size(x,2)<2; x = x'; end
        #%% Objective 1: Expected Improvement
        #xval = ((x - problem.ModelInfo_obj.X_mean)./problem.ModelInfo_obj.X_std)';
        #Neg_EI1 = calc_genEI_norm(xval,problem.ModelInfo_obj,1);
        #Neg_EI0 = calc_genEI_norm(xval,problem.ModelInfo_obj,0);
        #%% Objective 2: Maximize distance from any existing points
        #[~,dis] = knnsearch(problem.ModelInfo_obj.X_org, x, 'k',1);
        #f = [Neg_EI1,-1*dis,Neg_EI0];


def norm_pdf(x, mu=0.0, sigma=1.0):
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 * 0.5 * sigma**2)


def calc_genEI_norm(xval, obj_surrogate, gg):
    """
    Compute the generalized expected improvement.

    Parameters
    ----------
    xval : ndarray
        Value of the current integer design variables.
    obj_surrogate : <AMIEGOKrigingSurrogate>
        Surrogate model of optimized objective with respect to integer design variables.
    gg : float
        Exponent used in generalized expected improvement.

    Returns
    -------
    float
        The generalized expected improvement evaluated at xval.
    """
    y_min = (obj_surrogate.best_obj_norm - obj_surrogate.Y_mean) / obj_surrogate.Y_std

    X = obj_surrogate.X
    Y = obj_surrogate.Y
    c_r = obj_surrogate.c_r
    thetas = obj_surrogate.thetas
    SigmaSqr = obj_surrogate.SigmaSqr
    R_inv = obj_surrogate.R_inv
    mu = obj_surrogate.mu
    p = 2

    r = np.exp(-np.einsum("ij->i", thetas.T * (xval - X)**p))

    # Calculate prediction and error.
    y_hat = mu + np.dot(r, c_r)
    term0 = np.dot(R_inv, r)

    SSqr = SigmaSqr * (1.0 - r.dot(term0) +
                       (1.0 - np.einsum('i->', term0))**2 / np.einsum('ij->', R_inv))

    if SSqr <= 1.0e-30:
        if abs(SSqr) <= 1.0e-30:
            Neg_genEI = -0.0

    else:
        # Calculate the generalized expected improvement function
        sqrt_SSqr = np.sqrt(SSqr)
        z = (y_min - y_hat) / sqrt_SSqr
        sg = sqrt_SSqr ** gg

        phi_s = norm_pdf(z)
        phi_C = np.cumsum(phi_s)

        T_k = np.zeros((gg + 1, ))
        T_k[0] = phi_C
        T_k[1] = -phi_s
        SS = 0;
        for kk in range(gg):
            if kk >= 1:
                T_k[kk] = -phi_s * z**(kk - 2) + (kk - 2)*T_k[kk - 2]
            SS += ((-1)**kk) * (factorial(gg) / (factorial(kk) * factorial(gg - kk))) * \
                  (z**(gg - kk)) * T_k[kk + 1]

        Neg_genEI = -sg * SS

    return Neg_genEI


"""
for i=1:n
    r(i,1)=exp(-sum(theta'.*(x-X(i,:)).^p));
end
%Calculate prediction and error
y_hat = mu + r'*(R_inv*(y-one*mu));
SSqr = SigmaSqr*(1 - (r'*(R_inv*r)) + ((1-one'*(R_inv*r))^2/(one'*(R_inv*one))));

if SSqr <= 1e-30
    Neg_genEI = -0.0;
else
    %Calculate the generalized expected improvement function
    z = (y_min - y_hat)/sqrt(SSqr);
    sg = (sqrt(SSqr))^gg;

    phi_C = normcdf(z);
    phi_s = normpdf(z);

    T_k = zeros(gg+1,1); T_k(1) = phi_C; T_k(2) = -phi_s;
    SS = 0;
    for kk = 0:gg
        if kk>=2
            T_k(kk+1) = -phi_s*z^(kk-1) + (kk-1)*T_k(kk-2+1);
        end
        SS=SS+((-1)^kk)*(factorial(gg)/(factorial(kk)*factorial(gg-kk)))*(z^(gg-kk))*T_k(kk+1);
    end
    Neg_genEI = -sg*SS; % + y_hat;
    if isnan(Neg_genEI)
        keyboard
    end
end
"""

"""
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