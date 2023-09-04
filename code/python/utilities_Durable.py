import numpy as np
import numba
import ipywidgets as widgets
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import sparse as sp

from HARK.interpolation import LinearInterp, ValueFuncCRRA, MargValueFuncCRRA, CubicInterp
from HARK.econforgeinterp import LinearFast

# Additional:
from HARK.ConsumptionSaving.ConsIndShockModel import utility, utility_inv, utilityP_inv
from HARK.utilities import make_grid_exp_mult
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope, calc_linear_crossing

# From consav
from consav import linear_interp
from consav.grids import nonlinspace

# FUES
from FUES import FUES

# numba stuff
from numba import njit, float64, int32

import scipy.optimize as sp_optimize


### Additional Functions we need:
def func_nopar(c,d,d_ubar,alpha,rho): # U(C,D)
    dtot = d+d_ubar
    c_total = c**alpha*dtot**(1.0-alpha)
    return c_total**(1-rho)/(1-rho)

def create(ufunc, use_inv_w=False):
    """ create upperenvelope function from the utility function ufunc

    Args:

        ufunc (callable): utility function with *args (must be decorated with @njit)

    Returns:

        upperenvelope (callable): upperenvelope called as (grid_a,m_vec,c_vec,inv_w_vec,use_inv_w,grid_m,c_ast_vec,v_ast_vec,*args)
        use_inv_w (bool,optional): assume that the post decision value-of-choice vector is a negative inverse

    """

   # @numba.njit
    def upperenvelope(grid_a, m_vec, c_vec, inv_w_vec, grid_m, c_ast_vec, v_ast_vec, n, d_ubar, alpha, rho): # *args):
        """ upperenvelope function

        Args:

            grid_a (numpy.ndarray): input, end-of-period asset vector of length Na
            m_vec (numpy.ndarray): input, cash-on-hand vector from egm of length Na
            c_vec (numpy.ndarray): input, consumption vector from egm of length Na
            inv_w_vec (numpy.ndarray): input, post decision value-of-choice vector from egm of length Na
            grid_m (numpy.ndarray): input, common grid for cash-on-hand of length Nm
            c_ast_vec (numpy.ndarray): output, consumption on common grid for cash-on-hand of length Nm
            v_ast_vec (numpy.ndarray): output, value-of-choice on common grid for cash-on-hand of length Nm
            *args: additional arguments to the utility function

        """

        # for given m_vec, c_vec and w_vec (coming from grid_a)
        # find the optimal consumption choices (c_ast_vec) at the common grid (grid_m)
        # using the upper envelope + also value the implied values-of-choice (v_ast_vec)

        Na = grid_a.size
        Nm = grid_m.size

        c_ast_vec[:] = 0
        v_ast_vec[:] = -np.inf

        # constraint
        # the constraint is binding if the common m is smaller
        # than the smallest m implied by EGM step (m_vec[0])

        im = 0
        while im < Nm and grid_m[im] <= m_vec[0]:

            # a. consume all
            c_ast_vec[im] = grid_m[im]

            # b. value of choice
            u = ufunc(c_ast_vec[im], n, d_ubar, alpha, rho)
            if use_inv_w:
                v_ast_vec[im] = u + (-1.0 / inv_w_vec[0])
            else:
                v_ast_vec[im] = u + inv_w_vec[0]

            im += 1

        # upper envelope
        # apply the upper envelope algorithm

        for ia in range(Na - 1):

            # a. a inteval and w slope
            a_low = grid_a[ia]
            a_high = grid_a[ia + 1]

            inv_w_low = inv_w_vec[ia]
            inv_w_high = inv_w_vec[ia + 1]

            if a_low > a_high:
                continue

            inv_w_slope = (inv_w_high - inv_w_low) / (a_high - a_low)

            # b. m inteval and c slope
            m_low = m_vec[ia]
            m_high = m_vec[ia + 1]

            c_low = c_vec[ia]
            c_high = c_vec[ia + 1]

            c_slope = (c_high - c_low) / (m_high - m_low)

            # c. loop through common grid
            for im in range(Nm):

                # i. current m
                m = grid_m[im]

                # ii. interpolate?
                interp = (m >= m_low) and (m <= m_high)
                extrap_above = ia == Na - 2 and m > m_vec[Na - 1]

                # iii. interpolation (or extrapolation)
                if interp or extrap_above:

                    # o. implied guess
                    c_guess = c_low + c_slope * (m - m_low)
                    a_guess = m - c_guess

                    # oo. implied post-decision value function
                    inv_w = inv_w_low + inv_w_slope * (a_guess - a_low)

                    # ooo. value-of-choice
                    u = ufunc(c_guess, n, d_ubar, alpha, rho)
                    if use_inv_w:
                        v_guess = u + (-1 / inv_w)
                    else:
                        v_guess = u + inv_w

                    # oooo. update
                    if v_guess > v_ast_vec[im]:
                        v_ast_vec[im] = v_guess
                        c_ast_vec[im] = c_guess

    return upperenvelope

negm_upperenvelope = create(func_nopar, use_inv_w=True)


def obj_last_period(d, x, d_ubar,alpha,rho):
    """ objective function in last period """

    # implied consumption (rest)
    c = x - d

    return -func_nopar(c, d, d_ubar,alpha,rho)

def obj_adj(d, x, inv_v_keep, grid_d, grid_m):
    """ evaluate bellman equation """

    # a. cash-on-hand
    m = x - d

    # b. durables
    n = d

    # c. value-of-choice
    return -linear_interp.interp_2d(grid_d, grid_m, inv_v_keep, n, m)  # we are minimizing

#@numba.njit
def optimizer(obj,a,b,args=(),tol = 1e-6): # making tolerance smaller doesn't change anything
    """ golden section search optimizer

    Args:

        obj (callable): 1d function to optimize over
        a (double): minimum of starting bracket
        b (double): maximum of starting bracket
        args (tuple): additional arguments to the objective function
        tol (double,optional): tolerance

    Returns:

        (float): optimization result

    """

    inv_phi = (np.sqrt(5) - 1) / 2 # 1/phi
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2

    # a. distance
    dist = b - a
    if dist <= tol:
        return (a+b)/2

    # b. number of iterations
    n = int(np.ceil(np.log(tol/dist)/np.log(inv_phi)))

    # c. potential new mid-points
    c = a + inv_phi_sq * dist
    d = a + inv_phi * dist
    yc = obj(c,*args)
    yd = obj(d,*args)

    # d. loop
    for _ in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            dist = inv_phi*dist
            c = a + inv_phi_sq * dist
            yc = obj(c,*args)
        else:
            a = c
            c = d
            yc = yd
            dist = inv_phi*dist
            d = a + inv_phi * dist
            yd = obj(d,*args)

    # e. return
    if yc < yd:
        return (a+d)/2
    else:
        return (c+b)/2

#@numba.njit
def construct_grid(Min, Max, Count, grid_type, NestFac):
    """
    Based on: construct_assets_grid

    Constructs grids in different grid-types.
    Options are: linear or multi-exponentially

    Parameters
    ----------
    Min:                  float
        Minimum value for the grid
    Max:                  float
        Maximum value for the grid
    Count:                 int
        Size of the grid
    grid_type:
        linear
        exp_mult
        nonlinear
        exp_mult_extended
    # aXtraExtra:                [float]
    #     Extra values for the a-grid.
    # exp_nest:               int
    #     Level of nesting for the exponentially spaced grid

    Returns
    -------
    Grid:     np.ndarray
        Base array of values for the post-decision-state grid.
    """
    # Unpack the parameters
    Min = Min
    Max = Max
    Count = Count
    # aXtraExtra = parameters.aXtraExtra
    grid_type = grid_type
    exp_nest = NestFac

    # Set up post decision state grid:
    Grid = None
    if grid_type == "linear":
        Grid = np.linspace(Min, Max, Count)
    elif grid_type == "exp_mult":
        Grid = make_grid_exp_mult(
            ming=Min, maxg=Max, ng=Count, timestonest=exp_nest
        )
    elif grid_type == "nonlinear":
        Grid = nonlinspace(Min, Max, Count, 1.1)
    elif grid_type == "exp_mult_extended":
        Count = np.int(Count/2)
        Grid1 = make_grid_exp_mult(
            ming=Min, maxg=Max, ng=Count, timestonest=exp_nest
        )
        Grid2 = np.linspace(Min, Max, Count)
        Grid = np.unique(np.concatenate((Grid1, Grid2[:-1])))
    else:
        raise Exception(
            "grid_type not recognized in __init__."
            + "Please ensure grid_type is 'linear', 'nonlinear', or 'exp_mult'"
        )

    return Grid

# 2. Jump to Grid # n x m
@numba.njit
def jump_to_grid_dur(n_vals, m_vals, probs, dist_nGrid, dist_mGrid
                     ):
    '''
    Distributes values onto a predefined grid, maintaining the means. m_vals and perm_vals are realizations of market resources and permanent income while
    dist_mGrid and dist_pGrid are the predefined grids of market resources and permanent income, respectively. That is, m_vals and perm_vals do not necesarily lie on their
    respective grids. Returns probabilities of each gridpoint on the combined grid of market resources and permanent income.


    Parameters
    ----------
    n_vals: np.array
            Durable stock values

    m_vals: np.array
            Market resource values

    probs: np.array
            Shock probabilities associated with m_vals

    dist_nGrid : np.array
            Grid over normalized durable stock

    dist_mGrid : np.array
            Grid over normalized market resources

    Returns
    -------
    probGrid.flatten(): np.array
             Probabilities of each gridpoint on the combined grid of market resources and permanent income
    '''

    probGrid = np.zeros((len(dist_nGrid), len(dist_mGrid)))  # nxm

    # Maybe use np.searchsorted as opposed to np.digitize
    nIndex = np.digitize(n_vals, dist_nGrid) - 1
    nIndex[n_vals <= dist_nGrid[0]] = -1
    nIndex[n_vals >= dist_nGrid[-1]] = len(dist_nGrid) - 1

    # the following three lines hold the same intuition as above
    mIndex = np.digitize(m_vals, dist_mGrid) - 1  # Array indicating in which bin each values of m_vals lies in relative to dist_mGrid. Bins lie between between point of Dist_mGrid.
    # For instance, if mval lies between dist_mGrid[4] and dist_mGrid[5] it is in bin 4 (would be 5 if 1 was not subtracted in the previous line).
    mIndex[m_vals <= dist_mGrid[0]] = -1  # if the value is less than the smallest value on dist_mGrid assign it an index of -1
    mIndex[m_vals >= dist_mGrid[-1]] = len(dist_mGrid) - 1  # if value if greater than largest value on dist_mGrid assign it an index of the length of the grid minus 1

    # Same logic as above except the weights here concern the permanent income grid
    for i in range(len(n_vals)):
        if nIndex[i] == -1:
            nlowerIndex = 0
            nupperIndex = 0
            nlowerWeight = 1.0
            nupperWeight = 0.0
        elif nIndex[i] == len(dist_nGrid) - 1:
            nlowerIndex = -1
            nupperIndex = -1
            nlowerWeight = 1.0
            nupperWeight = 0.0
        else:
            nlowerIndex = nIndex[i]
            nupperIndex = nIndex[i] + 1
            nlowerWeight = (dist_nGrid[nupperIndex] - n_vals[i]) / (
                    dist_nGrid[nupperIndex] - dist_nGrid[nlowerIndex])
            nupperWeight = 1.0 - nlowerWeight

    for i in range(len(m_vals)):
        if mIndex[i] == -1:  # if mval is below smallest gridpoint, then assign it a weight of 1.0 for lower weight.
            mlowerIndex = 0
            mupperIndex = 0
            mlowerWeight = 1.0
            mupperWeight = 0.0
        elif mIndex[i] == len(
                dist_mGrid) - 1:  # if mval is greater than maximum gridpoint, then assign the following weights
            mlowerIndex = -1
            mupperIndex = -1
            mlowerWeight = 1.0
            mupperWeight = 0.0
        else:  # Standard case where mval does not lie past any extremes
            # identify which two points on the grid the mval is inbetween
            mlowerIndex = mIndex[i]
            mupperIndex = mIndex[i] + 1
            # Assign weight to the indices that bound the m_vals point. Intuitively, an mval perfectly between two points on the mgrid will assign a weight of .5 to the gridpoint above and below
            mlowerWeight = (dist_mGrid[mupperIndex] - m_vals[i]) / (dist_mGrid[mupperIndex] - dist_mGrid[
                mlowerIndex])  # Metric to determine weight of gridpoint/index below. Intuitively, mvals that are close to gridpoint/index above are assigned a smaller mlowerweight.
            mupperWeight = 1.0 - mlowerWeight  # weight of gridpoint/ index above



        probGrid[nlowerIndex][mlowerIndex] += probs[i] * nlowerWeight * mlowerWeight   # probability of gridpoint below mval and nval
        probGrid[nlowerIndex][mupperIndex] += probs[i] * nlowerWeight * mupperWeight  # probability of gridpoint below mval and above nval
        probGrid[nupperIndex][mlowerIndex] += probs[i] * nupperWeight * mlowerWeight  # probability of gridpoint above mval and below nval
        probGrid[nupperIndex][mupperIndex] += probs[i] * nupperWeight * mupperWeight  # probability of gridpoint above mval and above nval

    return probGrid.flatten()

@numba.njit(parallel=True)
def gen_tran_matrix_dur(dist_nGrid, dist_mGrid, bNext, nNext_unshocked, shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist):
    TranMatrix = np.zeros((len(dist_nGrid) * len(dist_mGrid), len(dist_nGrid) * len(dist_mGrid)))
    for i in numba.prange(len(dist_nGrid)):
        for j in numba.prange(len(dist_mGrid)):
            nNext_ij = nNext_unshocked[
                           i, j] / perm_shks  # Computes next period's normalized durable stock by applying permanent income shock:
            mNext_ij = bNext[
                           i, j] / perm_shks + tran_shks  # Compute next period's market resources given todays bank balances bnext[i]
            TranMatrix[:, i * len(dist_mGrid) + j] = LivPrb * jump_to_grid_dur(nNext_ij, mNext_ij, shk_prbs,
                                                                               dist_nGrid, dist_mGrid) + (
                                                             1.0 - LivPrb) * NewBornDist
    return TranMatrix

@numba.njit
def compute_erg_dstn(transition_matrix,method):
    if method == None:
        eigen, ergodic_distr = sp.linalg.eigs(transition_matrix[0], v0=np.ones(len(transition_matrix[0])), k=1,
                                          which='LM')  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real / np.sum(ergodic_distr.real)

    if method == 'Pontus':

        b = np.zeros(len(transition_matrix))
        b[0] = 1

        T = transition_matrix - np.identity(len(transition_matrix))

        T[0, :] = b

        s_hat = np.linalg.solve(T, b)

        ergodic_distr = s_hat.real / np.sum(s_hat.real)

    return ergodic_distr
# For graphs:
### Plot decision function of adjusting for each n and m grid
def decision_function(model):
    widgets.interact(_decision_functions,
                     model=widgets.fixed(model),
                     t=widgets.Dropdown(description='t',
                                        options=list(range(model.T_cycle + 1)), value=0),
                     name=widgets.Dropdown(description='name',
                                           options=['discrete', 'total', 'adj', 'keep'], value='discrete')
                     )

def _decision_functions(model, t, name):
    if model.cycles == 0:
        t = 0
    if name == 'discrete':
        _discrete(model, t)
    elif name == 'total':
        _total(model, t)
    elif name == 'adj':
        _adj(model, t)
    elif name == 'keep':
        _keep(model, t)

def _discrete(model, t):
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    I = model.solution[t].adjusting(n, m) > 0

    x = m[I].ravel()
    y = n[I].ravel()
    ax.scatter(x, y, s=2, label='adjust')

    x = m[~I].ravel()
    y = n[~I].ravel()
    ax.scatter(x, y, s=2, label='keep')

    ax.set_title(f'optimal discrete choice ($t = {t}$)', pad=10)

    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # g. details
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_xlim([mNrmGrid[0], mNrmGrid[-1]])
    ax.set_ylabel('$n_t$')
    ax.set_ylim([nNrmGrid[0], nNrmGrid[-1]])

    plt.show()

def _adj(model, t):
    # grids
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection='3d')
    ax_d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
    ax_v = fig.add_subplot(2, 2, 4, projection='3d')

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFuncAdj_plt = np.zeros(shape)
    dFuncAdj_plt = np.zeros(shape)
    exFuncAdj_plt = np.zeros(shape)
    vFuncAdj_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            x = (1 - model.adjC) * nNrmGrid[i_n] + mNrmGrid[i_m]
            cFuncAdj_plt[i_n, i_m] = model.solution[t].cFuncAdj(x)
            dFuncAdj_plt[i_n, i_m] = model.solution[t].dFuncAdj(x)
            exFuncAdj_plt[i_n, i_m] = model.solution[t].exFuncAdj(x)
            vFuncAdj_plt[i_n, i_m] = model.solution[t].vFuncAdj(x)

    ax_c.plot_surface(n, m, cFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
    ax_c.set_title(f'$c^{{adj}}$ ($t = {t}$)', pad=10)

    ax_d.plot_surface(n, m, dFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
    ax_d.set_title(f'$d^{{adj}}$ ($t = {t}$)', pad=10)

    ax_ex.plot_surface(n, m, exFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
    ax_ex.set_title(f'$ex^{{adj}}$ ($t = {t}$)', pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{adj}}$ ($t = {t}$)', pad=10)

    # e. details
    for ax in [ax_c, ax_v]:
        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()

def _keep(model, t):
    # grids
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection='3d')
    ax_d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
    ax_v = fig.add_subplot(2, 2, 4, projection='3d')

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFuncKeep_plt = np.zeros(shape)
    dFuncKeep_plt = np.zeros(shape)
    exFuncKeep_plt = np.zeros(shape)
    vFuncKeep_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFuncKeep_plt[i_n, i_m] = model.solution[t].cFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            dFuncKeep_plt[i_n, i_m] = model.solution[t].dFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            exFuncKeep_plt[i_n, i_m] = model.solution[t].exFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            vFuncKeep_plt[i_n, i_m] = model.solution[t].vFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])

    ax_c.plot_surface(n, m, cFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_c.set_title(f'$c^{{keep}}$ ($t = {t}$)', pad=10)

    ax_d.plot_surface(n, m, dFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_d.set_title(f'$d^{{keep}}$ ($t = {t}$)', pad=10)

    ax_ex.plot_surface(n, m, exFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_ex.set_title(f'$ex^{{keep}}$ ($t = {t}$)', pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{keep}}$ ($t = {t}$)', pad=10)

    # e. details
    for ax in [ax_c, ax_v]:
        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()

def _total(model, t):
    # grids
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection='3d')
    ax_d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
    ax_v = fig.add_subplot(2, 2, 4, projection='3d')

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFunc_plt = np.zeros(shape)
    dFunc_plt = np.zeros(shape)
    exFunc_plt = np.zeros(shape)
    vFunc_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFunc_plt[i_n, i_m] = model.solution[t].cFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            dFunc_plt[i_n, i_m] = model.solution[t].dFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            exFunc_plt[i_n, i_m] = model.solution[t].exFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            vFunc_plt[i_n, i_m] = model.solution[t].vFunc(nNrmGrid[i_n], mNrmGrid[i_m])

    ax_c.plot_surface(n, m, cFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_c.set_title(f'$c^{{total}}$ ($t = {t}$)', pad=10)

    ax_d.plot_surface(n, m, dFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_d.set_title(f'$d^{{total}}$ ($t = {t}$)', pad=10)

    ax_ex.plot_surface(n, m, exFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_ex.set_title(f'$ex^{{total}}$ ($t = {t}$)', pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{total}}$ ($t = {t}$)', pad=10)

    # e. details
    for ax in [ax_c, ax_d, ax_ex, ax_v]:
        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()



### Plot decision function of adjusting for each n and m grid
def decision_function_nm(model):
    widgets.interact(_decision_functions_nm,
                     model=widgets.fixed(model),
                     t=widgets.Dropdown(description='t',
                                        options=list(range(model.T_cycle + 1)), value=0),
                     name=widgets.Dropdown(description='name',
                                           options=['discrete', 'total', 'adj', 'keep'], value='discrete')
                     )

def _decision_functions_nm(model, t, name):
    if model.cycles == 0:
        t = 0
    if name == 'discrete':
        _discrete_nm(model, t)
    elif name == 'total':
        _total_nm(model, t)
    elif name == 'adj':
        _adj_nm(model, t)
    elif name == 'keep':
        _keep_nm(model, t)

def _discrete_nm(model, t):
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    #I = model.solution[t].adjusting(n, m) > 0
    I = model.solution[t].inv_vFuncAdj(n, m) >= model.solution[t].inv_vFuncKeep(n, m)
    # inv_vFunc_Diff = inv_vFuncAdj_array[i_d] - inv_vFuncKeep_array[i_d]
    # # Find the lower and upper threshold where consumer keeps
    # lS = np.where(inv_vFunc_Diff < - tol)[0]

    x = m[I].ravel()
    y = n[I].ravel()
    ax.scatter(x, y, s=2, label='adjust')

    x = m[~I].ravel()
    y = n[~I].ravel()
    ax.scatter(x, y, s=2, label='keep')

    ax.set_title(f'optimal discrete choice ($t = {t}$)', pad=10)

    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # g. details
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_xlim([mNrmGrid[0], mNrmGrid[-1]])
    ax.set_ylabel('$n_t$')
    ax.set_ylim([nNrmGrid[0], nNrmGrid[-1]])

    plt.show()

def _adj_nm(model, t):
    # grids
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection='3d')
    ax_d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
    ax_v = fig.add_subplot(2, 2, 4, projection='3d')

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFuncAdj_plt = np.zeros(shape)
    dFuncAdj_plt = np.zeros(shape)
    exFuncAdj_plt = np.zeros(shape)
    vFuncAdj_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFuncAdj_plt[i_n, i_m] = model.solution[t].cFuncAdj(nNrmGrid[i_n], mNrmGrid[i_m])
            dFuncAdj_plt[i_n, i_m] = model.solution[t].dFuncAdj(nNrmGrid[i_n], mNrmGrid[i_m])
            exFuncAdj_plt[i_n, i_m] = model.solution[t].exFuncAdj(nNrmGrid[i_n], mNrmGrid[i_m])
            vFuncAdj_plt[i_n, i_m] = model.solution[t].inv_vFuncAdj(nNrmGrid[i_n], mNrmGrid[i_m])

    ax_c.plot_surface(n, m, cFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
    ax_c.set_title(f'$c^{{adj}}$ ($t = {t}$)', pad=10)

    ax_d.plot_surface(n, m, dFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
    ax_d.set_title(f'$d^{{adj}}$ ($t = {t}$)', pad=10)

    ax_ex.plot_surface(n, m, exFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
    ax_ex.set_title(f'$ex^{{adj}}$ ($t = {t}$)', pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{adj}}$ ($t = {t}$)', pad=10)

    # e. details
    for ax in [ax_c, ax_v]:
        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()

def _keep_nm(model, t):
    # grids
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection='3d')
    ax_d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
    ax_v = fig.add_subplot(2, 2, 4, projection='3d')

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFuncKeep_plt = np.zeros(shape)
    dFuncKeep_plt = np.zeros(shape)
    exFuncKeep_plt = np.zeros(shape)
    vFuncKeep_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFuncKeep_plt[i_n, i_m] = model.solution[t].cFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            dFuncKeep_plt[i_n, i_m] = model.solution[t].dFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            exFuncKeep_plt[i_n, i_m] = model.solution[t].exFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            vFuncKeep_plt[i_n, i_m] = model.solution[t].inv_vFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])

    ax_c.plot_surface(n, m, cFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_c.set_title(f'$c^{{keep}}$ ($t = {t}$)', pad=10)

    ax_d.plot_surface(n, m, dFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_d.set_title(f'$d^{{keep}}$ ($t = {t}$)', pad=10)

    ax_ex.plot_surface(n, m, exFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_ex.set_title(f'$ex^{{keep}}$ ($t = {t}$)', pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{keep}}$ ($t = {t}$)', pad=10)

    # e. details
    for ax in [ax_c, ax_v]:
        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()

def _total_nm(model, t):
    # grids
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection='3d')
    ax_d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
    ax_v = fig.add_subplot(2, 2, 4, projection='3d')

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFunc_plt = np.zeros(shape)
    dFunc_plt = np.zeros(shape)
    exFunc_plt = np.zeros(shape)
    vFunc_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFunc_plt[i_n, i_m] = model.solution[t].cFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            dFunc_plt[i_n, i_m] = model.solution[t].dFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            exFunc_plt[i_n, i_m] = model.solution[t].exFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            vFunc_plt[i_n, i_m] = model.solution[t].inv_vFunc(nNrmGrid[i_n], mNrmGrid[i_m])

    ax_c.plot_surface(n, m, cFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_c.set_title(f'$c^{{total}}$ ($t = {t}$)', pad=10)

    ax_d.plot_surface(n, m, dFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_d.set_title(f'$d^{{total}}$ ($t = {t}$)', pad=10)

    ax_ex.plot_surface(n, m, exFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_ex.set_title(f'$ex^{{total}}$ ($t = {t}$)', pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{total}}$ ($t = {t}$)', pad=10)

    # e. details
    for ax in [ax_c, ax_d, ax_ex, ax_v]:
        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()


### Plot decision function of adjusting for each n and m grid
def decision_function_latest(model):
    widgets.interact(_decision_functions,
                     model=widgets.fixed(model),
                     t=widgets.Dropdown(description='t',
                                        options=list(range(model.T_cycle + 1)), value=0),
                     name=widgets.Dropdown(description='name',
                                           options=['discrete', 'total', 'adj', 'keep'], value='discrete')
                     )

def _decision_functions(model, t, name):
    if model.cycles == 0:
        t = 0
    if name == 'discrete':
        _discrete(model, t)
    elif name == 'total':
        _total(model, t)
    elif name == 'adj':
        _adj(model, t)
    elif name == 'keep':
        _keep(model, t)

def _discrete(model, t):
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    I = model.solution[t].adjusting(n, m) > 0

    # I = model.solution[t].inv_vFuncAdj(n, m) >= model.solution[t].inv_vFuncKeep(n, m)
    # inv_vFunc_Diff = inv_vFuncAdj_array[i_d] - inv_vFuncKeep_array[i_d]
    # # Find the lower and upper threshold where consumer keeps
    # lS = np.where(inv_vFunc_Diff < - tol)[0]

    x = m[I].ravel()
    y = n[I].ravel()
    ax.scatter(x, y, s=2, label='adjust')

    x = m[~I].ravel()
    y = n[~I].ravel()
    ax.scatter(x, y, s=2, label='keep')

    ax.set_title(f'optimal discrete choice ($t = {t}$)', pad=10)

    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # g. details
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_xlim([mNrmGrid[0], mNrmGrid[-1]])
    ax.set_ylabel('$n_t$')
    ax.set_ylim([nNrmGrid[0], nNrmGrid[-1]])

    plt.show()

def _adj(model, t):
    if model.AdjX == True:
        # grids
        xNrmMin = model.mNrmMin + (1 - model.adjC) * model.nNrmMin
        xNrmMax = model.mNrmMax + (1 - model.adjC) * model.nNrmMax
        xNrmCount = model.mNrmCount
        xNrmGrid = construct_grid(xNrmMin, xNrmMax, xNrmCount, model.grid_type, model.NestFac)

        # xNrmGrid = construct_grid(model.xNrmMin, model.xNrmMax, model.xNrmCount, model.grid_type, model.NestFac)

        # b. figure
        fig = plt.figure(figsize=(12, 6))
        ax_c = fig.add_subplot(2, 2, 1)
        ax_d = fig.add_subplot(2, 2, 2)
        ax_ex = fig.add_subplot(2, 2, 3)
        ax_v = fig.add_subplot(2, 2, 4)

        # n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

        # c. plot consumption
        cFuncAdj_plt = model.solution[t].cFuncAdj(xNrmGrid)
        dFuncAdj_plt = model.solution[t].dFuncAdj(xNrmGrid)
        exFuncAdj_plt = model.solution[t].exFuncAdj(xNrmGrid)
        vFuncAdj_plt = model.solution[t].inv_vFuncAdj(xNrmGrid)

        ax_c.plot(xNrmGrid, cFuncAdj_plt, lw=2)
        ax_c.set_title(f'$c^{{adj}}$ ($t = {t}$)', pad=10)

        ax_d.plot(xNrmGrid, dFuncAdj_plt, lw=2)
        ax_d.set_title(f'$d^{{adj}}$ ($t = {t}$)', pad=10)

        ax_ex.plot(xNrmGrid, exFuncAdj_plt, lw=2)
        ax_ex.set_title(f'$ex^{{adj}}$ ($t = {t}$)', pad=10)

        # d. plot value function
        ax_v.plot(xNrmGrid, vFuncAdj_plt, lw=2)
        ax_v.set_title(f'neg. inverse $v^{{adj}}$ ($t = {t}$)', pad=10)

        # e. details
        for ax in [ax_c, ax_v]:
            ax.grid(True)
            ax.set_xlabel('$x_t$')
            ax.set_xlim([xNrmGrid[0], xNrmGrid[-1]])
            # ax.invert_xaxis()
        plt.legend()
        plt.show()

    else:
        # grids
        nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
        mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

        # b. figure
        fig = plt.figure(figsize=(12, 6))
        ax_c = fig.add_subplot(2, 2, 1, projection='3d')
        ax_d = fig.add_subplot(2, 2, 2, projection='3d')
        ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
        ax_v = fig.add_subplot(2, 2, 4, projection='3d')

        n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

        # c. plot consumption
        shape = (model.nNrmCount, model.mNrmCount)
        cFuncAdj_plt = np.zeros(shape)
        dFuncAdj_plt = np.zeros(shape)
        exFuncAdj_plt = np.zeros(shape)
        vFuncAdj_plt = np.zeros(shape)

        for i_n in range(model.nNrmCount):
            for i_m in range(model.mNrmCount):
                cFuncAdj_plt[i_n, i_m] = model.solution[t].cFuncAdj(nNrmGrid[i_n], mNrmGrid[i_m])
                dFuncAdj_plt[i_n, i_m] = model.solution[t].dFuncAdj(nNrmGrid[i_n], mNrmGrid[i_m])
                exFuncAdj_plt[i_n, i_m] = model.solution[t].exFuncAdj(nNrmGrid[i_n], mNrmGrid[i_m])
                vFuncAdj_plt[i_n, i_m] = model.solution[t].inv_vFuncAdj(nNrmGrid[i_n], mNrmGrid[i_m])

        ax_c.plot_surface(n, m, cFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
        ax_c.set_title(f'$c^{{adj}}$ ($t = {t}$)', pad=10)

        ax_d.plot_surface(n, m, dFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
        ax_d.set_title(f'$d^{{adj}}$ ($t = {t}$)', pad=10)

        ax_ex.plot_surface(n, m, exFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
        ax_ex.set_title(f'$ex^{{adj}}$ ($t = {t}$)', pad=10)

        # d. plot value function
        ax_v.plot_surface(n, m, vFuncAdj_plt, cmap=cm.viridis, edgecolor='none')
        ax_v.set_title(f'neg. inverse $v^{{adj}}$ ($t = {t}$)', pad=10)

        # e. details
        for ax in [ax_c, ax_v]:
            ax.grid(True)
            ax.set_xlabel('$n_t$')
            ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
            ax.set_ylabel('$m_t$')
            ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
            ax.invert_xaxis()
        plt.legend()
        plt.show()

def _keep(model, t):
    # grids
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection='3d')
    ax_d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
    ax_v = fig.add_subplot(2, 2, 4, projection='3d')

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFuncKeep_plt = np.zeros(shape)
    dFuncKeep_plt = np.zeros(shape)
    exFuncKeep_plt = np.zeros(shape)
    vFuncKeep_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFuncKeep_plt[i_n, i_m] = model.solution[t].cFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            dFuncKeep_plt[i_n, i_m] = model.solution[t].dFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            exFuncKeep_plt[i_n, i_m] = model.solution[t].exFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])
            vFuncKeep_plt[i_n, i_m] = model.solution[t].inv_vFuncKeep(nNrmGrid[i_n], mNrmGrid[i_m])

    ax_c.plot_surface(n, m, cFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_c.set_title(f'$c^{{keep}}$ ($t = {t}$)', pad=10)

    ax_d.plot_surface(n, m, dFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_d.set_title(f'$d^{{keep}}$ ($t = {t}$)', pad=10)

    ax_ex.plot_surface(n, m, exFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_ex.set_title(f'$ex^{{keep}}$ ($t = {t}$)', pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFuncKeep_plt, cmap=cm.viridis, edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{keep}}$ ($t = {t}$)', pad=10)

    # e. details
    for ax in [ax_c, ax_d, ax_ex, ax_v]:
        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()

def _total(model, t):
    # grids
    nNrmGrid = construct_grid(model.nNrmMin, model.nNrmMax, model.nNrmCount, model.grid_type, model.NestFac)
    mNrmGrid = construct_grid(model.mNrmMin, model.mNrmMax, model.mNrmCount, model.grid_type, model.NestFac)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection='3d')
    ax_d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_ex = fig.add_subplot(2, 2, 3, projection='3d')
    ax_v = fig.add_subplot(2, 2, 4, projection='3d')

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing='ij')

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFunc_plt = np.zeros(shape)
    dFunc_plt = np.zeros(shape)
    exFunc_plt = np.zeros(shape)
    vFunc_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFunc_plt[i_n, i_m] = model.solution[t].cFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            dFunc_plt[i_n, i_m] = model.solution[t].dFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            exFunc_plt[i_n, i_m] = model.solution[t].exFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            vFunc_plt[i_n, i_m] = model.solution[t].inv_vFunc(nNrmGrid[i_n], mNrmGrid[i_m])

    ax_c.plot_surface(n, m, cFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_c.set_title(f'$c^{{total}}$ ($t = {t}$)', pad=10)

    ax_d.plot_surface(n, m, dFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_d.set_title(f'$d^{{total}}$ ($t = {t}$)', pad=10)

    ax_ex.plot_surface(n, m, exFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_ex.set_title(f'$ex^{{total}}$ ($t = {t}$)', pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFunc_plt, cmap=cm.viridis, edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{total}}$ ($t = {t}$)', pad=10)

    # e. details
    for ax in [ax_c, ax_d, ax_ex, ax_v]:
        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()


# DCEGM: Make cFuncKeep for each durable
def DCEGM(nNrmGrid, aNrmGrid, mNrmGrid, qFunc_array, invwFunc_array, alpha, d_ubar, CRRA, BoroCnstArt):
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iii. Inverse uPC U(C,D) wrt C
    CRRAutilityP_inv = lambda C, D: ((C/(alpha*(D+d_ubar) ** ((1 - alpha) * (1 - CRRA))))**(1/(alpha*(1 - CRRA) - 1)))

    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGrid))
    q_c = np.zeros(keep_shape)
    q_m = np.zeros(keep_shape)
    q_v = np.zeros(keep_shape)

    cKeepFunctionList = []
    vKeepFunctionList = []

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # use euler equation
        q_c[i_d] = CRRAutilityP_inv(qFunc_array[i_d], d_keep_aux)
        q_m[i_d] = aNrmGrid + q_c[i_d]
        q_v[i_d] = CRRAutility(q_c[i_d], d_keep_aux) + (-1.0 / invwFunc_array[i_d])

        # Transformations for value funtion interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        # a) transform q_c, q_m, q_v as a function of m instead of a
        # Constraint functions add 20 points
        mNrmGrid_aux = construct_grid(0, q_m[i_d,0], 20, 'exp_mult', 3)
        invwFunc_con_array = np.ones(len(mNrmGrid_aux)) * (-1.0 / invwFunc_array[i_d, 0])
        q_v_con = CRRAutility(mNrmGrid_aux, np.ones(len(mNrmGrid_aux)) * d_keep) + invwFunc_con_array # invwFunc_array = (nNrmGrid,aNrmGrid) -> take the lowest

        m_egm = np.concatenate(([mNrmGrid_aux, q_m[i_d,1:]]))
        c_egm = np.concatenate(([mNrmGrid_aux, q_c[i_d,1:]]))
        v_egm = np.concatenate(([q_v_con, q_v[i_d,1:]]))

        vt_egm = -1.0 / v_egm
        vt_egm = vTransf(vt_egm)

        # b) Compute non-decreasing segments
        start, end = calc_nondecreasing_segments(m_egm, vt_egm)

        # c) Get segments
        segments = []
        m_segments = []
        vt_segments = []
        c_segments = []
        for j in range(len(start)):
            idx = range(start[j], end[j] + 1)
            segments.append([m_egm[idx], vt_egm[idx]])
            m_segments.append(m_egm[idx])
            vt_segments.append((vt_egm[idx]))
            c_segments.append(c_egm[idx])

        # d) Upper envelope
        m_upper, v_upper, inds_upper = upper_envelope(segments)

        # e) Envelope Consumption
        c_env = np.zeros_like(m_upper) + np.nan
        for k, c_segm in enumerate(c_segments):
            c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])

        v_env = vUntransf(v_upper)

        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cKeepFunctionList.append(LinearFast(c_env, [m_upper]))
        vKeepFunctionList.append(LinearFast(v_env, [m_upper]))

    # Add kink point to mNrmGrid:
    mNrmGrid_New = np.insert(mNrmGrid, 0, q_m[:, 0])
    mNrmGrid_New = np.unique(mNrmGrid_New)
    mNrmGrid_New.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGrid_New))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    vFuncKeep_array = np.zeros(keep_shape)
    uPFuncKeep_array = np.zeros(keep_shape)

    #for cFunc in cFuncKeepList:


    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d,:] = np.ones(len(mNrmGrid_New)) * nNrmGrid[i_d]
        for i_m in range(len(mNrmGrid)):

            cFuncKeep_array[i_d,:] = cKeepFunctionList[i_d](mNrmGrid_New)
            vFuncKeep_array[i_d, :] = vKeepFunctionList[i_d](mNrmGrid_New)
            uPFuncKeep_array[i_d, :] = 1 / CRRAutilityP(cFuncKeep_array[i_d, :], d_keep)

    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGrid_New": mNrmGrid_New,
            "vFuncKeep_array": vFuncKeep_array,
            "uPFuncKeep_array": uPFuncKeep_array,
            }


# @njit
# def DCEGM(nNrmGrid, aNrmGrid, mNrmGrid, qFunc_array, invwFunc_array, alpha, d_ubar, CRRA, BoroCnstArt):
#     # i. U(C,D)
#     u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
#     CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)
#
#     # ii. uPC U(C,D) wrt C
#     CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))
#
#     # iii. Inverse uPC U(C,D) wrt C
#     CRRAutilityP_inv = lambda C, D: ((C/(alpha*(D+d_ubar) ** ((1 - alpha) * (1 - CRRA))))**(1/(alpha*(1 - CRRA) - 1)))
#
#     # Create empty container
#     keep_shape = (len(nNrmGrid), len(mNrmGrid))
#     #dcegm_shape = (len(nNrmGrid), len(mNrmGrid))
#     cFuncKeep_array = np.zeros(keep_shape)
#     dFuncKeep_array = np.zeros(keep_shape)
#     vFuncKeep_array = np.zeros(keep_shape)
#     uPFuncKeep_array = np.zeros(keep_shape)
#
#     m_upper_array = np.zeros(keep_shape)
#     #c_egm = np.zeros(keep_shape)
#     #v_egm = np.zeros(keep_shape)
#     q_c = np.zeros(keep_shape)
#     q_m = np.zeros(keep_shape)
#     q_v = np.zeros(keep_shape)
#
#     for i_d in range(len(nNrmGrid)):
#         d_keep = nNrmGrid[i_d]
#         d_keep_aux = np.ones(len(aNrmGrid)) * d_keep
#
#         # use euler equation
#         q_c[i_d] = CRRAutilityP_inv(qFunc_array[i_d], d_keep_aux)
#         q_m[i_d] = aNrmGrid + q_c[i_d]
#         q_v[i_d] = CRRAutility(q_c[i_d], d_keep_aux) + (-1.0 / invwFunc_array[i_d])
#
#         # Transformations for value funtion interpolation
#         vTransf = lambda x: np.exp(x)
#         vUntransf = lambda x: np.log(x)
#
#         # a) transform q_c, q_m, q_v as a function of m instead of a
# #        c_egm_func = LinearInterp(q_m[i_d], q_c[i_d])
# #        v_egm_func = LinearInterp(q_m[i_d], q_v[i_d])
# #        m_egm = mNrmGrid
#
#         # Constraint functions add 20 points
#         mNrmGrid_aux = construct_grid(0, q_m[i_d,0], 20, 'exp_mult', 3)
# #        q_c_con = mNrmGrid
#         invwFunc_con_array = np.ones(len(mNrmGrid_aux)) * (-1.0 / invwFunc_array[i_d, 0])
#         q_v_con = CRRAutility(mNrmGrid_aux, np.ones(len(mNrmGrid_aux)) * d_keep) + invwFunc_con_array # invwFunc_array = (nNrmGrid,aNrmGrid) -> take the lowest
#
#         # c_egm_func does only work for non-constraint parts. Add the constraint one.
#         # Find where mNrmGrid <= q_m[i_d,0]
#         # Stack mNrmGrid part before q_m[i_d,0]
#
#         #idx = np.max(np.where(mNrmGrid < q_m[i_d,0] )[0])
#         m_egm = np.concatenate(([mNrmGrid_aux, q_m[i_d,1:len(mNrmGrid)-len(mNrmGrid_aux)+1]]))
#         c_egm = np.concatenate(([mNrmGrid_aux, q_c[i_d,1:len(mNrmGrid)-len(mNrmGrid_aux)+1]]))
#         v_egm = np.concatenate(([q_v_con, q_v[i_d,1:len(mNrmGrid)-len(mNrmGrid_aux)+1]]))
#
#         # cCrit = q_m[i_d,0] - BoroCnstArt
#         # mNrmNow = np.concatenate(([BoroCnstArt, q_m[i_d,0]], q_m[i_d,1:]))
#         # cNrmNow = np.concatenate(([0.0, cCrit], q_c[i_d,1:]))
#
#         # vNrmNow = np.concatenate(([(-1.0 / invwFunc_array[i_d, 0])], q_v[i_d,1:]))
#         # for i_m in range(len(mNrmNow)):
#         #     if mNrmNow[i_m] <= q_m[i_d, 0]:  # Constraint
#         #         #print('constraint')
#         #         c_egm[i_d, i_m] = mNrmNow[i_m]
#         #         v_egm[i_d, i_m] = q_v_con[i_m]
#         #     else:
#         #         #print('not constraint')
#         #         c_egm[i_d, i_m] = c_egm_func(mNrmGrid[i_m])
#         #         v_egm[i_d, i_m] = v_egm_func(mNrmGrid[i_m])
#
#         vt_egm = -1.0 / v_egm
#         vt_egm = vTransf(vt_egm)
#
#         # b) Compute non-decreasing segments
#         start, end = calc_nondecreasing_segments(m_egm, vt_egm)
#
#         # c) Get segments
#         segments = []
#         m_segments = []
#         vt_segments = []
#         c_segments = []
#         for j in range(len(start)):
#             idx = range(start[j], end[j] + 1)
#             segments.append([m_egm[idx], vt_egm[idx]])
#             m_segments.append(m_egm[idx])
#             vt_segments.append((vt_egm[idx]))
#             c_segments.append(c_egm[idx])
#
#         # d) Upper envelope
#         m_upper, v_upper, inds_upper = upper_envelope(segments)
#
#         # e) Envelope Consumption
#         c_env = np.zeros_like(m_upper) + np.nan
#         for k, c_segm in enumerate(c_segments):
#             c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])
#
#         v_env = vUntransf(v_upper)
#
#         cFuncKeep_array[i_d, :] = c_env
#         vFuncKeep_array[i_d, :] = v_env
#         uPFuncKeep_array[i_d, :] = 1/CRRAutilityP(cFuncKeep_array[i_d, :], d_keep)
#         dFuncKeep_array[i_d, :] = nNrmGrid[i_d]
#         m_upper_array[i_d, :] = m_upper
#
#     print(q_m[:, 0])
#     # for i_d in range(len(nNrmGrid)):
#     #     d_keep = nNrmGrid[i_d]
#     #     d_keep_aux = np.ones(len(aNrmGrid)) * d_keep
#     #
#     #     # use euler equation
#     #     q_c[i_d] = CRRAutilityP_inv(qFunc_array[i_d], d_keep_aux)
#     #     q_m[i_d] = aNrmGrid + q_c[i_d]
#     #     q_v[i_d] = CRRAutility(q_c[i_d], d_keep_aux) + (-1.0 / invwFunc_array[i_d])
#     #
#     #     # Transformations for value funtion interpolation
#     #     vTransf = lambda x: np.exp(x)
#     #     vUntransf = lambda x: np.log(x)
#     #
#     #     # a) transform q_c, q_m, q_v as a function of m instead of a
#     #     c_egm_func = LinearInterp(q_m[i_d], q_c[i_d])
#     #     v_egm_func = LinearInterp(q_m[i_d], q_v[i_d])
#     #     m_egm = mNrmGrid
#     #
#     #     # Constraint functions:
#     #     q_c_con = mNrmGrid
#     #     invwFunc_con_array = np.ones(len(mNrmGrid)) * (-1.0 / invwFunc_array[i_d, 0])
#     #     q_v_con = CRRAutility(q_c_con, d_keep_aux) + invwFunc_con_array # invwFunc_array = (nNrmGrid,aNrmGrid) -> take the lowest
#     #
#     #     # c_egm_func does only work for non-constraint parts. Add the constraint one.
#     #     for i_m in range(len(mNrmGrid)):
#     #         if mNrmGrid[i_m] <= q_m[i_d, 0]:  # Constraint
#     #             #print('constraint')
#     #             c_egm[i_d, i_m] = mNrmGrid[i_m]
#     #             v_egm[i_d, i_m] = q_v_con[i_m]
#     #         else:
#     #             #print('not constraint')
#     #             c_egm[i_d, i_m] = c_egm_func(mNrmGrid[i_m])
#     #             v_egm[i_d, i_m] = v_egm_func(mNrmGrid[i_m])
#     #
#     #     vt_egm = -1.0 / v_egm[i_d]
#     #     vt_egm = vTransf(vt_egm)
#     #
#     #     # b) Compute non-decreasing segments
#     #     start, end = calc_nondecreasing_segments(m_egm, vt_egm)
#     #
#     #     # c) Get segments
#     #     segments = []
#     #     m_segments = []
#     #     vt_segments = []
#     #     c_segments = []
#     #     for j in range(len(start)):
#     #         idx = range(start[j], end[j] + 1)
#     #         segments.append([m_egm[idx], vt_egm[idx]])
#     #         m_segments.append(m_egm[idx])
#     #         vt_segments.append((vt_egm[idx]))
#     #         c_segments.append(c_egm[i_d, idx])
#     #
#     #     # d) Upper envelope
#     #     m_upper, v_upper, inds_upper = upper_envelope(segments)
#     #
#     #     # e) Envelope Consumption
#     #     c_env = np.zeros_like(m_upper) + np.nan
#     #     for k, c_segm in enumerate(c_segments):
#     #         c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])
#     #
#     #     v_env = vUntransf(v_upper)
#     #
#     #     cFuncKeep_array[i_d, :] = c_env
#     #     vFuncKeep_array[i_d, :] = v_env
#     #     uPFuncKeep_array[i_d, :] = 1/CRRAutilityP(cFuncKeep_array[i_d, :], d_keep)
#     #     dFuncKeep_array[i_d, :] = nNrmGrid[i_d]
#     #     m_upper_array[i_d, :] = m_upper
#
#     return {"cFuncKeep_array": cFuncKeep_array,
#             "dFuncKeep_array": dFuncKeep_array,
#             "m_upper": m_upper,
#             "vFuncKeep_array": vFuncKeep_array,
#             "uPFuncKeep_array": uPFuncKeep_array,
#             }

# @njit
def DCEGM_invVFunc(nNrmGrid, aNrmGrid, mNrmGrid, qFunc_array, wFunc_array, inv_wFunc_array, alpha, d_ubar, CRRA):
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    # CRRAutilityP_inv = lambda C, D: ((C/(alpha*(D+d_ubar) ** ((1 - alpha) * (1 - CRRA))))**(1/(alpha*(1 - CRRA) - 1)))
    CRRAutilityP_inv = lambda C, D: utilityP_inv(CRRAutilityP(C, D), CRRA)

    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGrid))
    q_c = np.zeros(keep_shape)
    q_m = np.zeros(keep_shape)
    q_v = np.zeros(keep_shape)
    # inv_q_v = np.zeros(keep_shape)
    # q_uP  = np.zeros(keep_shape)

    cFuncKeepList = []
    vFuncKeepList = []
    inv_vFuncKeepList = []
    uPFuncKeepList = []

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # use euler equation
        q_c[i_d] = utilityP_inv(qFunc_array[i_d],CRRA)
        #q_c[i_d] = CRRAutilityP_inv(CRRAutilityP(aNrmGrid, d_keep_aux),CRRA)
        q_m[i_d] = aNrmGrid + q_c[i_d]
        q_v[i_d] = CRRAutility(q_c[i_d], d_keep_aux) + wFunc_array[i_d] #inv_wFunc_array[i_d]
        # q_uP[i_d] = CRRAutilityP(q_c[i_d], d_keep_aux) + qFunc_array[i_d]

        # inv_q_v[i_d] = CRRAutility_inv(q_c[i_d], d_keep_aux) + inv_wFunc_array[i_d] #wFunc_array[i_d]

        # Transformations for value funtion interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        # a) transform q_c, q_m, q_v as a function of m instead of a
        # Constraint functions add points from mNrmGrid prior to q_m[i_d,0]
        idx = np.max(np.where(mNrmGrid < q_m[i_d, 0])[0])
        mNrmGrid_aux = mNrmGrid[:idx]
        mNrmGrid_aux = np.concatenate((mNrmGrid_aux, [q_m[i_d, 0]]))

        # Alternatively: create a new grid for this piece with length 20
        # mNrmGrid_aux = construct_grid(0, q_m[i_d,0], 20, 'exp_mult', 3)
        wFunc_con_array = np.ones(len(mNrmGrid_aux)) * wFunc_array[i_d, 0]
        # uPFunc_con_array = np.ones(len(mNrmGrid_aux)) * qFunc_array[i_d, 0]
        # inv_wFunc_con_array = np.ones(len(mNrmGrid_aux)) * inv_wFunc_array[i_d, 0]

        q_v_con = CRRAutility(mNrmGrid_aux, np.ones(len(mNrmGrid_aux)) * d_keep) + wFunc_con_array # invwFunc_array = (nNrmGrid,aNrmGrid) -> take the lowest
        # q_uP_con = CRRAutilityP(mNrmGrid_aux, np.ones(len(mNrmGrid_aux)) * d_keep) + uPFunc_con_array

        # inv_q_v_con = CRRAutility_inv(mNrmGrid_aux, np.ones(len(mNrmGrid_aux)) * d_keep) + inv_wFunc_con_array

        m_egm = np.concatenate(([mNrmGrid_aux, q_m[i_d,1:]]))
        c_egm = np.concatenate(([mNrmGrid_aux, q_c[i_d,1:]]))
        v_egm = np.concatenate(([q_v_con, q_v[i_d,1:]]))
        # uP_egm = np.concatenate(([q_uP_con, q_uP[i_d,1:]]))
        # inv_uP_egm = utilityP_inv(uP_egm, CRRA)
        # inv_v_egm = np.concatenate(([inv_q_v_con, inv_q_v[i_d,1:]]))

        vt_egm = vTransf(v_egm)
        # b) Compute non-decreasing segments
        start, end = calc_nondecreasing_segments(m_egm, vt_egm)

        # c) Get segments
        segments = []
        m_segments = []
        vt_segments = []
        c_segments = []
        # inv_uP_segments = []
        # inv_v_segments = []
        for j in range(len(start)):
            idx = range(start[j], end[j] + 1)
            segments.append([m_egm[idx], vt_egm[idx]])
            m_segments.append(m_egm[idx])
            vt_segments.append((vt_egm[idx]))
            c_segments.append(c_egm[idx])
            # inv_uP_segments.append(uP_egm[idx])
            # inv_v_segments.append(inv_v_egm[idx])

        # d) Upper envelope
        m_upper, v_upper, inds_upper = upper_envelope(segments)

        # e) Envelope Consumption
        c_env = np.zeros_like(m_upper) + np.nan
        # inv_uP_env = np.zeros_like(m_upper) + np.nan
        # inv_v_env = np.zeros_like(m_upper) + np.nan

        for k, c_segm in enumerate(c_segments):
            c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])

        # for k, inv_uP_segm in enumerate(inv_uP_segments):
        #     inv_uP_env[inds_upper == k] = LinearInterp(m_segments[k], inv_uP_segm)(m_upper[inds_upper == k])

        # for k, inv_v_segm in enumerate(inv_v_segments):
        #    inv_v_env[inds_upper == k] = LinearInterp(m_segments[k], inv_v_segm)(m_upper[inds_upper == k])

        v_env = vUntransf(v_upper)
        inv_v_env = utility_inv(v_env,CRRA)
        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cFuncKeepList.append(LinearFast(c_env, [m_upper]))
        #vFuncKeepList.append(LinearFast(v_env, [m_upper]))
        inv_vFuncKeepList.append(LinearFast(inv_v_env, [m_upper]))
        vFuncKeepList.append(ValueFuncCRRA(LinearFast(inv_v_env, [m_upper]),CRRA))
        uPFuncKeepList.append(MargValueFuncCRRA(LinearFast(c_env, [m_upper]), CRRA))

    # Add kink point to mNrmGrid:
    mNrmGrid_New = np.insert(mNrmGrid, 0, q_m[:, 0])
    mNrmGrid_New = np.unique(mNrmGrid_New)
    mNrmGrid_New.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGrid_New))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    vFuncKeep_array = np.zeros(keep_shape)
    uPFuncKeep_array = np.zeros(keep_shape)
    inv_vFuncKeep_array = np.zeros(keep_shape)
    inv_uPFuncKeep_array = np.zeros(keep_shape)

    #for cFunc in cFuncKeepList:
    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d] = np.ones(len(mNrmGrid_New)) * nNrmGrid[i_d]
        for i_m in range(len(mNrmGrid)):

            cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGrid_New)
            vFuncKeep_array[i_d] = vFuncKeepList[i_d](mNrmGrid_New)
            inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGrid_New)
            uPFuncKeep_array[i_d] = uPFuncKeepList[i_d](mNrmGrid_New)   #CRRAutilityP(cFuncKeep_array[i_d], d_keep)
            inv_uPFuncKeep_array[i_d] = utilityP_inv(uPFuncKeep_array[i_d], CRRA) #CRRAutilityP_inv(cFuncKeep_array[i_d], d_keep)

    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGrid_New": mNrmGrid_New,
            "vFuncKeep_array": vFuncKeep_array,
            "uPFuncKeep_array": uPFuncKeep_array,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_uPFuncKeep_array": inv_uPFuncKeep_array,
            }



########################################################################################################################
def DCEGM_invVFunc_short(nNrmGrid, aNrmGrid, mNrmGrid, BoroCnstNat, qFunc_array, wFunc_array, alpha, d_ubar, CRRA): #, inv_wFunc_array, qFunc_array
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    CRRAutilityP_inv = lambda C, D: utilityP_inv(CRRAutilityP(C, D), CRRA)

    # Create empty container
    m_kink = np.zeros_like(nNrmGrid)
    cFuncKeepList = []
    vFuncKeepList = []
    inv_vFuncKeepList = []
    inv_vPFuncKeepList = []

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # use euler equation
        c_egm = utilityP_inv(qFunc_array[i_d], CRRA)
        # Consumption cannot be negative
        c_egm = np.clip(c_egm, 0, len(mNrmGrid))
        m_egm = aNrmGrid + c_egm
        v_egm = CRRAutility(c_egm, d_keep_aux) + wFunc_array[i_d]
        m_kink[i_d] = m_egm[0] # store kink point

        # Transformations for value funtion interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        vt_egm = vTransf(v_egm)

        # b) Compute non-decreasing segments
        start, end = calc_nondecreasing_segments(m_egm, vt_egm)

        # c) Get segments
        segments = []
        m_segments = []
        vt_segments = []
        c_segments = []

        for j in range(len(start)):
            idx = range(start[j], end[j] + 1)
            segments.append([m_egm[idx], vt_egm[idx]])
            m_segments.append(m_egm[idx])
            vt_segments.append((vt_egm[idx]))
            c_segments.append(c_egm[idx])

        # d) Upper envelope
        m_upper, v_upper, inds_upper = upper_envelope(segments)

        # e) Envelope Consumption
        c_env = np.zeros_like(m_upper) + np.nan

        for k, c_segm in enumerate(c_segments):
            c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])

        v_env = vUntransf(v_upper)
        inv_v_env = utility_inv(v_env, CRRA)

        c_for_interpolation = np.insert(c_env, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(m_upper, 0, BoroCnstNat, axis=-1)
        inv_v_for_interpolation = np.insert(inv_v_env, 0, 0, axis=-1)

        d_for_interpolation = np.ones_like(c_for_interpolation) * d_keep
        inv_vPFuncKeep_for_interpolation = CRRAutilityP_inv(c_for_interpolation, d_for_interpolation)
        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cFuncKeepList.append(LinearFast(c_for_interpolation, [m_for_interpolation]))
        inv_vFuncKeepList.append(LinearFast(inv_v_for_interpolation, [m_for_interpolation]))
        # vFuncKeepList.append(ValueFuncCRRA(LinearFast(inv_v_for_interpolation, [m_for_interpolation]),CRRA))
        inv_vPFuncKeepList.append(LinearFast(inv_vPFuncKeep_for_interpolation, [m_for_interpolation])) #MargValueFuncCRRA(LinearFast(c_for_interpolation, [m_for_interpolation]), CRRA))

    # Add kink point to mNrmGrid:
    mNrmGrid_New = np.insert(m_for_interpolation, 0, m_kink)
    mNrmGrid_New = np.unique(mNrmGrid_New)
    mNrmGrid_New.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGrid_New))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    # vPFuncKeep_array = np.zeros(keep_shape)
    inv_vFuncKeep_array = np.zeros(keep_shape)
    inv_vPFuncKeep_array = np.zeros(keep_shape)

    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d] = np.ones(len(mNrmGrid_New)) * nNrmGrid[i_d]
        for i_m in range(len(mNrmGrid)):
            cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGrid_New)
            #vFuncKeep_array[i_d] = vFuncKeepList[i_d](mNrmGrid_New)
            inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGrid_New)
            inv_vPFuncKeep_array[i_d] = inv_vPFuncKeepList[i_d](mNrmGrid_New)   #CRRAutilityP(cFuncKeep_array[i_d], d_keep)
    # vPFuncKeep_array = utilityP_inv(inv_vPFuncKeep_array, CRRA) #CRRAutilityP_inv(cFuncKeep_array[i_d], d_keep)

    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGrid_New": mNrmGrid_New,
            #"vFuncKeep_array": vFuncKeep_array,
            # "vPFuncKeep_array": vPFuncKeep_array,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
            }

########################################################################################################################
########################################################################################################################
def DCEGM_invVFunc_Latest(nNrmGrid, aNrmGrid, BoroCnstNat, qFunc, wFunc, alpha, d_ubar,
                         CRRA):  # , inv_wFunc_array, qFunc_array
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    # CRRAutilityP_inv = lambda C, D: utilityP_inv(CRRAutilityP(C, D), CRRA)
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # Create empty container
    m_kink = np.zeros_like(nNrmGrid)
    mNrmGrid_array = []#np.zeros(1)
    mNrmGridList = []
    cFuncKeepList = []
    # vFuncKeepList = []
    inv_vFuncKeepList = []
    inv_vPFuncKeepList = []

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # Create qFunc_arra and wFunc_array:
        qFunc_array = qFunc(d_keep_aux, aNrmGrid)
        wFunc_array = wFunc(d_keep_aux, aNrmGrid)

        # use euler equation
        # c_egm = utilityP_inv(qFunc_array, CRRA)
        c_egm = CRRAutilityP_inv(qFunc_array, d_keep_aux)

        # Consumption cannot be negative
        c_egm = np.clip(c_egm, 0, np.inf)# len(mNrmGrid))
        m_egm = aNrmGrid + c_egm
        v_egm = CRRAutility(c_egm, d_keep_aux) + wFunc_array
        m_kink[i_d] = m_egm[0]  # store kink point

        # If last point is smaller than the previous: get rid of it:
        if c_egm[-1] < c_egm[-2]:
            print("last point is smaller than the previous one")
            print(i_d)
            c_egm = c_egm[:-2]
            m_egm = m_egm[:-2]
            v_egm = v_egm[:-2]

        # Transformations for value funtion interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        vt_egm = vTransf(v_egm)

        # b) Compute non-decreasing segments
        start, end = calc_nondecreasing_segments(m_egm, vt_egm)

        # c) Get segments
        segments = []
        m_segments = []
        vt_segments = []
        c_segments = []
        w_segments = [] # New part

        for j in range(len(start)):
            idx = range(start[j], end[j] + 1)
            segments.append([m_egm[idx], vt_egm[idx]])
            m_segments.append(m_egm[idx])
            vt_segments.append((vt_egm[idx]))
            c_segments.append(c_egm[idx])
            w_segments.append(wFunc_array[idx]) # New part

        # d) Upper envelope
        m_upper, v_upper, inds_upper = upper_envelope(segments)

        # e) Envelope Consumption
        c_env = np.zeros_like(m_upper) + np.nan

        for k, c_segm in enumerate(c_segments):
            # Problem if last point is a decreasing segment. TODO
            if start[-1] + 1 == len(m_egm):
                print("Last point is lower")
                print(i_d)
                c_Interp = LinearInterp(m_segments[0], c_segments[0])
                c_env = c_Interp(m_upper)

            else:
                c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])
            # if len(c_segments[1]) == 1:
            #     c_env[inds_upper == k] = c_segm #(m_upper[inds_upper == k])
            # else:
            #     c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])
            #     # c_env[inds_upper == k] = LinearFast(c_segm, [m_segments[k]])(m_upper[inds_upper == k])

        # Problem: If lower part is at c = N-1, c_env has a negative slope between N-1 and N
        if start[-1] + 2 == len(m_egm):
            print("2nd to last is lower")
            #c_env = np.zeros_like(m_upper + 1) + np.nan
            c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])
            # Add additional point
            c_Interp = LinearInterp(m_egm[start[-1]:], c_egm[start[-1]:])
            m_upper_add = np.array([m_upper[-1], m_upper[-1] + 0.01])
            c_env_add = c_Interp(m_upper_add)
            m_upper = np.insert(m_upper, len(m_upper), m_upper_add[1], axis=-1)
            c_env = np.insert(c_env, len(c_env), c_env_add[1], axis=-1)

        # # new part start
        # w_env = np.zeros_like(m_upper) + np.nan
        #
        # for k, w_segm in enumerate(w_segments):
        #     w_env[inds_upper == k] = LinearInterp(m_segments[k], w_segm)(m_upper[inds_upper == k])
        #
        # # new part end
        v_env = vUntransf(v_upper)

        # v_env can get -inf if v_egm is too large negative. Replace it with v_egm value. Note that for d_aux = 0 all values are -inf.
        if v_env[0] == -np.inf and v_env[1]!= -np.inf:
            v_env[0] = v_egm[0]

        # # Alternative: create new v_env
        # if np.any(v_env == -np.inf):
        #     # print('here')
        #     v_env = CRRAutility(c_env, np.ones_like(c_env) * d_keep) + w_env

        inv_v_env = utility_inv(v_env, CRRA)

        c_for_interpolation = np.insert(c_env, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(m_upper, 0, BoroCnstNat, axis=-1)
        inv_v_for_interpolation = np.insert(inv_v_env, 0, 0, axis=-1)

        d_for_interpolation = np.ones_like(c_for_interpolation) * d_keep
        inv_vPFuncKeep_for_interpolation = CRRAutilityP_inv(c_for_interpolation, d_for_interpolation)
        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cFuncKeepList.append(LinearFast(c_for_interpolation, [m_for_interpolation]))
        inv_vFuncKeepList.append(LinearFast(inv_v_for_interpolation, [m_for_interpolation]))
        # vFuncKeepList.append(ValueFuncCRRA(LinearFast(inv_v_for_interpolation, [m_for_interpolation]),CRRA))
        inv_vPFuncKeepList.append(LinearFast(inv_vPFuncKeep_for_interpolation, [
            m_for_interpolation]))  # MargValueFuncCRRA(LinearFast(c_for_interpolation, [m_for_interpolation]), CRRA))
        mNrmGridList.append(m_for_interpolation)
        mNrmGrid_array = np.append(mNrmGrid_array, m_for_interpolation)

    # Option 1: Add nothing
    # mNrmGrid_New = m_for_interpolation

    # Option 2: Combine first and last mNrmGrid:
    mNrmGrid_red = np.hstack([mNrmGrid_array[0:len(aNrmGrid)+1], mNrmGrid_array[-len(aNrmGrid)-1:]])#m_for_interpolation #np.insert(m_for_interpolation, 0, m_kink[0])
    mNrmGrid_red = np.clip(mNrmGrid_red, BoroCnstNat, np.max(aNrmGrid))
    mNrmGrid_red = np.unique(mNrmGrid_red)
    mNrmGrid_red.sort()

    # Option 3: Add everything
    mNrmGrid_New = np.unique(mNrmGrid_array)
    # mNrmGrid_New = np.clip(mNrmGrid_New, BoroCnstNat, np.max(aNrmGrid)) # does not make a difference
    mNrmGrid_New = np.unique(mNrmGrid_New)
    mNrmGrid_New.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGrid_New))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    # vPFuncKeep_array = np.zeros(keep_shape)
    inv_vFuncKeep_array = np.zeros(keep_shape)
    inv_vPFuncKeep_array = np.zeros(keep_shape)

    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d] = np.ones(len(mNrmGrid_New)) * nNrmGrid[i_d]
        # for i_m in range(len(mNrmGrid_New)):
        cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGrid_New)
        # vFuncKeep_array[i_d] = vFuncKeepList[i_d](mNrmGrid_New)
        inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGrid_New)
        inv_vPFuncKeep_array[i_d] = inv_vPFuncKeepList[i_d](
            mNrmGrid_New)  # CRRAutilityP(cFuncKeep_array[i_d], d_keep)
    # vPFuncKeep_array = utilityP_inv(inv_vPFuncKeep_array, CRRA) #CRRAutilityP_inv(cFuncKeep_array[i_d], d_keep)

    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGrid_New": mNrmGrid_New,
            "mNrmGrid_red": mNrmGrid_red,
            # "vFuncKeep_array": vFuncKeep_array,
            # "vPFuncKeep_array": vPFuncKeep_array,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
            }

    # count = 0
    # for i_d in range(len(nNrmGrid)):
    #     print(count)
    #     print(np.min(cFuncKeep_array[i_d]))
    #     count = count + 1

    ### Points to test:
    # 91: 2nd to last is lower
    # 92- 99: last point is smaller than the previous one

    #
    ### TEST
    # exFuncKeepUnc = LinearFast(cFuncKeep_array + dFuncKeep_array, [nNrmGrid, mNrmGrid_New])
    # #grid = construct_grid(0, 10, 48, 'nonlinear', 3)
    # grid = construct_grid(BoroCnstNat, 20, 48, 'nonlinear', 3)
    #
    # Test_exFuncKeep = np.zeros(len(grid))
    # # Solve IndShockConsumer until line 962: cFuncNowUnc = interpolator(mNrm, cNrm)
    # # from DurableModel_Latest import construct_grid
    # # grid = construct_grid(construct_grid(mNrm[0], 10, 48, 'nonlinear', 3), 10, 48, 'nonlinear', 3)
    # # cFuncNowUnc(grid)
    #
    # Test_cFuncUnc = np.array([ 0.                ,  0.1542761166089551,  0.309958434521785 ,
    #     0.4670862047042164,  0.6257006110504474,  0.7858449098034986,
    #     0.947564582412552 ,  1.1109075034662066,  1.2759241255846951,
    #     1.4426676834415355,  1.6111944194247845,  1.7815638338511912,
    #     1.9538389631269177,  2.1280866898235016,  2.304378089329204 ,
    #     2.4827888185715343,  2.6633995533218653,  2.846296481832968 ,
    #     3.0315718640837006,  3.2193246677881873,  3.409661294670086 ,
    #     3.6026964134385944,  3.7985539196093034,  3.9973680470293163,
    #     4.199284662018809 ,  4.404462778880732 ,  4.613076345784866 ,
    #     4.825316363590133 ,  5.04139341830279  ,  5.261540732430442 ,
    #     5.486017874223608 ,  5.715115310827161 ,  5.94916005804001  ,
    #     6.188522775673808 ,  6.433626799464475 ,  6.68495981468885  ,
    #     6.943089208408591 ,  7.208682666734379 ,  7.482536458114644 ,
    #     7.765615347654629 ,  8.059110799791915 ,  8.364529306380224 ,
    #     8.683833291178864 ,  9.01968081994729  ,  9.37587024307031  ,
    #     9.75827349517861  , 10.177216571208213 , 10.656297637582316 ])
    # # Test_cFunc = np.zeros(len(grid))
    # for i in range(len(grid)):
    #     # Test_cFunc[i] = IndShockExample_life.solution[0].cFunc(grid[i])
    #     Test_exFuncKeep[i] = exFuncKeepUnc(
    #         Test_cFuncUnc[i] * (1 - alpha),
    #         grid[i] - Test_cFuncUnc[i] * (1 - alpha))
    #
    # Test_exFuncKeep - Test_cFuncUnc # mean error: 0.00042690580820613916
    # np.mean(Test_exFuncKeep - Test_cFuncUnc)
    # np.max(np.abs(Test_exFuncKeep - Test_cFuncUnc))

    ### TEST extrap_mode="linear"
    # inv_vFuncKeepUnc = LinearFast(inv_vFuncKeep_array, [nNrmGrid, mNrmGrid_New])
    # inv_vFuncKeepUnc(0.0, -0.0001) ## is positive
    # inv_vFuncKeepUnc = LinearFast(inv_vFuncKeep_array, [nNrmGrid, mNrmGrid_New], extrap_mode="constant")
    # inv_vFuncKeepUnc(0.0, -1.0) ## is positive
    ### TEST
    # exFuncKeepUnc = LinearFast(exFuncKeep_array, [nNrmGrid, mNrmGrid_New])
    # Grid = construct_grid(0, 10, 48, 'nonlinear', 3)
    # d_aux = np.zeros(len(Grid))
    # exFuncKeepUnc(d_aux, Grid)
    # inv_wFunc(d_aux, Grid)
    #
    # # "aNrmMin": 0.0,
    # # "aNrmMax": 11,  # xNrmMax+1.0
    # # "aNrmCount": 100,
    # aNrmGrid = construct_grid(0, 11, 100, 'nonlinear', 3)





    ### Test if inv_vFuncKeep_array is optimal for d = c
    ###
    # import scipy as sp
    # inv_vFuncKeepUnc = LinearFast(inv_vFuncKeep_array, [nNrmGrid, mNrmGrid_New])
    # cFuncKeepUnc = LinearFast(cFuncKeep_array, [nNrmGrid, mNrmGrid_New])
    # x = 1
    # d_low = np.fmin(x / 2, 1e-8)
    # d_high = np.fmax(x, max(nNrmGrid))
    #
    #
    # x0 = np.mean([d_high, d_low])
    # sol_opt = sp.optimize.minimize(lambda d: - inv_vFuncKeepUnc(d, x - d), x0, method='nelder-mead',
    #                            options={'fatol': 1e-15})
    # d = sol_opt.x
    # m = x - d  # This is correct, it is not: x - (1 - adjC) * dFuncAdj_array[i_x]
    # c = cFuncKeepUnc(d, m)
    # d - c
    #
    # ### How does inv_vFuncKeepUnc look like?!
    # d_grid = nNrmGrid
    # m_grid = np.ones(len(nNrmGrid)) - d_grid
    # minimize = - inv_vFuncKeepUnc(d_grid, m_grid)

########################################################################################################################
# def DCEGM_Latest(nNrmGrid, aNrmGrid, xNrmMax, BoroCnstNat, qFunc, wFunc, alpha, d_ubar,
#                          CRRA):  # , inv_wFunc_array, qFunc_array
#     # 1. Update utility functions:
#     # i. U(C,D)
#     u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
#     CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)
#
#     # ii. Inverse U(C, D)
#     CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)
#
#     # iii. uPC U(C,D) wrt C
#     CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))
#
#     # iv. Inverse uPC U(C,D) wrt C
#     CRRAutilityP_inv = lambda C, D: (
#                 (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))
#
#     # Create empty container
#     # m_kink = np.zeros_like(nNrmGrid)
#     mNrmGrid_array = []
#     # mNrmGridList = []
#     cFuncKeepList = []
#     inv_vFuncKeepList = []
#     inv_vPFuncKeepList = []
#
#     # Empty container to save EGM output:
#     shape = (len(nNrmGrid), len(aNrmGrid))
#     m_egm_array = np.zeros(shape)
#     c_egm_array = np.zeros(shape)
#     v_egm_array = np.zeros(shape)
#
#     for i_d in range(len(nNrmGrid)):
#         d_keep = nNrmGrid[i_d]
#         d_keep_aux = np.ones(len(aNrmGrid)) * d_keep
#
#         # Create qFunc_arra and wFunc_array:
#         qFunc_array = qFunc(d_keep_aux, aNrmGrid)
#         wFunc_array = wFunc(d_keep_aux, aNrmGrid)
#
#         # use euler equation
#         c_egm = CRRAutilityP_inv(qFunc_array, d_keep_aux)
#
#         # # Consumption cannot be negative
#         # c_egm = np.clip(c_egm, 0, np.inf)
#
#         # m_Grid and v_Grid
#         m_egm = aNrmGrid + c_egm
#         v_egm = CRRAutility(c_egm, d_keep_aux) + wFunc_array
#         # m_kink[i_d] = m_egm[0]  # store kink point
#
#         # Save EGM output to compare Upper Envelope algorithm
#         c_egm_array[i_d] = c_egm
#         m_egm_array[i_d] = m_egm
#         v_egm_array[i_d] = v_egm
#
#         # # If last point is smaller than the previous: get rid of it:
#         if c_egm[-1] < c_egm[-2]:
#             # # Add another point
#             # print("kink at last point: add another point")
#             # aNrmGrid_aux = np.max(aNrmGrid) + 5 #np.append(aNrmGrid, np.max(aNrmGrid) + 0.001)
#             #
#             # qFunc_aux = qFunc(d_keep, aNrmGrid_aux)
#             # wFunc_aux = wFunc(d_keep, aNrmGrid_aux)
#             #
#             # # use euler equation
#             # c_aux = CRRAutilityP_inv(qFunc_aux, d_keep)
#             #
#             # # m_Grid and v_Grid
#             # m_aux = aNrmGrid_aux + c_aux
#             # v_aux = CRRAutility(c_aux, d_keep) + wFunc_aux
#             #
#             # c_egm = np.append(c_egm, c_aux)
#             # m_egm = np.append(m_egm, m_aux)
#             # v_egm = np.append(v_egm, v_aux)
#             # print("kink at last point: discard")
#             # # print("last point is smaller than the previous one: re-adjust grid")
#             # print(i_d)
#             # print("Problem if np.max(xNrmGridNow) >= ", m_egm[-1])
#             if xNrmMax >= m_egm[-1]:
#                 print("Kink at last point which is within xNrmGridNow. Increase aNrmGridNow")
#             c_egm = c_egm[:-2]
#             m_egm = m_egm[:-2]
#             v_egm = v_egm[:-2]
#
#         # Transformations for value funtion interpolation
#         vTransf = lambda x: np.exp(x)
#         vUntransf = lambda x: np.log(x)
#
#         vt_egm = vTransf(v_egm)
#
#         # b) Compute non-decreasing segments
#         start, end = calc_nondecreasing_segments(m_egm, vt_egm)
#
#         # c) Get segments
#         segments = []
#         m_segments = []
#         vt_segments = []
#         c_segments = []
#
#         for j in range(len(start)):
#             idx = range(start[j], end[j] + 1)
#             segments.append([m_egm[idx], vt_egm[idx]])
#             m_segments.append(m_egm[idx])
#             vt_segments.append((vt_egm[idx]))
#             c_segments.append(c_egm[idx])
#
#         # d) Upper envelope
#         m_upper, v_upper, inds_upper = upper_envelope(segments)
#
#         # e) Envelope Consumption
#         c_env = np.zeros_like(m_upper) + np.nan
#
#         for k, c_segm in enumerate(c_segments):
#             # print(k)
#             # if len(m_segments[k]) == 1:
#             #     print('only a point')
#             # if len(c_segm) == 1:
#             #     print('only a point')
#             # if len(m_upper[inds_upper == k]) == 1:
#             #     print('only a point')
#             # if k == []:
#             #     print('here')
#             # if inds_upper == []:
#             #     print('here')
#             # Problem if last point is a decreasing segment. TODO
#             if start[-1] + 1 == len(m_egm):
#                 print("Last point is lower")
#                 print(i_d)
#                 c_Interp = LinearInterp(m_segments[0], c_segments[0])
#                 c_env = c_Interp(m_upper)
#             # If only one point
#             elif len(m_segments[k]) == 1:
#                 c_env[inds_upper == k] = c_segm
#             else:
#                 c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])
#
#         # # Last point again:
#         # if c_env[-1] < c_env[-2]:
#         #     c_env = c_env[:-2]
#         #     m_upper = m_upper[:-2]
#         #     v_upper = v_upper[:-2]
#         # Problem: If lower part is at c = N-1, c_env has a negative slope between N-1 and N
#         # if start[-1] + 2 == len(m_egm):
#         #     print("2nd to last is lower")
#         #     #c_env = np.zeros_like(m_upper + 1) + np.nan
#         #     c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])
#         #     # Add additional point
#         #     c_Interp = LinearInterp(m_egm[start[-1]:], c_egm[start[-1]:])
#         #     m_upper_add = np.array([m_upper[-1], m_upper[-1] + 0.01])
#         #     c_env_add = c_Interp(m_upper_add)
#         #     m_upper = np.insert(m_upper, len(m_upper), m_upper_add[1], axis=-1)
#         #     c_env = np.insert(c_env, len(c_env), c_env_add[1], axis=-1)
#
#         v_env = vUntransf(v_upper)
#
#         # v_env can get -inf if v_egm is too large negative. Replace it with v_egm value. Note that for d_aux = 0 all values are -inf.
#         if v_env[0] == -np.inf and v_env[1]!= -np.inf:
#             v_env[0] = v_egm[0]
#
#         inv_v_env = utility_inv(v_env, CRRA)
#
#         c_for_interpolation = np.insert(c_env, 0, 0.0, axis=-1)
#         m_for_interpolation = np.insert(m_upper, 0, BoroCnstNat, axis=-1)
#         inv_v_for_interpolation = np.insert(inv_v_env, 0, 0, axis=-1)
#
#         # If durables = 0: Consume everything, but value is still zero
#         if d_keep == 0 and alpha<1:
#             c_for_interpolation = m_for_interpolation - BoroCnstNat
#             inv_v_for_interpolation = np.zeros(len(m_for_interpolation))
#
#         d_for_interpolation = np.ones_like(c_for_interpolation) * d_keep
#         inv_vPFuncKeep_for_interpolation = CRRAutilityP_inv(c_for_interpolation, d_for_interpolation)
#         # Create cFuncs and vFuncs for each i_d given the correct mGrid
#         cFuncKeepList.append(LinearFast(c_for_interpolation, [m_for_interpolation]))
#         inv_vFuncKeepList.append(LinearFast(inv_v_for_interpolation, [m_for_interpolation]))
#         inv_vPFuncKeepList.append(LinearFast(inv_vPFuncKeep_for_interpolation, [
#             m_for_interpolation]))  # MargValueFuncCRRA(LinearFast(c_for_interpolation, [m_for_interpolation]), CRRA))
#         # mNrmGridList.append(m_for_interpolation)
#         mNrmGrid_array = np.append(mNrmGrid_array, m_for_interpolation)
#
#     # Option 1: Add nothing
#     # mNrmGrid_New = m_for_interpolation
#
#     # # Option 2: Combine first and last mNrmGrid:
#     # mNrmGrid_red = np.hstack([mNrmGrid_array[0:len(aNrmGrid)+1], mNrmGrid_array[-len(aNrmGrid)-1:]])#m_for_interpolation #np.insert(m_for_interpolation, 0, m_kink[0])
#     # mNrmGrid_red = np.clip(mNrmGrid_red, BoroCnstNat, 30) #np.max(aNrmGrid))
#     # mNrmGrid_red = np.unique(mNrmGrid_red)
#     # mNrmGrid_red.sort()
#
#     # Option 3: Add everything
#     mNrmGridKeeper = np.unique(mNrmGrid_array)
#     mNrmGridKeeper = np.unique(mNrmGridKeeper)
#     mNrmGridKeeper.sort()
#
#     # Create c and vFunc
#     # Create empty container
#     keep_shape = (len(nNrmGrid), len(mNrmGridKeeper))
#     cFuncKeep_array = np.zeros(keep_shape)
#     dFuncKeep_array = np.zeros(keep_shape)
#     inv_vFuncKeep_array = np.zeros(keep_shape)
#     inv_vPFuncKeep_array = np.zeros(keep_shape)
#
#     for i_d in range(len(nNrmGrid)):
#         dFuncKeep_array[i_d] = np.ones(len(mNrmGridKeeper)) * nNrmGrid[i_d]
#         if nNrmGrid[i_d] == 0 and alpha < 1:  # If durables = 0: Consume everything, but value is still zero
#             cFuncKeep_array[i_d] = mNrmGridKeeper - BoroCnstNat
#             inv_vFuncKeep_array[i_d] = np.zeros(len(mNrmGridKeeper))
#             inv_vPFuncKeep_array[i_d] = np.ones(len(mNrmGridKeeper)) * np.inf
#         else:
#             cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGridKeeper)
#             inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGridKeeper)
#             inv_vPFuncKeep_array[i_d] = inv_vPFuncKeepList[i_d](
#             mNrmGridKeeper)
#
#     # Check results:
#     if np.min(cFuncKeep_array) < 0:
#         print("cFuncKeep_array is negative")
#     # for i_d in range(len(nNrmGrid)):
#     #     if np.min(cFuncKeep_array[i_d]) < 0:
#     #         print(i_d)
#     return {"cFuncKeep_array": cFuncKeep_array,
#             "dFuncKeep_array": dFuncKeep_array,
#             "mNrmGridKeeper": mNrmGridKeeper,
#             # "mNrmGrid_red": mNrmGrid_red,
#             "inv_vFuncKeep_array": inv_vFuncKeep_array,
#             "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
#             "c_egm_array": c_egm_array,
#             "m_egm_array": m_egm_array,
#             "v_egm_array": v_egm_array,
#             }
#
#     # for i_d in range(len(nNrmGrid)):
#     #     if np.min(cFuncKeep_array[i_d]) < 0:
#     #         print(i_d)
def DCEGM_Latest(nNrmGrid, aNrmGrid, BoroCnstNat, qFunc, wFunc, alpha, d_ubar, CRRA):  # , inv_wFunc_array, qFunc_array
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # Create empty container
    # m_kink = np.zeros_like(nNrmGrid)
    mNrmGrid_array = []
    # mNrmGridList = []
    cFuncKeepList = []
    inv_vFuncKeepList = []
    inv_vPFuncKeepList = []

    # Empty container to save EGM output:
    shape = (len(nNrmGrid), len(aNrmGrid))
    m_egm_array = np.zeros(shape)
    c_egm_array = np.zeros(shape)
    v_egm_array = np.zeros(shape)

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # Create qFunc_arra and wFunc_array:
        qFunc_array = qFunc(d_keep_aux, aNrmGrid)
        wFunc_array = wFunc(d_keep_aux, aNrmGrid)

        # use euler equation
        c_egm = CRRAutilityP_inv(qFunc_array, d_keep_aux)

        # m_Grid and v_Grid
        m_egm = aNrmGrid + c_egm
        v_egm = CRRAutility(c_egm, d_keep_aux) + wFunc_array

        # Save EGM output to compare Upper Envelope algorithm
        c_egm_array[i_d] = c_egm
        m_egm_array[i_d] = m_egm
        v_egm_array[i_d] = v_egm

        # If last point is smaller than the previous
        if c_egm[-1] < c_egm[-2]:
            print("Kink at last point: Increase aNrmGrid")

        # Transformations for value function interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        vt_egm = vTransf(v_egm)

        # b) Compute non-decreasing segments
        start, end = calc_nondecreasing_segments(m_egm, vt_egm)

        # c) Get segments
        segments = []
        m_segments = []
        vt_segments = []
        c_segments = []

        for j in range(len(start)):
            idx = range(start[j], end[j] + 1)
            segments.append([m_egm[idx], vt_egm[idx]])
            m_segments.append(m_egm[idx])
            vt_segments.append((vt_egm[idx]))
            c_segments.append(c_egm[idx])

        # d) Upper envelope
        m_upper, v_upper, inds_upper = upper_envelope(segments)

        # e) Envelope Consumption
        c_env = np.zeros_like(m_upper) + np.nan

        for k, c_segm in enumerate(c_segments):
            # Problem if last point is a decreasing segment. TODO
            if start[-1] + 1 == len(m_egm):
                print("Last point is lower")
                print(i_d)
                # c_Interp = LinearFast(c_segments[0], [m_segments[0]])
                # c_env = c_Interp(m_upper)
            # If only one point
            elif len(m_segments[k]) == 1:
                c_env[inds_upper == k] = c_segm
            else:
                c_env[inds_upper == k] = LinearFast(c_segm, [m_segments[k]])(m_upper[inds_upper == k])

        v_env = vUntransf(v_upper)

        # v_env can get -inf if v_egm is too large negative. Replace it with v_egm value. Note that for d_aux = 0 all values are -inf.
        if v_env[0] == -np.inf and v_env[1]!= -np.inf:
            v_env[0] = v_egm[0]

        inv_v_env = utility_inv(v_env, CRRA)

        c_for_interpolation = np.insert(c_env, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(m_upper, 0, BoroCnstNat, axis=-1)
        inv_v_for_interpolation = np.insert(inv_v_env, 0, 0, axis=-1)

        # If durables = 0: Consume everything, but value is still zero
        if d_keep == 0 and alpha<1:
            c_for_interpolation = m_for_interpolation - BoroCnstNat
            inv_v_for_interpolation = np.zeros(len(m_for_interpolation))

        d_for_interpolation = np.ones_like(c_for_interpolation) * d_keep
        inv_vPFuncKeep_for_interpolation = CRRAutilityP_inv(c_for_interpolation, d_for_interpolation)
        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cFuncKeepList.append(LinearFast(c_for_interpolation, [m_for_interpolation]))
        inv_vFuncKeepList.append(LinearFast(inv_v_for_interpolation, [m_for_interpolation]))
        inv_vPFuncKeepList.append(LinearFast(inv_vPFuncKeep_for_interpolation, [
            m_for_interpolation]))  # MargValueFuncCRRA(LinearFast(c_for_interpolation, [m_for_interpolation]), CRRA))
        mNrmGrid_array = np.append(mNrmGrid_array, m_for_interpolation)

    # Overall mNrmGrid
    mNrmGridKeeper = np.unique(mNrmGrid_array)
    mNrmGridKeeper = np.unique(mNrmGridKeeper)
    mNrmGridKeeper.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGridKeeper))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    inv_vFuncKeep_array = np.zeros(keep_shape)
    inv_vPFuncKeep_array = np.zeros(keep_shape)

    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d] = np.ones(len(mNrmGridKeeper)) * nNrmGrid[i_d]
        if nNrmGrid[i_d] == 0 and alpha < 1:  # If durables = 0: Consume everything, but value is still zero
            cFuncKeep_array[i_d] = mNrmGridKeeper - BoroCnstNat
            inv_vFuncKeep_array[i_d] = np.zeros(len(mNrmGridKeeper))
            inv_vPFuncKeep_array[i_d] = np.ones(len(mNrmGridKeeper)) * np.inf
        else:
            cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGridKeeper)
            inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGridKeeper)
            inv_vPFuncKeep_array[i_d] = inv_vPFuncKeepList[i_d](
            mNrmGridKeeper)

    # Check results:
    if np.min(cFuncKeep_array) < 0:
        print("cFuncKeep_array is negative")
    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGridKeeper": mNrmGridKeeper,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
            "c_egm_array": c_egm_array,
            "m_egm_array": m_egm_array,
            "v_egm_array": v_egm_array,
            }

@njit("(float64[:])(float64[:], float64)", cache=True)
def utility_njit(c, gam):
    """
    Evaluates constant relative risk aversion (CRRA) utility of consumption c
    given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Utility

    Tests
    -----
    Test a value which should pass:
    >>> c, gamma = 1.0, 2.0    # Set two values at once with Python syntax
    >>> utility(c=c, gam=gamma)
    -1.0
    """

    if gam == 1:
        return np.log(c)
    else:
        return c ** (1.0 - gam) / (1.0 - gam)

@njit("(float64[:])(float64[:], float64[:], float64, float64, float64)", cache=True)
def CRRAutilityP_inv_njit(c, d, alpha, d_ubar, CRRA):
    power = 1 / ((alpha * (1 - CRRA) - 1))
    return (c / (alpha * (d + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** power


@njit("(float64[:](float64[:], float64[:], float64, float64, float64))", cache=True)
def CRRAutility_njit(c, d, alpha, d_ubar, CRRA):
    c_prime = np.power(c, alpha)
    d_prime = np.power(d + d_ubar, 1 - alpha)
    u_inner = c_prime * d_prime
    return (utility_njit(u_inner, CRRA))


@njit(
    "Tuple((float64[:,:], float64[:,:], float64[:,:]))(float64[:], float64[:], float64[:,:], float64[:,:], float64, float64, float64)",
    cache=True)
def EGM_njit(nNrmGrid, aNrmGrid, qFunc_array, wFunc_array, alpha, d_ubar, CRRA):
    # Empty container to save EGM output:
    shape = (len(nNrmGrid), len(aNrmGrid))
    m_egm_array = np.zeros(shape)
    c_egm_array = np.zeros(shape)
    v_egm_array = np.zeros(shape)

    for i_d, d_keep in enumerate(nNrmGrid):
        d_keep_aux = np.ones(len(aNrmGrid), dtype=np.float64) * d_keep

        # Create qFunc_arra and wFunc_array:
        qFuncNow = qFunc_array[i_d]
        wFuncNow = wFunc_array[i_d]

        # use euler equation
        c_egm = CRRAutilityP_inv_njit(qFuncNow, d_keep_aux, alpha, d_ubar, CRRA)
        # m_Grid and v_Grid
        m_egm = aNrmGrid + c_egm
        v_egm = CRRAutility_njit(c_egm, d_keep_aux, alpha, d_ubar, CRRA) + wFuncNow

        # Save EGM output to compare Upper Envelope algorithm
        c_egm_array[i_d] = c_egm
        m_egm_array[i_d] = m_egm
        v_egm_array[i_d] = v_egm

        # # If last point is smaller than the previous:
        if c_egm[-1] < c_egm[-2]:
            print("Kink at last point: Increase aNrmGrid")

    return c_egm_array, m_egm_array, v_egm_array


def UpperEnvelope_njit(nNrmGrid, aNrmGrid, BoroCnstNat, qFunc, wFunc, alpha, d_ubar, CRRA, c_egm_array, m_egm_array,
                  v_egm_array):  # , inv_wFunc_array, qFunc_array
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    CRRAutilityP_inv = lambda C, D: (
            (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # Create empty container
    mNrmGrid_array = []
    cFuncKeepList = []
    inv_vFuncKeepList = []
    inv_vPFuncKeepList = []

    for i_d in range(len(nNrmGrid)):
        c_egm = c_egm_array[i_d]
        m_egm = m_egm_array[i_d]
        v_egm = v_egm_array[i_d]

        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # Transformations for value funtion interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        vt_egm = vTransf(v_egm)

        # b) Compute non-decreasing segments
        start, end = calc_nondecreasing_segments(m_egm, vt_egm)

        # c) Get segments
        segments = []
        m_segments = []
        vt_segments = []
        c_segments = []

        for j in range(len(start)):
            idx = range(start[j], end[j] + 1)
            segments.append([m_egm[idx], vt_egm[idx]])
            m_segments.append(m_egm[idx])
            vt_segments.append((vt_egm[idx]))
            c_segments.append(c_egm[idx])

        # d) Upper envelope
        m_upper, v_upper, inds_upper = upper_envelope(segments)

        # e) Envelope Consumption
        c_env = np.zeros_like(m_upper) + np.nan

        for k, c_segm in enumerate(c_segments):
            # Problem if last point is a decreasing segment. TODO
            if start[-1] + 1 == len(m_egm):
                print("Last point is lower")
                print(i_d)
                c_Interp = LinearFast(c_segments[0], [m_segments[0]])
                c_env = c_Interp(m_upper)
            # If only one point
            elif len(m_segments[k]) == 1:
                c_env[inds_upper == k] = c_segm
            else:
                c_env[inds_upper == k] = LinearFast(c_segm, [m_segments[k]])(m_upper[inds_upper == k])

        v_env = vUntransf(v_upper)

        # v_env can get -inf if v_egm is too large negative. Replace it with v_egm value. Note that for d_aux = 0 all values are -inf.
        if v_env[0] == -np.inf and v_env[1] != -np.inf:
            v_env[0] = v_egm[0]

        inv_v_env = utility_inv(v_env, CRRA)

        c_for_interpolation = np.insert(c_env, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(m_upper, 0, BoroCnstNat, axis=-1)
        inv_v_for_interpolation = np.insert(inv_v_env, 0, 0, axis=-1)

        d_for_interpolation = np.ones_like(c_for_interpolation) * d_keep
        inv_vPFuncKeep_for_interpolation = CRRAutilityP_inv(c_for_interpolation, d_for_interpolation)
        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cFuncKeepList.append(LinearFast(c_for_interpolation, [m_for_interpolation]))
        inv_vFuncKeepList.append(LinearFast(inv_v_for_interpolation, [m_for_interpolation]))
        inv_vPFuncKeepList.append(LinearFast(inv_vPFuncKeep_for_interpolation, [
            m_for_interpolation]))  # MargValueFuncCRRA(LinearFast(c_for_interpolation, [m_for_interpolation]), CRRA))
        mNrmGrid_array = np.append(mNrmGrid_array, m_for_interpolation)

    # Overall mNrmGrid
    mNrmGridKeeper = np.unique(mNrmGrid_array)
    mNrmGridKeeper = np.unique(mNrmGridKeeper)
    mNrmGridKeeper.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGridKeeper))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    inv_vFuncKeep_array = np.zeros(keep_shape)
    inv_vPFuncKeep_array = np.zeros(keep_shape)

    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d] = np.ones(len(mNrmGridKeeper)) * nNrmGrid[i_d]
        if nNrmGrid[i_d] == 0 and alpha < 1:  # If durables = 0: Consume everything, but value is still zero
            cFuncKeep_array[i_d] = mNrmGridKeeper - BoroCnstNat
            inv_vFuncKeep_array[i_d] = np.zeros(len(mNrmGridKeeper))
            inv_vPFuncKeep_array[i_d] = np.ones(len(mNrmGridKeeper)) * np.inf
        else:
            cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGridKeeper)
            inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGridKeeper)
            inv_vPFuncKeep_array[i_d] = inv_vPFuncKeepList[i_d](
                mNrmGridKeeper)

    # Check results:
    if np.min(cFuncKeep_array) < 0:
        print("cFuncKeep_array is negative")
    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGridKeeper": mNrmGridKeeper,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
            }


def optimization_problem_with_guess(x, initial_guess, function):
    sol_opt = sp_optimize.minimize(lambda d: -function(d, x - d), initial_guess, method='nelder-mead',
                                   options={'fatol': 1e-15})
    initial_guess = sol_opt.x  # Update x0 with the current optimal solution

    return sol_opt.x

@njit("Tuple((float64[:], float64[:,:]))(float64[:], float64[:], float64[:,:], float64[:,:], int32)", cache=True)
def durable_adjusting_function_fast(nNrmGrid, mNrmGrid, inv_vFuncAdj_array, inv_vFuncKeep_array, tol):
    """
    This function evaluates the region in the S,s-model where the agent should adjust.
    The structure of an S,s model is that we have a lower and an upper bound. If the agent is below or above it: adjust
    and keep inbetween the bound.
    First, the points are detected where the inverse value function of the keeper is larger than the adjuster.
    Second, the exact point is evaluated assuming the inverse value function is linear between the grid points.

    Parameters
    ----------
    nNrmGrid: Grid over the 2nd state variable
    mNrmGrid: Grid over the 1st state variable
    inv_vFuncAdj_array: inverse value Function of the adjuster Problem over 1st and 2nd state variable
    inv_vFuncKeep_array: inverse value Function of the keeper Problem over 1st and 2nd state variable
    tol: Integer: Small value for which keeper inverse value Function has to be higher in order to trigger bound.

    Returns
    -------
    mNrmGrid_Total: New Grid of 1st state variable
    lSuS_array: Array of values for mNrmGrid_Total to indicate the bounds
    """
    shape = (len(nNrmGrid), 2)
    lSuS_array = np.zeros(shape)
    mNrmGrid_Total = np.copy(mNrmGrid)
    for i_d in range(len(nNrmGrid)):
        if i_d == 0:
            lSuS_array[i_d] = [0.0, 0.0]
            continue
        inv_vFunc_Diff = inv_vFuncAdj_array[i_d] - inv_vFuncKeep_array[i_d]
        # Find the lower and upper threshold where consumer keeps
        lS = np.where(inv_vFunc_Diff < - tol)[0]
        lS = 0 if lS.size == 0 else np.min(lS)
        uS = np.where(inv_vFunc_Diff[lS:] > - tol)[0]
        uS = len(mNrmGrid) if uS.size == 0 else np.min(uS) + lS  # take min if exists

        # Find the exact point
        if uS == 0:
            lS_exact = mNrmGrid_Total[0]
        else:
            lS_x = np.array([mNrmGrid[np.maximum(lS - 1, 0)], mNrmGrid[lS]])
            lS_left_y = np.array([inv_vFuncAdj_array[i_d][np.maximum(lS - 1, 0)],
                                  inv_vFuncKeep_array[i_d][np.maximum(lS - 1, 0)]])
            lS_right_y = np.array([inv_vFuncAdj_array[i_d][lS], inv_vFuncKeep_array[i_d][lS]])
            lS_exact = calc_linear_crossing(lS_x, lS_left_y, lS_right_y)[0]

        if uS == len(mNrmGrid):
            uS_exact = np.max(mNrmGrid)
        elif uS == 0:
            uS_exact = mNrmGrid_Total[0]
        else:
            uS_x = np.array([mNrmGrid[uS - 1], mNrmGrid[uS]])
            uS_left_y = np.array([inv_vFuncAdj_array[i_d][uS - 1], inv_vFuncKeep_array[i_d][uS - 1]])
            uS_right_y = np.array([inv_vFuncAdj_array[i_d][uS], inv_vFuncKeep_array[i_d][uS]])
            uS_exact = calc_linear_crossing(uS_x, uS_left_y, uS_right_y)[0]
        lSuS_array[i_d] = [lS_exact, uS_exact]
        added_values = np.array([lS_exact, uS_exact])
        mNrmGridNew = np.zeros(len(added_values) + len(mNrmGrid_Total))
        mNrmGridNew[:len(added_values)] = added_values
        mNrmGridNew[len(added_values):] = mNrmGrid_Total
        mNrmGrid_Total = np.sort(np.unique(mNrmGridNew))

        if np.isnan(lSuS_array).any():
            print('Nans detected')
    return mNrmGrid_Total, lSuS_array


@njit(
    "Tuple((float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:]))(float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])",
    cache=True)
def durable_solution_function_fast(nNrmGridNow, mNrmGrid_Total, lSuS_array, adjusting_array,
                                   cFuncAdj_NM_array, dFuncAdj_NM_array,
                                   inv_vFuncAdj_NM_array,
                                   cFuncKeep_NM_array, dFuncKeep_NM_array,
                                   inv_vFuncKeep_NM_array):
    cFunc_array = cFuncAdj_NM_array
    dFunc_array = dFuncAdj_NM_array
    inv_vFunc_array = inv_vFuncAdj_NM_array

    for i_d in range(len(nNrmGridNow)):
        lS_idx = np.where(mNrmGrid_Total == lSuS_array[i_d][0])[0][0]
        uS_idx = np.where(mNrmGrid_Total == lSuS_array[i_d][1])[0][
            0]  # IndexError: index 0 is out of bounds for axis 0 with size 0
        if uS_idx == len(mNrmGrid_Total) - 1:
            uS_idx = uS_idx + 1
        length = uS_idx - lS_idx - 1
        if length < 0:
            length = 0

        cFunc_array[i_d][lS_idx + 1: uS_idx] = cFuncKeep_NM_array[i_d][lS_idx + 1: uS_idx]
        dFunc_array[i_d][lS_idx + 1: uS_idx] = dFuncKeep_NM_array[i_d][lS_idx + 1: uS_idx]

        # if kink at last point (primary or secondary)
        if cFunc_array[i_d, -1] < cFunc_array[i_d, -2]:
            print(i_d)
            print('cFunc becomes a decreasing function')

        inv_vFunc_array[i_d][lS_idx + 1: uS_idx] = inv_vFuncKeep_NM_array[i_d][lS_idx + 1: uS_idx]

        adjusting_array[i_d, lS_idx + 1:uS_idx] = np.zeros(length)
    exFunc_array = cFunc_array + dFunc_array

    return cFunc_array, dFunc_array, exFunc_array, inv_vFunc_array, adjusting_array

########################################################################################################################
def DCEGM_Latest_NewValueFunction(nNrmGrid, aNrmGrid, BoroCnstNat, qFunc, wFunc, alpha, d_ubar,
                         CRRA):  # , inv_wFunc_array, qFunc_array
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # Create empty container
    m_kink = np.zeros_like(nNrmGrid)
    mNrmGrid_array = []
    mNrmGridList = []
    cFuncKeepList = []
    inv_vFuncKeepList = []
    inv_vPFuncKeepList = []

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # Create qFunc_arra and wFunc_array:
        qFunc_array = qFunc(d_keep_aux, aNrmGrid)
        wFunc_array = wFunc(d_keep_aux, aNrmGrid)

        # use euler equation
        c_egm = CRRAutilityP_inv(qFunc_array, d_keep_aux)

        # Consumption cannot be negative
        c_egm = np.clip(c_egm, 0, np.inf)# len(mNrmGrid))
        m_egm = aNrmGrid + c_egm

        ### New value function based on HARK make_EndOfPrdvFunc
        EndOfPrdvNvrs = utility_inv(wFunc_array, CRRA)
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        aNrm_temp = np.insert(aNrmGrid, 0, BoroCnstNat)
        EndOfPrdvNvrsFunc = LinearInterp(aNrm_temp, EndOfPrdvNvrs)
        EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, CRRA)

        # v_egm = CRRAutility(c_egm, d_keep_aux) + wFunc_array
        m_kink[i_d] = m_egm[0]  # store kink point

        # If last point is smaller than the previous: get rid of it:
        if c_egm[-1] < c_egm[-2]:
            print("last point is smaller than the previous one")
            print(i_d)
            c_egm = c_egm[:-2]
            m_egm = m_egm[:-2]
            v_egm = v_egm[:-2]

        # Transformations for value funtion interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        vt_egm = vTransf(v_egm)

        # b) Compute non-decreasing segments
        start, end = calc_nondecreasing_segments(m_egm, vt_egm)

        # c) Get segments
        segments = []
        m_segments = []
        vt_segments = []
        c_segments = []

        for j in range(len(start)):
            idx = range(start[j], end[j] + 1)
            segments.append([m_egm[idx], vt_egm[idx]])
            m_segments.append(m_egm[idx])
            vt_segments.append((vt_egm[idx]))
            c_segments.append(c_egm[idx])

        # d) Upper envelope
        m_upper, v_upper, inds_upper = upper_envelope(segments)

        # e) Envelope Consumption
        c_env = np.zeros_like(m_upper) + np.nan

        for k, c_segm in enumerate(c_segments):
            # Problem if last point is a decreasing segment. TODO
            if start[-1] + 1 == len(m_egm):
                print("Last point is lower")
                print(i_d)
                c_Interp = LinearInterp(m_segments[0], c_segments[0])
                c_env = c_Interp(m_upper)

            else:
                c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])

        # Problem: If lower part is at c = N-1, c_env has a negative slope between N-1 and N
        if start[-1] + 2 == len(m_egm):
            print("2nd to last is lower")
            #c_env = np.zeros_like(m_upper + 1) + np.nan
            c_env[inds_upper == k] = LinearInterp(m_segments[k], c_segm)(m_upper[inds_upper == k])
            # Add additional point
            c_Interp = LinearInterp(m_egm[start[-1]:], c_egm[start[-1]:])
            m_upper_add = np.array([m_upper[-1], m_upper[-1] + 0.01])
            c_env_add = c_Interp(m_upper_add)
            m_upper = np.insert(m_upper, len(m_upper), m_upper_add[1], axis=-1)
            c_env = np.insert(c_env, len(c_env), c_env_add[1], axis=-1)

        v_env = vUntransf(v_upper)

        # v_env can get -inf if v_egm is too large negative. Replace it with v_egm value. Note that for d_aux = 0 all values are -inf.
        if v_env[0] == -np.inf and v_env[1]!= -np.inf:
            v_env[0] = v_egm[0]

        inv_v_env = utility_inv(v_env, CRRA)

        c_for_interpolation = np.insert(c_env, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(m_upper, 0, BoroCnstNat, axis=-1)
        inv_v_for_interpolation = np.insert(inv_v_env, 0, 0, axis=-1)

        d_for_interpolation = np.ones_like(c_for_interpolation) * d_keep
        inv_vPFuncKeep_for_interpolation = CRRAutilityP_inv(c_for_interpolation, d_for_interpolation)
        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cFuncKeepList.append(LinearFast(c_for_interpolation, [m_for_interpolation]))
        inv_vFuncKeepList.append(LinearFast(inv_v_for_interpolation, [m_for_interpolation]))
        inv_vPFuncKeepList.append(LinearFast(inv_vPFuncKeep_for_interpolation, [
            m_for_interpolation]))  # MargValueFuncCRRA(LinearFast(c_for_interpolation, [m_for_interpolation]), CRRA))
        mNrmGridList.append(m_for_interpolation)
        mNrmGrid_array = np.append(mNrmGrid_array, m_for_interpolation)

    # Option 1: Add nothing
    # mNrmGrid_New = m_for_interpolation

    # Option 2: Combine first and last mNrmGrid:
    mNrmGrid_red = np.hstack([mNrmGrid_array[0:len(aNrmGrid)+1], mNrmGrid_array[-len(aNrmGrid)-1:]])#m_for_interpolation #np.insert(m_for_interpolation, 0, m_kink[0])
    # mNrmGrid_red = np.clip(mNrmGrid_red, BoroCnstNat, np.max(aNrmGrid))
    mNrmGrid_red = np.unique(mNrmGrid_red)
    mNrmGrid_red.sort()

    # Option 3: Add everything
    mNrmGrid_New = np.unique(mNrmGrid_array)
    mNrmGrid_New = np.unique(mNrmGrid_New)
    mNrmGrid_New.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGrid_New))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    inv_vFuncKeep_array = np.zeros(keep_shape)
    inv_vPFuncKeep_array = np.zeros(keep_shape)

    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d] = np.ones(len(mNrmGrid_New)) * nNrmGrid[i_d]
        cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGrid_New)
        inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGrid_New)
        inv_vPFuncKeep_array[i_d] = inv_vPFuncKeepList[i_d](
            mNrmGrid_New)

    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGrid_New": mNrmGrid_New,
            "mNrmGrid_red": mNrmGrid_red,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
            }

########################################################################################################################
def FUES_EGM(nNrmGrid, aNrmGrid, BoroCnstNat, qFunc, wFunc, alpha, d_ubar,
                         CRRA):  # , inv_wFunc_array, qFunc_array
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    # CRRAutilityP_inv = lambda C, D: utilityP_inv(CRRAutilityP(C, D), CRRA)
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # Create empty container
    mNrmGrid_array = []
    mNrmGridList = []
    cFuncKeepList = []
    inv_vFuncKeepList = []
    inv_vPFuncKeepList = []

    # Empty container to save EGM output:
    shape = (len(nNrmGrid), len(aNrmGrid))
    m_egm_array = np.zeros(shape)
    c_egm_array = np.zeros(shape)
    v_egm_array = np.zeros(shape)

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # Create qFunc_arra and wFunc_array:
        qFunc_array = qFunc(d_keep_aux, aNrmGrid)
        wFunc_array = wFunc(d_keep_aux, aNrmGrid)

        # use euler equation
        c_egm = CRRAutilityP_inv(qFunc_array, d_keep_aux)

        # Consumption cannot be negative
        c_egm = np.clip(c_egm, 0, np.inf)
        m_egm = aNrmGrid + c_egm
        v_egm = CRRAutility(c_egm, d_keep_aux) + wFunc_array

        # Save EGM output to compare Upper Envelope algorithm
        c_egm_array[i_d] = c_egm
        m_egm_array[i_d] = m_egm
        v_egm_array[i_d] = v_egm

        # Transformations for value funtion interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        vt_egm = vTransf(v_egm)

        # Alternative
        # vt2_egm = utility_inv(v_egm, CRRA)



        ### FUES PART STARTS HERE
        m_upper, v_upper, c_env, a_upper, dela \
            = FUES(m_egm, vt_egm, c_egm, aNrmGrid) #, M_bar=2, LB=10)
        ### FUES PART ENDS HERE

        v_env = vUntransf(v_upper)
        # v_env can get -inf if v_egm is too large negative. Replace it with v_egm value. Note that for d_aux = 0 all values are -inf.
        if v_env[0] == -np.inf and v_env[1]!= -np.inf:
            v_env[0] = v_egm[0]
        inv_v_env = utility_inv(v_env, CRRA)

        # Alternative
        # v_env = utility_inv(v_upper, CRRA)
        # inv_v_env = v_upper

        c_for_interpolation = np.insert(c_env, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(m_upper, 0, BoroCnstNat, axis=-1)
        inv_v_for_interpolation = np.insert(inv_v_env, 0, 0, axis=-1)

        d_for_interpolation = np.ones_like(c_for_interpolation) * d_keep
        inv_vPFuncKeep_for_interpolation = CRRAutilityP_inv(c_for_interpolation, d_for_interpolation)
        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cFuncKeepList.append(LinearFast(c_for_interpolation, [m_for_interpolation]))
        inv_vFuncKeepList.append(LinearFast(inv_v_for_interpolation, [m_for_interpolation]))
        # vFuncKeepList.append(ValueFuncCRRA(LinearFast(inv_v_for_interpolation, [m_for_interpolation]),CRRA))
        inv_vPFuncKeepList.append(LinearFast(inv_vPFuncKeep_for_interpolation, [
            m_for_interpolation]))  # MargValueFuncCRRA(LinearFast(c_for_interpolation, [m_for_interpolation]), CRRA))
        mNrmGridList.append(m_for_interpolation)
        mNrmGrid_array = np.append(mNrmGrid_array, m_for_interpolation)

    # Option 1: Add nothing
    # mNrmGrid_New = m_for_interpolation

    # Option 2: Combine first and last mNrmGrid:
    mNrmGrid_red = np.hstack([mNrmGrid_array[0:len(aNrmGrid)+1], mNrmGrid_array[-len(aNrmGrid)-1:]])#m_for_interpolation #np.insert(m_for_interpolation, 0, m_kink[0])
    mNrmGrid_red = np.clip(mNrmGrid_red, BoroCnstNat, np.max(aNrmGrid))
    mNrmGrid_red = np.unique(mNrmGrid_red)
    mNrmGrid_red.sort()

    # Option 3: Add everything
    mNrmGrid_New = np.unique(mNrmGrid_array)
    # mNrmGrid_New = np.clip(mNrmGrid_New, BoroCnstNat, np.max(aNrmGrid)) # does not make a difference
    mNrmGrid_New = np.unique(mNrmGrid_New)
    mNrmGrid_New.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGrid_New))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    inv_vFuncKeep_array = np.zeros(keep_shape)
    inv_vPFuncKeep_array = np.zeros(keep_shape)

    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d] = np.ones(len(mNrmGrid_New)) * nNrmGrid[i_d]
        cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGrid_New)
        inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGrid_New)
        inv_vPFuncKeep_array[i_d] = inv_vPFuncKeepList[i_d](
            mNrmGrid_New)

    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGrid_New": mNrmGrid_New,
            "mNrmGrid_red": mNrmGrid_red,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
            "c_egm_array": c_egm_array,
            "m_egm_array": m_egm_array,
            "v_egm_array": v_egm_array,
            }

######################################################################################
def FUES_EGM_Latest(nNrmGrid, aNrmGrid, BoroCnstNat, qFunc, wFunc, alpha, d_ubar,
                         CRRA):  # , inv_wFunc_array, qFunc_array
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # Create empty container
    # m_kink = np.zeros_like(nNrmGrid)
    mNrmGrid_array = []
    # mNrmGridList = []
    cFuncKeepList = []
    inv_vFuncKeepList = []
    inv_vPFuncKeepList = []

    # Empty container to save EGM output:
    shape = (len(nNrmGrid), len(aNrmGrid))
    m_egm_array = np.zeros(shape)
    c_egm_array = np.zeros(shape)
    v_egm_array = np.zeros(shape)

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # Create qFunc_arra and wFunc_array:
        qFunc_array = qFunc(d_keep_aux, aNrmGrid)
        wFunc_array = wFunc(d_keep_aux, aNrmGrid)

        # use euler equation
        c_egm = CRRAutilityP_inv(qFunc_array, d_keep_aux)

        # # Consumption cannot be negative
        # c_egm = np.clip(c_egm, 0, np.inf)

        # m_Grid and v_Grid
        m_egm = aNrmGrid + c_egm
        v_egm = CRRAutility(c_egm, d_keep_aux) + wFunc_array
        # m_kink[i_d] = m_egm[0]  # store kink point

        # Save EGM output to compare Upper Envelope algorithm
        c_egm_array[i_d] = c_egm
        m_egm_array[i_d] = m_egm
        v_egm_array[i_d] = v_egm

        # # If last point is smaller than the previous: get rid of it:
        if c_egm[-1] < c_egm[-2]:
            print("kink at last point: discarded")
            # print("last point is smaller than the previous one: re-adjust grid")
            print(i_d)
            print(m_egm[-1])
            c_egm = c_egm[:-2]
            m_egm = m_egm[:-2]
            v_egm = v_egm[:-2]

        # Transformations for value funtion interpolation
        vTransf = lambda x: np.exp(x)
        vUntransf = lambda x: np.log(x)

        vt_egm = vTransf(v_egm)

        ### FUES PART STARTS HERE
        m_upper, v_upper, c_env, a_upper, dela \
            = FUES(m_egm, vt_egm, c_egm, aNrmGrid) #, M_bar=2, LB=10)
        ### FUES PART ENDS HERE

        v_env = vUntransf(v_upper)

        # v_env can get -inf if v_egm is too large negative. Replace it with v_egm value. Note that for d_aux = 0 all values are -inf.
        if v_env[0] == -np.inf and v_env[1]!= -np.inf:
            v_env[0] = v_egm[0]

        inv_v_env = utility_inv(v_env, CRRA)

        c_for_interpolation = np.insert(c_env, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(m_upper, 0, BoroCnstNat, axis=-1)
        inv_v_for_interpolation = np.insert(inv_v_env, 0, 0, axis=-1)

        # If durables = 0: Consume everything, but value is still zero
        if d_keep == 0 and alpha<1:
            c_for_interpolation = m_for_interpolation - BoroCnstNat
            inv_v_for_interpolation = np.zeros(len(m_for_interpolation))

        d_for_interpolation = np.ones_like(c_for_interpolation) * d_keep
        inv_vPFuncKeep_for_interpolation = CRRAutilityP_inv(c_for_interpolation, d_for_interpolation)
        # Create cFuncs and vFuncs for each i_d given the correct mGrid
        cFuncKeepList.append(LinearFast(c_for_interpolation, [m_for_interpolation]))
        inv_vFuncKeepList.append(LinearFast(inv_v_for_interpolation, [m_for_interpolation]))
        inv_vPFuncKeepList.append(LinearFast(inv_vPFuncKeep_for_interpolation, [
            m_for_interpolation]))  # MargValueFuncCRRA(LinearFast(c_for_interpolation, [m_for_interpolation]), CRRA))
        # mNrmGridList.append(m_for_interpolation)
        mNrmGrid_array = np.append(mNrmGrid_array, m_for_interpolation)

    # Option 1: Add nothing
    # mNrmGrid_New = m_for_interpolation

    # # Option 2: Combine first and last mNrmGrid:
    # mNrmGrid_red = np.hstack([mNrmGrid_array[0:len(aNrmGrid)+1], mNrmGrid_array[-len(aNrmGrid)-1:]])#m_for_interpolation #np.insert(m_for_interpolation, 0, m_kink[0])
    # mNrmGrid_red = np.clip(mNrmGrid_red, BoroCnstNat, 30) #np.max(aNrmGrid))
    # mNrmGrid_red = np.unique(mNrmGrid_red)
    # mNrmGrid_red.sort()

    # Option 3: Add everything
    mNrmGridKeeper = np.unique(mNrmGrid_array)
    mNrmGridKeeper = np.unique(mNrmGridKeeper)
    mNrmGridKeeper.sort()

    # Create c and vFunc
    # Create empty container
    keep_shape = (len(nNrmGrid), len(mNrmGridKeeper))
    cFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)
    inv_vFuncKeep_array = np.zeros(keep_shape)
    inv_vPFuncKeep_array = np.zeros(keep_shape)

    for i_d in range(len(nNrmGrid)):
        dFuncKeep_array[i_d] = np.ones(len(mNrmGridKeeper)) * nNrmGrid[i_d]
        if nNrmGrid[i_d] == 0 and alpha < 1:  # If durables = 0: Consume everything, but value is still zero
            cFuncKeep_array[i_d] = mNrmGridKeeper - BoroCnstNat
            inv_vFuncKeep_array[i_d] = np.zeros(len(mNrmGridKeeper))
            inv_vPFuncKeep_array[i_d] = np.ones(len(mNrmGridKeeper)) * np.inf
        else:
            cFuncKeep_array[i_d] = cFuncKeepList[i_d](mNrmGridKeeper)
            inv_vFuncKeep_array[i_d] = inv_vFuncKeepList[i_d](mNrmGridKeeper)
            inv_vPFuncKeep_array[i_d] = inv_vPFuncKeepList[i_d](
            mNrmGridKeeper)

    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGridKeeper": mNrmGridKeeper,
            # "mNrmGrid_red": mNrmGrid_red,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
            "c_egm_array": c_egm_array,
            "m_egm_array": m_egm_array,
            "v_egm_array": v_egm_array,
            }

    # for i_d in range(len(nNrmGrid)):
    #     if np.min(cFuncKeep_array[i_d]) < 0:
    #         print(i_d)

########################################################################################################################
def Upper_Envelope_EGM_Comparison(nNrmGrid, aNrmGrid, mNrmGrid, BoroCnstNat, qFunc, wFunc, alpha, d_ubar,
                         CRRA):  # , inv_wFunc_array, qFunc_array
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    # CRRAutilityP_inv = lambda C, D: utilityP_inv(CRRAutilityP(C, D), CRRA)
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # Create empty container
    shape = (len(nNrmGrid), len(aNrmGrid))
    m_egm_array = np.zeros(shape)
    c_egm_array = np.zeros(shape)
    v_egm_array = np.zeros(shape)

    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep

        # Create qFunc_arra and wFunc_array:
        qFunc_array = qFunc(d_keep_aux, aNrmGrid)
        wFunc_array = wFunc(d_keep_aux, aNrmGrid)

        # use euler equation
        c_egm = CRRAutilityP_inv(qFunc_array, d_keep_aux)

        # Consumption cannot be negative
        c_egm_array[i_d,:] = np.clip(c_egm, 0, np.inf)
        m_egm_array[i_d,:] = aNrmGrid + c_egm
        v_egm_array[i_d,:] = CRRAutility(c_egm, d_keep_aux) + wFunc_array


    return {"c_egm_array": c_egm_array,
            "m_egm_array": m_egm_array,
            "v_egm_array": v_egm_array,
            }

########################################################################################################################
def Upper_Envelope_Jeppe(nNrmGrid, aNrmGrid, mNrmGrid, qFunc, inv_wFunc_array, alpha, d_ubar, CRRA):
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    # CRRAutilityP_inv = lambda C, D: utilityP_inv(CRRAutilityP(C, D), CRRA)
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    keep_shape = (len(nNrmGrid), len(mNrmGrid))
    vPFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)

    post_shape = (len(nNrmGrid), len(aNrmGrid))
    cFuncKeep_array = np.zeros(post_shape)
    q_c = np.nan * np.zeros(post_shape)
    q_m = np.nan * np.zeros(post_shape)

    v_ast_vec = np.zeros(post_shape)
    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]
        d_keep_aux = np.ones(len(aNrmGrid)) * d_keep  # Creating a vector to use for interpolation
        # use euler equation
        q_c[i_d] = utilityP_inv(qFunc(d_keep_aux, aNrmGrid),
                                CRRA)  # CRRAutilityP_inv(qFunc(d_keep_aux, aNrmGrid), d_keep_aux)
        q_m[i_d] = aNrmGrid + q_c[i_d]

        # upperenvelope
        negm_upperenvelope(aNrmGrid, q_m[i_d], q_c[i_d], inv_wFunc_array[i_d],
                           mNrmGrid, cFuncKeep_array[i_d], v_ast_vec[i_d], d_keep, d_ubar, alpha,
                           CRRA)  # wFunc_array[i_d],

        # negative inverse
        vPFuncKeep_array[i_d] = CRRAutilityP_inv(cFuncKeep_array[i_d], d_keep)

        dFuncKeep_array[i_d] = nNrmGrid[i_d]

    vFuncKeep_array = v_ast_vec  # utility_inv(v_ast_vec, CRRA)
    # Creating Inverses
    inv_vFuncKeep_array = utility_inv(vFuncKeep_array, CRRA)
    inv_vPFuncKeep_array = utilityP_inv(vPFuncKeep_array, CRRA)

    return {"cFuncKeep_array": cFuncKeep_array,
            "dFuncKeep_array": dFuncKeep_array,
            "mNrmGrid_New": mNrmGrid,
            "mNrmGrid_red": mNrmGrid,
            # "vFuncKeep_array": vFuncKeep_array,
            # "vPFuncKeep_array": vPFuncKeep_array,
            "inv_vFuncKeep_array": inv_vFuncKeep_array,
            "inv_vPFuncKeep_array": inv_vPFuncKeep_array,
            }
# class MargValueFuncCRRA(MetricObject):
#     """
#     A class for representing a marginal value function in models where the
#     standard envelope condition of dvdm(state) = u'(c(state)) holds (with CRRA utility).
#
#     Parameters
#     ----------
#     cFunc : function.
#         Its first argument must be normalized market resources m.
#         A real function representing the marginal value function composed
#         with the inverse marginal utility function, defined on the state
#         variables: uP_inv(dvdmFunc(state)).  Called cFunc because when standard
#         envelope condition applies, uP_inv(dvdm(state)) = cFunc(state).
#     CRRA : float
#         Coefficient of relative risk aversion.
#     """
#
#     distance_criteria = ["cFunc", "dFunc", "CRRA"]
#
#     def __init__(self, cFunc, dFunc, CRRA):
#         self.cFunc = deepcopy(cFunc)
#         self.dFunc = deepcopy(dFunc)
#         self.CRRA = CRRA
#
#         if hasattr(cFunc, 'grid_list'):
#             self.grid_list = cFunc.grid_list
#         else:
#             self.grid_list = None
#
#         if hasattr(dFunc, 'grid_list'):
#             self.grid_list = dFunc.grid_list
#         else:
#             self.grid_list = None
#
#     def __call__(self, *cFuncArgs):
#         """
#         Evaluate the marginal value function at given levels of market resources m.
#
#         Parameters
#         ----------
#         cFuncArgs : floats or np.arrays
#             Values of the state variables at which to evaluate the marginal
#             value function.
#
#         Returns
#         -------
#         vP : float or np.array
#             Marginal lifetime value of beginning this period with state
#             cFuncArgs
#         """
#         return CRRAutilityP(self.cFunc(*cFuncArgs), gam=self.CRRA)
#
#     def derivativeX(self, *cFuncArgs):
#         """
#         Evaluate the derivative of the marginal value function with respect to
#         market resources at given state; this is the marginal marginal value
#         function.
#
#         Parameters
#         ----------
#         cFuncArgs : floats or np.arrays
#             State variables.
#
#         Returns
#         -------
#         vPP : float or np.array
#             Marginal marginal lifetime value of beginning this period with
#             state cFuncArgs; has same size as inputs.
#
#         """
#
#         # The derivative method depends on the dimension of the function
#         if isinstance(self.cFunc, (HARKinterpolator1D)):
#             c, MPC = self.cFunc.eval_with_derivative(*cFuncArgs)
#
#         elif hasattr(self.cFunc, "derivativeX"):
#             c = self.cFunc(*cFuncArgs)
#             MPC = self.cFunc.derivativeX(*cFuncArgs)
#
#         else:
#             raise Exception(
#                 "cFunc does not have a 'derivativeX' attribute. Can't compute"
#                 + "marginal marginal value."
#             )
#
#         return MPC * CRRAutilityPP(c, gam=self.CRRA)



# def m_nrm_next(self, shocks, a_nrm, Rfree):
#     """
#     Computes normalized market resources of the next period
#     from income shocks and current normalized market resources.
#
#     Parameters
#     ----------
#     shocks: [float]
#         Permanent and transitory income shock levels.
#     a_nrm: float
#         Normalized market assets this period
#
#     Returns
#     -------
#     float
#        normalized market resources in the next period
#     """
#     return Rfree / (self.PermGroFac * shocks[0]) * a_nrm + shocks[1]
#
# def vp_next(shocks, a_nrm, Rfree, CRRA, vPfuncNext):
#     return shocks[0] ** (-CRRA) * vPfuncNext(
#         m_nrm_next(shocks, a_nrm, Rfree)
#     )



### Define A function which calculates consumption, value and adjusting function. This will come handy later.
def Graphfunctions(AgentType, grid, n):
    grid_x = (1 - AgentType.adjC)*n + grid
    # Consumption Functions
    cFunc_test = np.zeros(len(grid))
    dFunc_test = np.zeros(len(grid))
    exFunc_test = np.zeros(len(grid))
    cFuncKeep_test = np.zeros(len(grid))
    dFuncKeep_test = np.zeros(len(grid))
    exFuncKeep_test = np.zeros(len(grid))
    cFuncAdj_test = np.zeros(len(grid))
    dFuncAdj_test = np.zeros(len(grid))
    exFuncAdj_test = np.zeros(len(grid))

    # Value Functions
    vFunc_test = np.zeros(len(grid))
    vFuncKeep_test = np.zeros(len(grid))
    vFuncAdj_test = np.zeros(len(grid))

    # Adjuster Function
    adjusting = np.zeros(len(grid))

    for i in range(len(grid)):
        cFunc_test[i] = AgentType.solution[0].cFunc(n,grid[i])
        dFunc_test[i] = AgentType.solution[0].dFunc(n,grid[i])
        exFunc_test[i] = AgentType.solution[0].exFunc(n,grid[i])
        cFuncKeep_test[i] = AgentType.solution[0].cFuncKeep(n,grid[i])
        dFuncKeep_test[i] = AgentType.solution[0].dFuncKeep(n,grid[i])
        exFuncKeep_test[i] = AgentType.solution[0].exFuncKeep(n,grid[i])
        cFuncAdj_test[i] = AgentType.solution[0].cFuncAdj(grid_x[i])
        dFuncAdj_test[i] = AgentType.solution[0].dFuncAdj(grid_x[i])
        exFuncAdj_test[i] = AgentType.solution[0].exFuncAdj(grid_x[i])
        # Value Functions
        vFunc_test[i] = AgentType.solution[0].vFunc(n,grid[i])
        vFuncKeep_test[i] = AgentType.solution[0].vFuncKeep(n,grid[i])
        vFuncAdj_test[i] = AgentType.solution[0].vFuncAdj(grid_x[i])
        adjusting[i] = AgentType.solution[0].adjusting(n,grid[i])

    return {"cFunc_test": cFunc_test,
            "dFunc_test": dFunc_test,
            "exFunc_test": exFunc_test,
            "cFuncKeep_test": cFuncKeep_test,
            "dFuncKeep_test": dFuncKeep_test,
            "exFuncKeep_test": exFuncKeep_test,
            "cFuncAdj_test": cFuncAdj_test,
            "dFuncAdj_test": dFuncAdj_test,
            "exFuncAdj_test": exFuncAdj_test,
            "vFunc_test": vFunc_test,
            "vFuncKeep_test":vFuncKeep_test,
            "vFuncAdj_test":vFuncAdj_test,
            "adjusting":adjusting,}

### Define A function which calculates consumption, value and adjusting function. This will come handy later.
def Graphfunctions_lastPeriod(AgentType, grid, n):
    grid_x = (1 - AgentType.adjC)*n + grid
    # Consumption Functions
    cFunc_test = np.zeros(len(grid))
    dFunc_test = np.zeros(len(grid))
    exFunc_test = np.zeros(len(grid))
    cFuncKeep_test = np.zeros(len(grid))
    dFuncKeep_test = np.zeros(len(grid))
    exFuncKeep_test = np.zeros(len(grid))
    cFuncAdj_test = np.zeros(len(grid))
    dFuncAdj_test = np.zeros(len(grid))
    exFuncAdj_test = np.zeros(len(grid))

    # Value Functions
    vFunc_test = np.zeros(len(grid))
    vFuncKeep_test = np.zeros(len(grid))
    vFuncAdj_test = np.zeros(len(grid))

    # Adjuster Function
    adjusting = np.zeros(len(grid))

    for i in range(len(grid)):
        cFunc_test[i] = AgentType.solution[1].cFunc(n,grid[i])
        dFunc_test[i] = AgentType.solution[1].dFunc(n,grid[i])
        exFunc_test[i] = AgentType.solution[1].exFunc(n,grid[i])
        cFuncKeep_test[i] = AgentType.solution[1].cFuncKeep(n,grid[i])
        dFuncKeep_test[i] = AgentType.solution[1].dFuncKeep(n,grid[i])
        exFuncKeep_test[i] = AgentType.solution[1].exFuncKeep(n,grid[i])
        cFuncAdj_test[i] = AgentType.solution[1].cFuncAdj(grid_x[i])
        dFuncAdj_test[i] = AgentType.solution[1].dFuncAdj(grid_x[i])
        exFuncAdj_test[i] = AgentType.solution[1].exFuncAdj(grid_x[i])
        # Value Functions
        vFunc_test[i] = AgentType.solution[1].vFunc(n,grid[i])
        vFuncKeep_test[i] = AgentType.solution[1].vFuncKeep(n,grid[i])
        vFuncAdj_test[i] = AgentType.solution[1].vFuncAdj(grid_x[i])
        adjusting[i] = AgentType.solution[1].adjusting(n,grid[i])

    return {"cFunc_test": cFunc_test,
            "dFunc_test": dFunc_test,
            "exFunc_test": exFunc_test,
            "cFuncKeep_test": cFuncKeep_test,
            "dFuncKeep_test": dFuncKeep_test,
            "exFuncKeep_test": exFuncKeep_test,
            "cFuncAdj_test": cFuncAdj_test,
            "dFuncAdj_test": dFuncAdj_test,
            "exFuncAdj_test": exFuncAdj_test,
            "vFunc_test": vFunc_test,
            "vFuncKeep_test":vFuncKeep_test,
            "vFuncAdj_test":vFuncAdj_test,
            "adjusting":adjusting,}


def Graphfunctions_nm(AgentType, grid, n):
    #grid_x = (1 - AgentType.adjC)*n + grid
    # Consumption Functions
    cFunc_test = np.zeros(len(grid))
    dFunc_test = np.zeros(len(grid))
    exFunc_test = np.zeros(len(grid))
    cFuncKeep_test = np.zeros(len(grid))
    dFuncKeep_test = np.zeros(len(grid))
    exFuncKeep_test = np.zeros(len(grid))
    cFuncAdj_test = np.zeros(len(grid))
    dFuncAdj_test = np.zeros(len(grid))
    exFuncAdj_test = np.zeros(len(grid))

    # Value Functions
    vFunc_test = np.zeros(len(grid))
    vFuncKeep_test = np.zeros(len(grid))
    vFuncAdj_test = np.zeros(len(grid))

    # Adjuster Function
    adjusting = np.zeros(len(grid))

    for i in range(len(grid)):
        cFunc_test[i] = AgentType.solution[0].cFunc(n,grid[i])
        dFunc_test[i] = AgentType.solution[0].dFunc(n,grid[i])
        exFunc_test[i] = AgentType.solution[0].exFunc(n,grid[i])
        cFuncKeep_test[i] = AgentType.solution[0].cFuncKeep(n,grid[i])
        dFuncKeep_test[i] = AgentType.solution[0].dFuncKeep(n,grid[i])
        exFuncKeep_test[i] = AgentType.solution[0].exFuncKeep(n,grid[i])
        cFuncAdj_test[i] = AgentType.solution[0].cFuncAdj(n,grid[i])
        dFuncAdj_test[i] = AgentType.solution[0].dFuncAdj(n,grid[i])
        exFuncAdj_test[i] = AgentType.solution[0].exFuncAdj(n,grid[i])
        # Value Functions
        vFunc_test[i] = AgentType.solution[0].vFunc(n,grid[i])
        vFuncKeep_test[i] = AgentType.solution[0].vFuncKeep(n,grid[i])
        vFuncAdj_test[i] = AgentType.solution[0].vFuncAdj(n,grid[i])
        adjusting[i] = AgentType.solution[0].adjusting(n,grid[i])

    return {"cFunc_test": cFunc_test,
            "dFunc_test": dFunc_test,
            "exFunc_test": exFunc_test,
            "cFuncKeep_test": cFuncKeep_test,
            "dFuncKeep_test": dFuncKeep_test,
            "exFuncKeep_test": exFuncKeep_test,
            "cFuncAdj_test": cFuncAdj_test,
            "dFuncAdj_test": dFuncAdj_test,
            "exFuncAdj_test": exFuncAdj_test,
            "vFunc_test": vFunc_test,
            "vFuncKeep_test":vFuncKeep_test,
            "vFuncAdj_test":vFuncAdj_test,
            "adjusting":adjusting,}


from HARK.core import MetricObject
from copy import deepcopy

def CRRAutility_dur(c, d, alpha, d_ubar, gam):
    """
    Evaluates constant relative risk aversion (CRRA) utility of consumption c
    given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Utility

    Tests
    -----
    # Test a value which should pass:
    # >>> c, gamma = 1.0, 2.0    # Set two values at once with Python syntax
    # >>> utility(c=c, gam=gamma)
    # -1.0
    """

    # if gam == 1:
    #     return np.log(c)
    # else:
    #     return c ** (1.0 - gam) / (1.0 - gam)

    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)

    return utility(u_inner(c, d, d_ubar, alpha), gam)

def CRRAutilityP_dur(c, d, alpha, d_ubar, gam):
    """
    Evaluates constant relative risk aversion (CRRA) marginal utility of consumption
    c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal utility
    """
    # CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # if gam == 1:
    #     return 1 / c
    return ((alpha * c ** (alpha * (1 - gam) - 1)) * (d + d_ubar) ** ((1 - alpha) * (1 - gam)))

class MargValueFuncCRRA_dur(MetricObject):
    """
    A class for representing a marginal value function in models where the
    standard envelope condition of dvdm(state) = u'(c(state)) holds (with CRRA utility).

    Parameters
    ----------
    cFunc : function.
        Its first argument must be normalized market resources m.
        A real function representing the marginal value function composed
        with the inverse marginal utility function, defined on the state
        variables: uP_inv(dvdmFunc(state)).  Called cFunc because when standard
        envelope condition applies, uP_inv(dvdm(state)) = cFunc(state).
    CRRA : float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["cFunc", "CRRA"] # ["cFunc", "dFunc", "CRRA", "alpha", "d_ubar"]

    def __init__(self, cFunc, dFunc, alpha, d_ubar, CRRA):
        self.cFunc = deepcopy(cFunc)
        self.dFunc = deepcopy(dFunc)
        self.CRRA = CRRA
        self.alpha = alpha
        self.d_ubar = d_ubar

        if hasattr(cFunc, 'grid_list'):
            self.grid_list = cFunc.grid_list
        else:
            self.grid_list = None

    def __call__(self, *cFuncArgs):
        """
        Evaluate the marginal value function at given levels of durable stock n and market resources m.

        Parameters
        ----------
        cFuncArgs : floats or np.arrays
            Values of the state variables at which to evaluate the marginal
            value function.
        Returns
        -------
        vP : float or np.array
            Marginal lifetime value of beginning this period with state
            cFuncArgs
        """
        return CRRAutilityP_dur(self.cFunc(*cFuncArgs), self.dFunc(*cFuncArgs), alpha = self.alpha, d_ubar = self.d_ubar, gam=self.CRRA)

    # def derivativeX(self, *cFuncArgs):
    #     """
    #     Evaluate the derivative of the marginal value function with respect to
    #     market resources at given state; this is the marginal marginal value
    #     function.
    #
    #     Parameters
    #     ----------
    #     cFuncArgs : floats or np.arrays
    #         State variables.
    #
    #     Returns
    #     -------
    #     vPP : float or np.array
    #         Marginal marginal lifetime value of beginning this period with
    #         state cFuncArgs; has same size as inputs.
    #
    #     """
    #
    #     # The derivative method depends on the dimension of the function
    #     if isinstance(self.cFunc, (HARKinterpolator1D)):
    #         c, MPC = self.cFunc.eval_with_derivative(*cFuncArgs)
    #
    #     elif hasattr(self.cFunc, "derivativeX"):
    #         c = self.cFunc(*cFuncArgs)
    #         MPC = self.cFunc.derivativeX(*cFuncArgs)
    #
    #     else:
    #         raise Exception(
    #             "cFunc does not have a 'derivativeX' attribute. Can't compute"
    #             + "marginal marginal value."
    #         )
    #
    #     return MPC * CRRAutilityPP(c, gam=self.CRRA)


class ValueFuncCRRA_dur(MetricObject):
    """
    A class for representing a value function.  The underlying interpolation is
    in the space of (state,u_inv(v)); this class "re-curves" to the value function.

    Parameters
    ----------
    vFuncNvrs : function
        A real function representing the value function composed with the
        inverse utility function, defined on the state: u_inv(vFunc(state))
    CRRA : float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["func", "CRRA"]

    def __init__(self, vFuncNvrs, CRRA, alpha, d_ubar):
        self.vFuncNvrs = deepcopy(vFuncNvrs)
        self.CRRA = CRRA
        self.alpha = alpha
        self.d_ubar = d_ubar

        if hasattr(vFuncNvrs, 'grid_list'):
            self.grid_list = vFuncNvrs.grid_list
        else:
            self.grid_list = None

    def __call__(self, *vFuncArgs):
        """
        Evaluate the value function at given levels of market resources m.

        Parameters
        ----------
        vFuncArgs : floats or np.arrays, all of the same dimensions.
            Values for the state variables. These usually start with 'm',
            market resources normalized by the level of permanent income.

        Returns
        -------
        v : float or np.array
            Lifetime value of beginning this period with the given states; has
            same size as the state inputs.
        """
        #        return CRRAutility(self.func(*vFuncArgs), gam=self.CRRA)
        return CRRAutility_dur(self.vFuncNvrs(*vFuncArgs), alpha=self.alpha, d_ubar=self.d_ubar, gam=self.CRRA)

    def gradient(self, *args):

        NvrsGrad = self.vFuncNvrs.gradient(*args)
        grad = [CRRAutilityP_dur(g, self.CRRA) for g in NvrsGrad]
        # CRRAutilityP_dur(self.cFunc(*cFuncArgs), self.dFunc(*cFuncArgs), alpha=self.alpha, d_ubar=self.d_ubar,
        #                  gam=self.CRRA)
        return grad

    def _eval_and_grad(self, *args):

        return (self.__call__(*args), self.gradient(*args))

def durable_adjusting_function_m(nNrmGrid, mNrmGrid, inv_vFuncAdj_array, inv_vFuncKeep_array, tol):
    """
    This function evaluates the region in the S,s-model where the agent should adjust.
    The structure of an S,s model is that we have a lower and an upper bound. If the agent is below or above it: adjust
    and keep inbetween the bound.
    First, the points are detected where the inverse value function of the keeper is larger than the adjuster.
    Second, the exact point is evaluated assuming the inverse value function is linear between the grid points.

    Parameters
    ----------
    nNrmGrid: Grid over the 1nd state variable
    mNrmGrid: Grid over the 2st state variable
    inv_vFuncAdj_array: inverse value Function of the adjuster Problem over 1st and 2nd state variable
    inv_vFuncKeep_array: inverse value Function of the keeper Problem over 1st and 2nd state variable
    tol: Integer: Small value for which keeper inverse value Function has to be higher in order to trigger bound.

    Returns
    -------
    nNrmGrid_Total: New Grid of 1st state variable
    lSuS_array: Array of values for nNrmGrid_Total to indicate the bounds
    """
    shape = (len(mNrmGrid), 2)
    lSuS_array = np.zeros(shape)
    nNrmGrid_Total = np.copy(nNrmGrid)
    for i_m in range(len(mNrmGrid)):
        inv_vFunc_Diff = inv_vFuncAdj_array[:,i_m] - inv_vFuncKeep_array[:,i_m]
        # Find the lower and upper threshold where consumer keeps
        lS = np.where(inv_vFunc_Diff < - tol)[0]
        lS = 0 if lS.size == 0 else np.min(lS)
        uS = np.where(inv_vFunc_Diff[lS:] > - tol)[0]
        uS = len(nNrmGrid) if uS.size == 0 else np.min(uS) + lS  # take min if exists

        # Find the exact point
        if uS == 0:
            lS_exact = 0
        else:
            lS_x = np.array([nNrmGrid[np.maximum(lS - 1, 0)], nNrmGrid[lS]])
            lS_left_y = np.array([inv_vFuncAdj_array[np.maximum(lS - 1, 0)][i_m],
                                  inv_vFuncKeep_array[np.maximum(lS - 1, 0)][i_m]])
            lS_right_y = np.array([inv_vFuncAdj_array[lS][i_m], inv_vFuncKeep_array[lS][i_m]])
            lS_exact = calc_linear_crossing(lS_x, lS_left_y, lS_right_y)[0]

        if uS == len(nNrmGrid):
            uS_exact = np.max(nNrmGrid)
        elif uS == 0:
            uS_exact = 0
        else:
            uS_x = np.array([nNrmGrid[uS - 1], nNrmGrid[uS]])
            uS_left_y = np.array([inv_vFuncAdj_array[uS - 1][i_m], inv_vFuncKeep_array[uS - 1][i_m]])
            uS_right_y = np.array([inv_vFuncAdj_array[uS][i_m], inv_vFuncKeep_array[uS][i_m]])
            uS_exact = calc_linear_crossing(uS_x, uS_left_y, uS_right_y)[0]

        lSuS_array[i_m] = [lS_exact, uS_exact]
        added_values = [lS_exact, uS_exact]
        added_values = np.concatenate((added_values, nNrmGrid_Total))
        nNrmGrid_Total = np.sort(np.unique(added_values))

    return nNrmGrid_Total, lSuS_array



def durable_adjusting_function(nNrmGrid, mNrmGrid, inv_vFuncAdj_array, inv_vFuncKeep_array, tol):
    """
    This function evaluates the region in the S,s-model where the agent should adjust.
    The structure of an S,s model is that we have a lower and an upper bound. If the agent is below or above it: adjust
    and keep inbetween the bound.
    First, the points are detected where the inverse value function of the keeper is larger than the adjuster.
    Second, the exact point is evaluated assuming the inverse value function is linear between the grid points.

    Parameters
    ----------
    nNrmGrid: Grid over the 2nd state variable
    mNrmGrid: Grid over the 1st state variable
    inv_vFuncAdj_array: inverse value Function of the adjuster Problem over 1st and 2nd state variable
    inv_vFuncKeep_array: inverse value Function of the keeper Problem over 1st and 2nd state variable
    tol: Integer: Small value for which keeper inverse value Function has to be higher in order to trigger bound.

    Returns
    -------
    mNrmGrid_Total: New Grid of 1st state variable
    lSuS_array: Array of values for mNrmGrid_Total to indicate the bounds
    """
    shape = (len(nNrmGrid), 2)
    lSuS_array = np.zeros(shape)
    mNrmGrid_Total = np.copy(mNrmGrid)
    for i_d in range(len(nNrmGrid)):
        inv_vFunc_Diff = inv_vFuncAdj_array[i_d] - inv_vFuncKeep_array[i_d]
        # Find the lower and upper threshold where consumer keeps
        lS = np.where(inv_vFunc_Diff < - tol)[0]
        lS = 0 if lS.size == 0 else np.min(lS)
        uS = np.where(inv_vFunc_Diff[lS:] > - tol)[0]
        uS = len(mNrmGrid) if uS.size == 0 else np.min(uS) + lS  # take min if exists

        # Find the exact point
        if uS == 0:
            lS_exact = mNrmGrid_Total[0]
        else:
            lS_x = np.array([mNrmGrid[np.maximum(lS - 1, 0)], mNrmGrid[lS]])
            lS_left_y = np.array([inv_vFuncAdj_array[i_d][np.maximum(lS - 1, 0)],
                                  inv_vFuncKeep_array[i_d][np.maximum(lS - 1, 0)]])
            lS_right_y = np.array([inv_vFuncAdj_array[i_d][lS], inv_vFuncKeep_array[i_d][lS]])
            lS_exact = calc_linear_crossing(lS_x, lS_left_y, lS_right_y)[0]

        if uS == len(mNrmGrid):
            uS_exact = np.max(mNrmGrid)
        elif uS == 0:
            uS_exact = mNrmGrid_Total[0]
        else:
            uS_x = np.array([mNrmGrid[uS - 1], mNrmGrid[uS]])
            uS_left_y = np.array([inv_vFuncAdj_array[i_d][uS - 1], inv_vFuncKeep_array[i_d][uS - 1]])
            uS_right_y = np.array([inv_vFuncAdj_array[i_d][uS], inv_vFuncKeep_array[i_d][uS]])
            uS_exact = calc_linear_crossing(uS_x, uS_left_y, uS_right_y)[0]
        lSuS_array[i_d] = [lS_exact, uS_exact]
        added_values = [lS_exact, uS_exact]
        added_values = np.concatenate((added_values, mNrmGrid_Total))
        mNrmGrid_Total = np.sort(np.unique(added_values))

        if np.isnan(lSuS_array).any():
            print('Nans detected')
    return mNrmGrid_Total, lSuS_array

# def m_nrm_next(shocks, a_nrm, Rfree, PermGroFac):
#     """
#     Computes normalized market resources of the next period
#     from income shocks and current normalized market resources.
#
#     Parameters
#     ----------
#     shocks: [float]
#         Permanent and transitory income shock levels.
#     a_nrm: float
#         Normalized market assets this period
#
#     Returns
#     -------
#     float
#        normalized market resources in the next period
#     """
#     return Rfree / (PermGroFac * shocks[0]) * a_nrm + shocks[1]
#
# def n_nrm_next(shocks, a_nrm, n_nrm, dDepr):
#     """
#     Computes normalized market resources of the next period
#     from income shocks and current normalized market resources.
#
#     Parameters
#     ----------
#     shocks: [float]
#         Permanent and transitory income shock levels.
#     a_nrm: float
#         Normalized market assets this period
#
#     Returns
#     -------
#     float
#        normalized market resources in the next period
#     """
#     aux = np.ones(len(a_nrm)) * ((1 - dDepr) * n_nrm) # np.ones(len(aNrm)) * ((1 - dDepr) * nNrmGrid[0])
#     return aux.reshape(len(a_nrm),1) / (shocks[0])
#
# def vp_next(shocks, vPFunc_next, aNrm, nNrmGrid, CRRA, dDepr, Rfree, PermGroFac):
#     return shocks[0] ** (-CRRA) * vPFunc_next(n_nrm_next(shocks, aNrm, nNrmGrid, dDepr), m_nrm_next(shocks, aNrm, Rfree, PermGroFac))

def m_nrm_next(shocks, a_nrm, Rfree, PermGroFac): #TESTED: WORKS
    """
    Computes normalized market resources of the next period
    from income shocks and current normalized market resources.

    Parameters
    ----------
    shocks: [float]
        Permanent and transitory income shock levels.
    a_nrm: float
        Normalized market assets this period

    Returns
    -------
    float
       normalized market resources in the next period
    """
    return Rfree / (PermGroFac * shocks[0]) * a_nrm + shocks[1]

def n_nrm_next(shocks, a_nrm, n_nrm, dDepr): #TESTED: WORKS
    """
    Computes normalized market resources of the next period
    from income shocks and current normalized market resources.

    Parameters
    ----------
    shocks: [float]
        Permanent and transitory income shock levels.
    a_nrm: float
        Normalized market assets this period

    Returns
    -------
    float
       normalized market resources in the next period
    """
    aux = np.ones(len(a_nrm)) * ((1 - dDepr) * n_nrm)  # np.ones(len(aNrm)) * ((1 - dDepr) * nNrmGrid[0])
    return aux.reshape(len(a_nrm), 1) / (shocks[0])

def vp_next_vPFunc(shocks, vPFuncNext, aNrm, nNrmGrid, CRRA, dDepr, Rfree, PermGroFac):
    return shocks[0] ** (-CRRA) * vPFuncNext(n_nrm_next(shocks, aNrm, nNrmGrid, dDepr),
                                                         m_nrm_next(shocks, aNrm, Rfree, PermGroFac))


def vFunc_next(shocks, utility_inv, inv_vFunc_next, aNrm, nNrmGrid, CRRA, dDepr, Rfree, PermGroFac):
    inv_vFunc_plus_array = inv_vFunc_next(n_nrm_next(shocks, aNrm, nNrmGrid, dDepr),
                                       m_nrm_next(shocks, aNrm, Rfree, PermGroFac))
    vFunc_plus_array = utility_inv(inv_vFunc_plus_array, CRRA)

    return shocks[0] ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA) * vFunc_plus_array

