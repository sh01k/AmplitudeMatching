import numpy as np
from scipy import special
import scipy.spatial.distance as distfuncs
from scipy import optimize
import matplotlib.patches as pat

"""Sound field functions"""

def PlaneWave(amp, ang, pos, k):
    """Plane wave function
    Parameters
    ------
    amp : Amplitude
    ang : Propagation angle
    pos :  Receiver positions
    k : Wave number
    Returns
    ------
    p : Pressure
    """
    kVec = k * np.array([np.cos(ang), np.sin(ang)]) # Wave vector
    p = amp * np.exp(1j * (kVec[0] * np.array(pos[:,0]) + kVec[1] * np.array(pos[:,1])) )
    return p


def CylindricalWave(amp, posSrc, posRcv, k):
    """Cylindrical wave function
    Parameters
    ------
    amp : Amplitude
    posSrc : Source positions
    posRcv : Receiver positions
    k : Wave number
    Returns
    ------
    p : Pressure (complex)
    """
    r = distfuncs.cdist(posRcv, posSrc)
    if np.isscalar(k):
        p = amp * (1j/4) * special.hankel1(0, k * r)
    else:
        p = amp * (1j/4) * special.hankel1(0, k * r[None,:,:])
    return p


def SynthSoundField(posSim, posSPK, drv, numSim, k):
    """Simulate synthesized sound field
    Parameters
    ------
    posSim : Simulated positions
    posSPK : Loudspeaker positions
    drv : Driving signals of loudspeakers
    k : Wave number
    Returns
    ------
    p : Synthesized pressure distribution
    """
    G = CylindricalWave(1.0, posSPK, posSim, k)
    pvec = G @ drv
    p = pvec.reshape(numSim[0], numSim[1])
    return pvec, p


def TransFuncMat(posSPK, posCP, k):
    """Generate transfer function matrix
    Parameters
    ------
    posSPK : Positions of loudspeakers 
    posCP : Positions of control points
    k : Wave number
    Returns
    ------
    G : Transfer function matrix
    """
    G = CylindricalWave(1.0, posSPK, posCP, k)
    return G


"""Evaluation measures"""

def MSE(syn, des):
    """Mean Square Error (MSE)
    Parameters
    ------
    syn: Synthesized pressure
    des: Desired pressure
    Returns
    ------
    MSE: Mean Square Error (MSE)
    """
    MSE = 10 * np.log10(np.linalg.norm(np.abs(syn) - np.abs(des), ord=2)**2 / len(des))
    return MSE


def AC(synB, synQ):
    """Acoustic Contrast (AC)
    Parameters
    ------
    synB: Synthesized pressure in bright zone
    synQ: Synthesized pressure in quiet zone
    Returns
    ------
    MSE: Mean Square Error (MSE)
    """
    AC = 10 * np.log10( np.linalg.norm(synB)**2/np.linalg.norm(synQ)**2 )
    return AC


"""Sound field control functions"""

def PressureMatching(G, reg, posSPK, des):
    """Pressure matching
    Parameters
    ------
    G: Transfer function matrix
    reg: Regularization parameter
    posSPK: Positions of loudspeakers
    des: Desired pressure
    Returns
    ------
    drv: Loudspeaker driving signals
    """
    numSPK = posSPK.shape[0]
    if G.ndim == 2:
        drv = np.linalg.inv(G.conj().T @ G + reg * np.identity(numSPK)) @ G.conj().T @ des
    elif G.ndim == 3:
        drv = np.linalg.inv(np.transpose(G.conj(), (0, 2, 1)) @ G + reg * np.identity(numSPK) ) @ np.transpose(G.conj(), (0, 2, 1)) @ des
    return drv


def AcoustContrastControl(Gb, Gq):
    """Acoustic Contrast Control
    Parameters
    ------
    Gb: Transfer function matrix in bright zone
    Gq: Transfer function matrix in quiet zone
    Returns
    ------
    drv: Loudspeaker driving signals
    """
    reg = 1e-4
    numSPK = Gb.shape[1]
    A = np.linalg.inv((Gq.conj().T @ Gq) + reg * np.identity(numSPK)) @ (Gb.conj().T @ Gb)
    (S, U) = np.linalg.eig(A)
    drv = U[:,0]
    return drv


def CostFuncAM(drv, *args):
    """Cost function of amplitude matching
    Parameters
    ------
    drv: Loudspeaker driving signals
    args: (G, des, reg) = (Transfer function matrix, Desired pressure, Regularization parameter)
    Returns
    ------
    J: Cost function value
    """
    J = np.linalg.norm(np.abs(args[0] @ drv) - np.abs(args[1]), ord=2)**2 + args[2] * np.linalg.norm(drv, ord=2)**2
    return J


def CostFuncCmplxAM(drvCmplx, *args):
    """Cost function of amplitude matching with vectorized complex variables
    Parameters
    ------
    drvCmplx: Loudspeaker driving signals with vectorized complex variables
    args: (G, des, reg, numSPK) = (Transfer function matrix, Desired pressure, Regularization parameter, Number of loudspeakers)
    Returns
    ------
    J: Cost function value with vectorized complex variables
    """
    drv = drvCmplx[:args[3]] + 1j * drvCmplx[args[3]:]
    G = args[0]
    des = args[1]
    reg = args[2]
    J = np.linalg.norm(np.abs(G @ drv) - np.abs(des), ord=2)**2 + reg * np.linalg.norm(drv, ord=2)**2
    return J


def GradCostFuncAM(drv, *args):
    """Gradient of cost function of amplitude matching
    Parameters
    ------
    drv: Loudspeaker driving signals
    args: (G, des, reg, numSPK) = (Transfer function matrix, Desired pressure, Regularization parameter, Number of loudspeakers)
    Returns
    ------
    Jd: Gradient of cost function value
    """
    Jd = args[0].conj().T @ ((np.abs(args[0] @ drv) - np.abs(args[1])) * np.exp(1j * np.angle(args[0] @ drv))) + args[2] * drv
    return Jd


def GradCostFuncCmplxAM(drvCmplx, *args):
    """Gradient of cost function of amplitude matching with vectorized complex variables
    Parameters
    ------
    drv: Loudspeaker driving signals with vectorized complex variables
    args: (G, des, reg, numSPK) = (Transfer function matrix, Desired pressure, Regularization parameter, Number of loudspeakers)
    Returns
    ------
    Jd: Gradient of cost function value with vectorized complex variables
    """
    drv = drvCmplx[:args[3]] + 1j * drvCmplx[args[3]:]
    Jd = args[0].conj().T @ ((np.abs(args[0] @ drv) - np.abs(args[1])) * np.exp(1j * np.angle(args[0] @ drv))) + args[2] * drv
    JdCmplx = np.concatenate([Jd.real, Jd.imag])
    return JdCmplx


def MM(numSPK, des, G, reg, drv0, **keyargs):
    """MM algorithm for amplitude matching
    Parameters
    ------
    numSPK: Number of loudspeakers
    des: Desired pressure
    G: Transfer function matrix
    reg: Regularization parameter
    drv0: Initial value of driving signals
    keyargs: (max_iter, dtol) = (Maximum number of iterations, Threshold for gradient of cost function)
    Returns
    ------
    drv: Loudspeaker driving signals
    drvList: List of loudspeaker driving signals for each iteration
    """
    if 'max_iter' in keyargs:
        max_iter = keyargs['max_iter']
    if 'dtol' in keyargs:
        dtol = keyargs['dtol']
    else:
        dtol = 0
    A = np.linalg.inv( G.conj().T @ G + reg * np.identity(numSPK) ) @ G.conj().T
    drv = drv0
    drvList = [drv]
    v = np.abs(des) * np.exp(1j * np.angle( G @ drv ))
    k = 0
    ddiff = 1.0 
    for k in range(max_iter):
        drv = A @ v
        drvList.append(drv)
        ddiff = np.linalg.norm(drvList[k+1]-drvList[k]) / np.linalg.norm(drvList[k])
        v = np.abs(des) * np.exp(1j * np.angle( G @ drv))
        #print("itr: %d, ddiff: %f" % (k, ddiff))
        if ddiff <= dtol:
            break
    return drv, drvList


def ADMM(numSPK, des, G, reg, drv0, **keyargs):
    """ADMM for amplitude matching
    Parameters
    ------
    numSPK: Number of loudspeakers
    des: Desired pressure
    G: Transfer function matrix
    reg: Regularization parameter
    drv0: Initial value of driving signals
    keyargs: (max_iter, dtol, rho) = (Maximum number of iterations, Threshold for gradient of cost function, Penalty parameter)
    Returns
    ------
    d: Loudspeaker driving signals
    dList: List of loudspeaker driving signals for each iteration
    """
    if 'max_iter' in keyargs:
        max_iter = keyargs['max_iter']
    if 'dtol' in keyargs:
        dtol = keyargs['dtol']
    else:
        dtol = 0
    if 'rho' in keyargs:
        rho = keyargs['rho']
    else:
        rho = 1.0
    w = np.zeros(G.shape[0]) # Lagrange multiplier
    des = np.abs(des)
    d = drv0
    dList = [drv0]
    Gd = G @ d
    Ginv = np.linalg.inv((2 * reg / rho) * np.identity(numSPK) + G.conj().T @ G)
    k = 0
    ddiff = 1.0
    for k in range(max_iter):
        h = Gd + w / rho
        phase = h/np.abs(h)
        mag = (rho * np.abs(h) + 2 * des)/(rho + 2)
        d = Ginv @ G.conj().T @ (mag * phase - w / rho)
        Gd = G @ d
        w = w + rho * (Gd - mag * phase)
        dList.append(d)
        ddiff = np.linalg.norm(dList[k+1]-dList[k]) / np.linalg.norm(dList[k])
        #print("itr: %d, ddiff: %f" % (k, ddiff))
        if ddiff <= dtol:
            break
    return d, dList


"""Misc"""

def CircularGrid(rad, num):
    """Grid points equiangularly arranged on circle
    Parameters
    ------
    rad : Radius of circle
    num : Number of points
    Returns
    ------
    pos : Positions 
    """
    theta = np.linspace(0, 2 * np.pi, num, endpoint = False)
    pos = np.array([rad * np.cos(theta), rad * np.sin(theta)]).T
    return pos


def RectGrid(lenX, lenY, num):
    """Grid points regularly arranged on rectangular boundary
    Parameters
    ------
    lenX : Length of rectangle in x axis
    lenY : Length of rectangle in y axis
    num : Number of points
    Returns
    ------
    pos : Positions 
    """
    dd = (lenX * 2 + lenY * 2) / num
    pvec = ((np.arange(num)+0.5)*dd)
    pvec_ht = pvec[pvec <= lenX]
    x_ht = pvec_ht-lenX/2
    y_ht = (lenY/2)*np.ones(len(pvec_ht))
    pvec_vr = pvec[(pvec>lenX) & (pvec<=lenX+lenY)] - lenX
    x_vr = (lenX/2)*np.ones(len(pvec_vr))
    y_vr = -pvec_vr+lenY/2
    pvec_hd = pvec[(pvec>lenX+lenY) & (pvec<=lenX*2+lenY)] - lenX - lenY
    x_hd = -pvec_hd+lenX/2
    y_hd = (-lenY/2)*np.ones(len(pvec_hd))
    pvec_vl = pvec[(pvec>lenX*2+lenY)] - lenX*2 -lenY
    x_vl = (-lenX/2)*np.ones(len(pvec_vl))
    y_vl = pvec_vl-lenY/2
    x = np.concatenate([x_ht, x_vr, x_hd, x_vl], 0)
    y = np.concatenate([y_ht, y_vr, y_hd, y_vl], 0)
    pos = np.array([x, y]).T
    return pos


def plotCircles(ax, pos, rad):
    """Plot circles
    Parameters
    ------
    ax: pyplot.axis
    pos: Center positions
    rad: Radius
    """
    num = pos.shape[0]
    for i in np.arange(num):
        circle = pat.Circle(xy = (pos[i,0], pos[i,1]), radius=rad, ec='k', fill=False)
        ax.add_patch(circle)
    return





