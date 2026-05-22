from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

#-----------------------------------------------------------------


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:

    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time

#--------------------------------------------------------------

def stats(x):

    '''
    Given the constant array 'x', stats will return the tuple
    ( E[x], StDev(x) := sqrt( E[ (x - E[x])^2 ] ))
    where E, represents the sample average.
    '''

    return x.mean(), x.std()
#-------------------------------------------------------------

def BS_trj(Obj, nt, So, T, J, sigma):

    '''
    Generates J trajectories according to the Black+Scholes model
    Each trajectory is made up of nT equally spaced steps.
    The output matrix will have the geometry S[nt+1][J],
    For each trajectory S[0] will hold the initial value.
    '''

    DT = T/nt
    S  = np.ndarray(shape = ( nt+1, J), dtype=np.double)

    X  = Obj.normal( -.5*sigma*sigma*DT, sigma*sqrt(DT), (nt,J))

    # for j in range(0, J): 
    #   S[0,j] = So
    S[0] = So

    for n in range(nt):
        # for j in range(0, J): 
        #   S[n+1,j] = S[n,j] * exp( X[n,j] )
        S[n+1] = S[n] * np.exp( X[n] )

    return S


def mc(Obj, N, So, T, J, sigma):

    DT = T/N

    # S[t, j]
    S = BS_trj(Obj, N, So, T, J, sigma)

    cout.write("Iter %d\n" %(S[0].size))
    res = {"t" : np.ndarray(shape = (N+1,), dtype=np.double)
          ,"m" : np.ndarray(shape = (N+1,), dtype=np.double)
          ,"S2": np.ndarray(shape = (N+1,), dtype=np.double)
          }

    for n in range(N+1):
        Sn    = S[n]
        m,std = stats(Sn)
        res["t"][n]  = n*DT
        res['m'][n]  = m
        res["S2"][n] = std*std

    return res


def martingale_check(**keywrds):

    Obj = np.random.RandomState()
    T0  = Timer();

    J      = keywrds["J"]
    N      = keywrds["N"]
    So     = keywrds["So"]
    T      = keywrds["T"]
    sigma  = keywrds["Sigma"]
    Seed   = keywrds["Seed"]

    Obj.seed(Seed)
    # St[0] = So, ... , St[N] = So
    St     = np.full(shape = (N+1,), fill_value=So, dtype=np.double)


    ex     = "2^%d" %J
    print("\n")
    print("@ %-12s %8s\n"   %("J", ex))
    print("@ %-12s %8.4f\n" %("So", So))
    print("@ %-12s %8.4f\n" %("T", T))
    print("@ %-12s %8.4f\n" %("sigma", sigma))

    j   = ( 1 << J )
    T0.start()
    res = mc(Obj, N, So, T, j, sigma)
    t1 = T0.stop()

    print("\n")
    print("%8s  %8s   %8s -- %8s\n" %( "t", "E[S(t)]", "McErr", "ThErr" ))
    for n in range( res["t"].size):
        t = res["t"][n]

        # the theoretica error in 't'.
        ThErr = So*sqrt( (exp( sigma*sigma*t) -1)/j )
        print("%8.4f  %8.4f   %8.4f -- %8.4f\n" %( t, res["m"][n], sqrt(res["S2"][n]/j), ThErr ))

    print("@ Elapsed %.4f sec.\n" %t1);

    err = 3.*np.sqrt( res["S2"]/j)
    lbl = "Mc: E[ S(t) ]"
    
    # the lower value of the y-axis
    Ym  = So*(1. - .3)

    # the highest value of the y-axis
    YM  = So*(1. + .3)

    # spacing between lines
    Dy  =  (YM-Ym)/20.

    Xl  = Dy
    Yl  = YM-Dy

    fig, ax = plt.subplots(1,1, figsize=(8, 6), linewidth=1)
    ax.set_title(r"Martingale test      $\sigma=%.3f, T = %.3f,\;N=2^{%d}$" %(sigma, T, J) )
    ax.set_ylim(Ym, YM)
    ax.errorbar(res["t"], res["m"], yerr=err, fmt='x', color='g', label=lbl)

    ax.plot(res["t"], St, color='r', label="So")
    ax.legend(loc="best")

    plt.show()

#---------------------------------------------------------

def mc_heston( rand, So, vol, intVol, cir, rho, Dt, N  ):

    '''
    @parms So    : initial value
    @parms intVol: volatility integral trajectory
    @parms cir   : CIR object
    @parms rho   : correlation between vol and underlying innovations
    @parms Dt    : tenor of the underlying trajectory
                   must agree with the nodes of the volatility trajectory
    @parms N     : number of underlying trajectories
    '''

    # length of the volatility trajectory
    # (including initial point)
    L   = len(intVol)
    th  = cir.theta
    k   = cir.kappa
    eta = cir.sigma
    nu  = vol
    I   = intVol

    # underlying trajectorie
    S  = np.ndarray(shape = (L, N), dtype=np.double ) # S[N, L] in fortran matrix notation

    xi = rand.normal( loc = 0.0, scale = 1.0, size=(L-1, N))

    # prime with So the starting value of each trajectory
    S[0] = So

    for n in range(1,L):
        DI   = I[n] - I[n-1]
        X    = -.5 * DI + (rho/eta)*( nu[n] - nu[n-1] - k*( th*Dt - DI) ) + np.sqrt((1. - rho*rho)*DI)*xi[n-1]
        S[n] = S[n-1]*np.exp(X)

    return S


#---------------------------------------------------------------------

class CIR:

    def __init__(self, **kwargs):
        self.kappa = kwargs["kappa"] 
        self.sigma = kwargs["sigma"] 
        self.theta = kwargs["theta"] 
        self.ro    = kwargs["ro"] 
        self.gamma = sqrt( self.kappa * self.kappa + 2 * self.sigma*self.sigma)
    # --------------

    def B(self, t):
        g = self.gamma
        k = self.kappa

        #
        # when g >> 1 we do neglect terms of the 
        # type g*exp(-gt)
        # the situation g >> 1 occurs only when we try to test 
        # very large violation from the Feller condition
        #
        if g > 30: return 2 /(g+k)
        h = exp(g*t) - 1
        return 2 * h/( (g+k)*h + 2*g)
    # ------------------------

    def A(self, t):
        g  = self.gamma
        k  = self.kappa
        th = self.theta
        s  = self.sigma
        #
        # when g >> 1 we do neglect terms of the 
        # type g*exp(-gt)
        # the situation g >> 1 occurs only when we try to test 
        # very large violation from the Feller condition
        #
        if g > 30:
            return ( 2*k*th/(s*s) ) * ( log( 2 * g ) + .5 * (k+g)*t - g*t + log(g+k))

        h = exp(g*t) - 1
        return ( 2*k*th/(s*s) ) * ( log( 2 * g ) + .5 * (k+g)*t - log( (g+k)*h + 2*g) )
    # --------------------------------------

    def P_tT( self, t, r=None):
        if r == None: r = self.ro
        return exp( -self.B(t)*r + self.A(t) )


def QT_cir_evol( rand, cir, L, dt, Nt, DT, N):

    s   = cir.sigma
    th  = cir.theta
    k   = cir.kappa
    ro  = cir.ro

    PSI_c    = 1.5
    V      = np.ndarray(shape = ( L+1, N ), dtype=np.double )
    In     = np.ndarray(shape = ( L+1, N ), dtype=np.double )
    xi     = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))
    V[0]   = ro
    In[0]  = 0.0

    for n in range(L):
        Zero = V[n] == 0.0
        V[n+1]= np.where(Zero,k*th*dt, 0.0)

        h   = 1. - exp(-k*dt)
        m   = th + ( V[n] - th)*(1. - h)
        s2  = (s*s*h/k)*( V[n] * (1. - h ) + .5*th*h )
        PSI = s2/(m*m)

        #V[n+1] = 0.0
        Mask   = np.logical_and( PSI > PSI_c, ~Zero )
        u      = rand.uniform(low=0.0, high=1.0, size = N)
        p      = (PSI-1)/(PSI+1)
        opMask = np.logical_and( u > p, Mask == 1 )
        beta   = (1. - p)/m
        x      = np.where(opMask, np.log( (1-p)/(1-u))/beta, 0.0)

        Mask   = np.logical_and( PSI <= PSI_c, ~Zero )
        o      = np.where( Mask, 2/PSI - 1., 0.0)
        b2     = np.where(Mask, o + np.sqrt(o*(o+1)), 0.0) 
        a      = m/(1. + b2)
        c      = np.power( (np.sqrt(b2)+ xi[n]), 2, where=Mask)
        y      = np.where(Mask, a*c, 0)


        V[n+1] += (x + y)
        In[n+1] = In[n] + (dt/2.) * ( V[n]  + V[n+1] )

    X  = np.ndarray(shape = ( Nt+1, N ), dtype=np.double ) # Y[Nt+1, 2]
    I  = np.ndarray(shape = ( Nt+1, N ), dtype=np.double ) # Y[Nt+1, 2]

    for n in range(Nt+1):
        tn = n * DT
        pos = int(tn/dt)
        X[n] = V[pos]
        I[n] = In[pos]

    return  (X,I)

#-------------------------------------------------------------



class VG:

    def __init__(self, **kwargs):
        self.eta    = kwargs["eta"]
        self.nu     = kwargs["nu"]
        self.th     = kwargs["theta"]

        self.Phi()
    #-----------------

    def Phi(self):
        self.phi = log( 1. - self.nu*self.th - .5*self.nu*self.eta*self.eta)

    def get(self):
        return np.array([self.eta, self.nu, self.th]) 

    def set(self, x):
        self.eta   = x[0]
        self.nu    = x[1]
        self.th    = x[2]
        self.Phi()

    @property
    def intensity(self): return 1./self.nu
    def compensator(self): return self.phi

    def cf(self, c_k, t):
        # 
        # c_u = i c_k
        #
        c_u = c_k*1j

        c_x = cmath.log( 1.0 -self.nu*self.th*c_u -.5*self.nu*(self.eta*self.eta)*c_u*c_u)
        comp = self.compensator()
        JMP  = t*self.intensity*(comp*c_u - c_x)

        return cmath.exp(JMP)

# -----------------------------------------------------

def vg_evol_step( rand, Sn, vg, Dt, N ):
    nu  = vg.nu
    eta = vg.eta
    th  = vg.th
    I   = vg.intensity
    phi = vg.compensator()
    g   = rand.normal( loc = 0.0, scale = 1.0, size=(N))
    xi  = np.float64( rand.gamma(shape=Dt/nu, scale=nu, size=(N) ))
    X   = th*xi + eta*g*np.sqrt(xi) + Dt*I*phi
    return Sn*np.exp(X)

def vg_evol( rand, So, vg, L, Dt, N ):

    S  = np.ndarray(shape = (L+1, N), dtype=np.double ) # S[N, L] in fortran matrix notation
    S[0] = So
    for n in range(L):
        S[n+1] = vg_evol_step(rand, S[n], vg, Dt, N)

    return S

# --------------------------------------------------------------------------------------


def cn_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d)
# ------------------------------------

def an_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d-s)
# ------------------------------------
#
#def cn_put( T, sigma, kT):
#    if kT == 0.0: return 0.0
#    s    = sigma*sqrt(T)
#    if s < 1.e-08:
#        if kT > 1.0: return 1.
#        else       : return 0.
#    d    = ( log(kT) + .5*s*s)/s
#    return norm.cdf(d)
# ------------------------------------

#def an_put( T, sigma, kT):
#    if kT == 0.0: return 1.0
#    s    = sigma*sqrt(T)
#    if s < 1.e-08:
#        if kT > 1.0: return 1.0 
#        else       : return 0.
#    d    = ( log(kT) + .5*s*s)/s
#    return norm.cdf(d-s)
# ------------------------------------

'''
    PUT = exp(-rT)Em[ (K - S(T))^+]
        where
    S(T) = So exp( (r-q)*T)*M
        let
    Fw(T) = So exp( (r-q)*T)
    kT    = K/Fw
        then
    PUT = So exp(-qT) Em[ (kT - M)^+]
        = So exp(-qT) FwEuroPut( T, sigma, kT)
'''
#def FwEuroPut(T, sigma, kT):
#    return ( kT* cn_put( T, sigma, kT) - an_put( T, sigma, kT) )

#def FwEuroCall(T, sigma, kT):
#    return FwEuroPut(T, sigma, kT) + 1. - kT

def FwEuroPut(T, vSigma, vKt):
    return ( vKt* cn_put( T, vSigma, vKt) - an_put( T, vSigma, vKt) )

def FwEuroCall(T, sigma, vkT):
    return FwEuroPut(T, sigma, vkT) + 1. - vkT

def euro_put(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * FwEuroPut( T, sigma, kT)
# -----------------------

def euro_call(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * FwEuroCall( T, sigma, kT)
# -----------------------

def impVolFromFwPut(vPrice, T, vKt):

    scalar = isinstance(vKt, float)
    if scalar: vKt = np.array([vKt])

    vSl = np.zeros(vKt.shape[0])
    vPl = np.maximum(vKt - 1., 0.0)

    vSh = np.ones(vKt.shape[0])
    while True:
        vPh = FwEuroPut(T, vSh, vKt)
        if ( vPh > vPrice).all(): break
        vSh = 2*vSh

    # d = vSh-vSl
    # d/2^N < eps
    # d < eps* 2^N
    # N > log(d/eps)/log(2)
    eps = 1.e-08
    d   = vSh[0]-vSl[0]
    N   = 2+int(log(d/eps)/log(2))

    for n in range(N):
        vSm  = .5*(vSh + vSl)
        vPm  = FwEuroPut(T, vSm, vKt)
        mask = vPm > vPrice
        vSh[mask] = vSm[mask]
        vSl[~mask] = vSm[~mask]

    
    if scalar: return .5*(vSh + vSl)[0]
    return .5*(vSh + vSl)

# --------------------------------------------


def vanilla_options( **keywrds):

    So     = keywrds["S"]
    k      = keywrds["k"]
    r      = keywrds["r"]
    q      = keywrds["q"]
    T      = keywrds["T"]
    sigma  = keywrds["sigma"]
    fp     = keywrds["fp"]

    fp.write("@ %-24s %8.4f\n" %("So", So))
    fp.write("@ %-24s %8.4f\n" %("k", k))
    fp.write("@ %-24s %8.4f\n" %("T", T))
    fp.write("@ %-24s %8.4f\n" %("r", r))
    fp.write("@ %-24s %8.4f\n" %("q", q))
    fp.write("@ %-24s %8.4f\n" %("sigma", sigma))

    kT   = exp((q-r)*T)*k/So
    cnP  = k*exp(-r*T)*cn_put ( T, sigma, kT)
    anP  = So*exp(-q*T)*an_put ( T, sigma, kT)
    put  = euro_put ( So, r, q, T, sigma, k)
    call = euro_call( So, r, q, T, sigma, k)

    return {"put": put, "call": call, "anP": anP, "cnP": cnP}


#------------------------------------------------------------------

def cn_put_delta( S, r, T, sigma, B, M):
    mu = r - .5*sigma*sigma
    g  = 2.*mu/(sigma*sigma)
    mT = exp(-r*T)*M/S
    MT = exp(-r*T)*M*S/(B*B)
    return cn_put( T, sigma, mT) - exp( g * log(B/S)) *cn_put( T, sigma, MT)
# --

def an_put_delta( S, r, T, sigma, B, M):
    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    mT = exp(-r*T)*M/S
    MT = exp(-r*T)*M*S/(B*B)
    return S*an_put( T, sigma, mT) - ( B*B/S)*exp( g * log(B/S)) *an_put( T, sigma, MT);
# --

def cn_put_ko( S, r, T, sigma, k, B):

    # High barrier ...
    if S < B :  
        M = min(k,B)
        return cn_put_delta( S, r, T, sigma, B,  M);

    # low barrier ...
    if k < B :  
        return 0.0;

    return cn_put_delta( S, r, T, sigma, B,  k) - cn_put_delta( S, r, T, sigma, B,  B);
# ---

def cn_call_ko( S, r, T, sigma, k, B):

    # High barrier ...
    if S < B: 
        return cn_put_delta( S, r, T, sigma, B, B) - cn_put_ko( S, r, T, sigma, k,  B);

    # low barrier ...
    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    f  = exp( g * log(B/S)) ;
    return ( 1.0 - f )  - cn_put_delta( S, r, T, sigma, B, B) - cn_put_ko( S, r, T, sigma, k,  B);
# ---

def an_put_ko( S, r, T, sigma, k, B):

    # High barrier ...
    if S < B:
        M = min(k,B)
        return an_put_delta( S, r, T, sigma, B,  M);

    # Low barrier
    if k < B: return 0.0;
    return an_put_delta( S, r, T, sigma, B,  k) - an_put_delta( S, r, T, sigma, B,  B);
# --

def an_call_ko( S, r, T, sigma, k, B):

    # High barrier ...
    if S < B: 
        return an_put_delta( S, r, T, sigma, B, B) - an_put_ko( S, r, T, sigma, k,  B);

    # low barrier ...
    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    f  = exp( g * log(B/S)) ;
    return ( S - (B*B/S)*f ) - an_put_delta( S, r, T, sigma, B, B) - an_put_ko( S, r, T, sigma, k,  B);
# -----

# Knock out put option
def put_ko( S, r, T, sigma, k, B):
    return exp(-r*T)*k * cn_put_ko( S, r, T, sigma, k, B) - an_put_ko( S, r, T, sigma, k, B);

# Knock out call option
def call_ko( S, r, T, sigma, k, B):
    return an_call_ko( S, r, T, sigma, k, B) - exp(-r*T)*k * cn_call_ko( S, r, T, sigma, k, B)





#-------------------------------------------------------------------------------


def __recursive_finder( x, ll, s, e):
    '''
    upon entry this routine, the following constraints must hold:
    ll[s] <= x < ll[e]
    '''
    if (e == s) or ( e == s+1): return s 

    m = int( (e+s)/2 )

    if x <  ll[m]: 
        return __recursive_finder(x, ll, s, m)
    if x == ll[m]: 
        return m
    if x >  ll[m]: 
        return __recursive_finder(x, ll, m, e)
# -----------------------------------------------------

def find_pos( x, ll):
    if x < ll[0]: return -1
    if x >= ll[-1]: return  len(ll)-1

    return __recursive_finder(x, ll, 0, len(ll) - 1)
    
# -----------------------------------------------------

class Zc:

    def __init__( self, **keywrds):
        curve   = keywrds["curve"]
        self.tl = curve[0]
        self.rc = curve[1]
        self.pt = np.exp(-self.tl*self.rc)
    # ------------------------------------------

    @classmethod
    def from_discount_curve(cls, t, P):
        if t[0] == 0:
            t = np.array(t[1:])
            P = P[1:]
        r = -np.log(P)/t
        return cls(curve=(t, r))

    @classmethod
    def from_cc_zero_coupon_rates(cls, t, rc):
        return cls(curve=(t, rc))

    @classmethod
    def from_yc_zero_coupon_rates(cls, t, yc):
        r = np.log( 1 + yc)
        return cls(curve=(t, r))


    def f_0t( self, t ):

        '''
            f_0t := \int_0^t r(s) ds
            
            condition: t >= 0
        '''

        tl  = self.tl
        r  = self.rc
        pt = self.pt

        n = find_pos( t, tl )

        if n < 0            : return t*r[0]
        if n == tl.size - 1 : return t*r[-1]
        
        fs = tl[n]*r[n]
        fe = tl[n+1]*r[n+1]
        return (tl[n+1] - t) * fs/(tl[n+1] - tl[n]) + (t - tl[n])* fe/(tl[n+1]-tl[n])
    #------------------------------------------------

    def rz( self, t): return self.f_0t(t)/t
    def ry( self, t): return exp(self.f_0t(t)/t) - 1.
    def P_0t( self, t): return exp( -self.f_0t(t) )
    # -----------------------------------------------------------------

    def show( self ):

        tl = self.tl
        n  = 0
        print("%3s  %9s  %8s  %8s  %8s" %( "pos", "t", "P_0t", "rc", "ry"))
        for t in tl:
            if fabs(t) < 1.e-10: continue
            p  = self.P_0t(t)
            xc = self.rz(t)
            xy = self.ry(t)
            print("%3d  %9.6f  %8.6f  %8.6f  %8.6f" %( n, t, p, xc, xy))
            n += 1
        


# -----------------------------------------------------

def PF_MC_Price(S0s, Vlt, Cor, T, Nt, MC, r, q, dflt=1):        #Portfolio pricing
    '''
    :param S0s:     List of initial asset prices
    :param Vlt:     List of asset volatility
    :param Cor:     Correlation matrix of portfolio
    :param T:       Maturity
    :param Nt:      number of time steps
    :param MC:      number of MC simulations
    :param r:       Risk free interest rate
    :param q:       Dividend
    :param dflt:   Default threshold, if an asset looses more than 'dflt' consider defaulted
    :return:        A Dictionary of timeline(1 row), assets-price(multiple rows), MCerror(Multiple rows) 
    '''
      
    dt  = T / Nt
    Nas = len(S0s)  #number of assets
      
    # 3-D: MC matrices of the type (time-steps, MCs, #assets)
    S        = np.zeros(shape=(1+Nt, MC, Nas), dtype=float)
    S[0,:,:] = S0s          # First tensor is initial prices
      
    rw       = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(Nt, MC, Nas)) #random walks i.i.d
    Ch       = np.linalg.cholesky(Corrs)                                        #Cholesky lower triangular factor
    print(rw.shape)
    # i=j: number of assets; t: time steps; m: MCS;
    # multiply col (j) of 'Ch' with tensor (j) of 'rw' then sum and put in tensor (i)
    # thus each tensor {asset} is a correlated inner product of other assets
    # above statement is done through all rows (t) and columns (m) of 'rw'
    dWn = np.einsum('ij,tmj->tmi' , Ch, rw)
    Vlt = Vols.reshape(1, -1)    #vectorizing volatility
      
    for t in range(Nt):  # Next row simulated price of next time step
          S[t+1,:,:] = S[t,:,:] * np.exp( (-Vlt*Vlt/2)*dt + Vlt*dWn[t,:,:] )
          #S[t+1,:,:] = np.where( S[t+1,:,:] < (1-dflt)*S[0,:,:], 0, S[t+1,:,:] )    #default condition
    return S






def BSn_trj( Obj
           , Nt           # nr of steps in the trajectory
           , So           # array of initial values
           , C  
           , sigma
           , T
           , J
           ):

    '''
    Generates J trajectories according to the Black+Scholes model
    Each trajectory is made up of nT equally spaced steps.
    The output matrix will have the geometry S[Nt+1][J],
    For each trajectory S[0] will hold the initial value.
    '''
    Q  = len(So)
    DT = T/Nt

    S  = np.ndarray(shape = ( Nt+1, J, Q), dtype=np.double)

    m   = -.5*sigma*sigma*DT
    Vol = sigma*sqrt(DT)
    X   = Obj.normal( 0.0, 1.0, (Nt, J, Q))

    #
    # G[Nt, J, Q]
    # G[l,k,i]  = Sum_j C  [i,j] C[l,k,j]
    #
    G   = np.einsum('ij,lkj->lki', C  , X)

    #     m[Q] + G[Nt, J, Q]*Val[Q]
    X   = m + G*Vol

    S[0] = So
    for n in range(Nt):
        S[n+1] = S[n] * np.exp( X[n])

    return S

#----------------------------------

def martingale_test(**keywrds ):

    Rho   = keywrds["Rho"]   # required
    sigma = keywrds["sigma"] # required

    Nt    = keywrds.get("Nt", 12)
    Js    = keywrds.get("Js", 14)
    Seed  = keywrds.get("seed", 1)

    C     = np.linalg.cholesky(Rho)

    T     = keywrds.get("T", 2.0)
    J     = ( 1 << Js)

    print('\n')
    print("@ %-12s: %-12s %8s\n" %("Info", "type(C)", type(C)))
    print("@ %-12s: %-12s %8d\n" %("Info", "J", J))
    print("@ %-12s: %-12s %8d\n" %("Info", "Nt", Nt))
    print("@ %-12s: %-12s " %("Info", "Sigma") ); print(sigma)
    print('\n')

    Q     = len(sigma)
    So    = np.full(Q, 1., dtype=np.double)

    rand = np.random.RandomState()
    rand.seed(Seed)
    with FTimer('inner_step_test'):
        # S[Nt, J, Q]
        S = BSn_trj( rand
                   , Nt              # nr of steps in the trajectory
                   , So              # array of initial values
                   , C               # triangular choelesky component
                   , sigma           # array of volatilities
                   , T               # maturity
                   , J               # number of trajectories
                   )

        MCerr= np.ndarray(shape = (Nt+1,Q), dtype = np.double)
        Mean = np.ndarray(shape = (Nt+1,Q), dtype = np.double)

        # Fill the dictionary
        MCerr = 3.* np.sqrt( np.var(S, axis=1) / J )
        Mean  = np.mean(S, axis=1)
        #print(MCerr.shape)

        '''
        #print(" %12s %3s %12s" %( "E[S]", "+/-", "Err"))
        for nt in range(1, Nt+1):

            # St = S[J, Q]
            St = S[nt]
            S2 = St*St

            # A[2,3] = a_00, a_01, a_02, a_10, a_11, a_12
            # Avg[Q]
            # A[q] = Sum_j St[j,q]
            Avg = np.add.reduce(St,0)/J
            Err = np.add.reduce(S2,0)/J
            Err -= Avg*Avg
            Err  = np.sqrt( np.maximum(Err, 0.0)/J )
            print("NT : %3d     E[S]: %12s %3s Err: %12s" %(nt, Avg ,"+/-",Err ))

            fig, ax = plt.subplots(1,1, figsize=(8, 6), linewidth=1)
            ax.set_title(r"Martingale test      $\sigma=%.3f, T = %.3f,\;N=2^{%d}$" %(sigma, T, J) )
            ax.errorbar(t, traj["mu_hat"], yerr=err, fmt="x", label=lbl)
            ax.plot(t, So_vector, label = 'So')
            ax.set_ylim(ymin,ymax)
            ax.legend(loc="best")
        '''

        ############################### 2D Plot ###############################
        t_line = np.linspace(0, T, Nt+1)
        fig, ax = plt.subplots(Q,1, sharex=True, figsize=(5,20))
        ax[0].set_title("Martingale Test", fontsize=15)
        ax[-1].xaxis.set_ticks(t_line.round(2))
        ax[-1].xaxis.set_tick_params(labelsize=5)
        ax[-1].set_xlabel("Time Steps (Nt)", fontsize=10)
        
        for q in range(Q):
            ax[q].yaxis.set_tick_params(labelsize=5)
            ax[q].set_ylim(0.95, 1.05)
            ax[q].yaxis.set_ticks([0.95,1.00, 1.05])
            ax[q].set_ylabel("Asset" + str(q+1), size=8)
            ax[q].plot(t_line, np.ones(shape=Nt+1), color='r', label="Initial Price")
            ax[q].errorbar(t_line, Mean[:,q], yerr=MCerr[:,q], fmt='+' , color='g', label="MC E[S(t)]")
        #######################################################################



#--------------------------------------------------------------------
class QFError(Exception):
    "A custom exception used to report QF students errors "

    def __init__(self, text):
        self.text = text

    def showErr(self):
        print("@ %-12s: %s" %("Error", self.text))



#-----------

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:

    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time
    
def named_timer(name):
    
    scale = 1
    fmt   = "sec."
    Tot   = {}
    count = 0

    def middle_funct(f):
        Tot[name] = 0.0
        def inner_funct(*args, **kwargs):
            nonlocal Tot
            nonlocal count
            count += 1
            T0 = time.perf_counter()
            res = f(*args, **kwargs)
            T1 = time.perf_counter()
            #elapsed = "%12.4e %s" %( (T1-T0)*scale, fmt)
            elapsed = (T1-T0)*scale
            Tot[name] += elapsed
            print("@ %-12s: [%3d] executed %-32s  elapsed %12.4f %-4s, Tot %12.4f %-4s" %("Info", count, name, elapsed, fmt, Tot[name], fmt))
            return res
        return inner_funct

    return middle_funct


class FTimer():

    scale = 1.0
    fmt   = "sec."

    def __init__(self, name):
        self.name = name
        self.To = 0.0
        
    def __enter__(self):
        self.To = time.perf_counter()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = "%12.4e %s" %( (time.perf_counter()- self.To)*self.scale, self.fmt)
        print("@ %-12s: executed %-20s  elapsed %s" %("Info", self.name, elapsed ))
        return False



#========================================

class jumps:

    def __init__(self, **kwargs):
        self._intnsty = kwargs['lmbda']
        self._sgma    = kwargs['sigma']
# -----------------------------------------

    @property
    def intensity(self): return self._intnsty
    @property
    def sigma(self): return self._sgma

    def do_jmp(self, Obj, Dt, J):

        Z   = np.full(shape=J, fill_value=0.0, dtype=np.float)
        Nj  = Obj.poisson(lam=self.intensity * Dt, size=J)
        sup = Nj.max()
        j   = sup

        while j > 0:
            Z = Z + self.single_jump(Obj, Nj >= j)
            j -= 1

        return Z
    # -------------------------------------------------------

    def jd_evol_step(self, rand, Dt, J):

        '''
        Performs 1 step for J trajectories 
        Black-Scholes diffusion + jumps
        '''

        s = self.sigma * sqrt(Dt)
        X = rand.normal( -.5*s*s, s, J)
        X = X + self.do_jmp(rand, Dt, J) + Dt*self.intensity*self.compensator()
        return np.exp(X)

    def cf(self,c_k, t):
        s = self.sigma
        # 
        # c_u = i c_k
        #
        c_u = c_k*1j

        c_x  = -.5 *s*s*c_u*c_u
        comp = -.5 *s*s

        #
        # X_cf = dt * ( u*g - f )  
        #
        X_cf =  t*(comp*c_u - c_x)


        c_x  = self.phi_X(c_u)
        comp = self.compensator()
        JMP  =  t*self.intensity*(comp*c_u - c_x)

        return cmath.exp(X_cf + JMP)

    # =================================================================================


class jmp_binary(jumps):

    '''
    Pr( J==u ) = pi
    Pr( J==d ) = 1. - pi
    '''

    def __init__(self, **kwargs):
        self._pi      = kwargs["pi"] 
        self._u       = kwargs["u"] 
        self._d       = kwargs["d"] 
        super().__init__(**kwargs)
    # -------------------------------------------------------

    def single_jump(self, rand, mask):
            J       = len(mask)
            z       = rand.uniform(low=0.0, high=1.0, size=J)
            pi_mask = np.logical_and(mask, z < self._pi)
            up  = np.where(pi_mask, self._u, 0)

            pi_mask = np.logical_and(mask, z > self._pi)
            down  = np.where(pi_mask, self._d, 0)

            return ( up + down )
    # -------------------------------------------------------

    def compensator(self):
        phi_J =  self._pi*exp(self._u) + (1.0 - self._pi)*exp(self._d) 
        return (1.0 - phi_J)

    def phi_X(self, c_u):
        return 1. - self._pi*cmath.exp(self._u*c_u) - (1.-self._pi)*cmath.exp(self._d*c_u)
        
# ================================================================================

class jmp_normal(jumps):

    '''
    P( J < L ) = N_{0,1}( (L - m)/eta )
    '''

    def __init__(self, **kwargs):
        self._m       = kwargs["m"] 
        self._eta     = kwargs["eta"] 
        super().__init__(**kwargs)

    def single_jump(self, rand, mask):
            J       = len(mask)
            X   = rand.normal( self._m, self._eta, J)
            return np.where(mask, X, 0.)
    # -------------------------------------------------------

    def compensator(self):
       phi_J = exp( self._m + .5*(self._eta)*(self._eta)); 
       return (1.0 - phi_J)

    def phi_X(self, c_u):
        c_z = self._m*c_u + .5 * self._eta*self._eta*c_u*c_u
        return 1. - cmath.exp(c_z)

