from skimage.color import lab2rgb, xyz2rgb
import numpy as np
from scipy.special import erf, erfc
pi = np.pi


def iris(x):
    x = np.asarray( x )

    L = 60*np.ones_like( x )
    th = -pi + 2*pi*x

    a = 60*np.cos(th)
    b = 60*np.sin(th)
    
    Lab = np.asarray([L,a,b]).transpose()
    rgb = lab2rgb( Lab )

    return rgb

def purple_wing( x ):
    x = np.asarray( x )
    L = 100 * x
    C = 35*((erf((x-0.10)*15)+erfc((x-0.9)*15))-1)/2; 
    H = -pi/2 + pi*x

    a = C*np.cos(H)
    b = C*np.sin(H)
    
    Lab = np.asarray([L,a,b]).transpose()
    rgb = lab2rgb( Lab )

    return rgb

def chroma_cardioid( x ):
    x = np.asarray( x )
    L = 100 * x
    C = 50*((erf((x-0.10)*15)+erfc((x-0.9)*15))-1)/2; 
    H = 2*pi*(1.49*x +pi)

    a = C*np.cos(H)
    b = C*np.sin(H)
    
    Lab = np.asarray([L,a,b]).transpose()
    rgb = lab2rgb( Lab )

    return rgb

def copper_5N( x ):
    x = np.asarray( x )
    L = 100 * x
    C = 35*((erf((x-0.10)*15)+erfc((x-0.9)*15))-1)/2; 
    H = -pi/2+x*pi+pi/3

    a = C*np.cos(H)
    b = C*np.sin(H)
    
    Lab = np.asarray([L,a,b]).transpose()
    rgb = lab2rgb( Lab )

    return rgb

def cryo( x ):
    x = np.asarray( x )
    L = 100 * x
    C = 25*((erf((x-0.2)*5)+erfc((x-0.8)*5))-1)/2; 
    H = 1.5*(1.8*pi+x)

    a = C*np.cos(H)
    b = C*np.sin(H)
    
    Lab = np.asarray([L,a,b]).transpose()
    rgb = lab2rgb( Lab )

    return rgb

def crayon_box(x):
    x = np.asarray( x )
    L = 100 * x
    C = 65*((erf((x-0.2)*5)+erfc((x-0.8)*5))-1)/2; 
    H = 2*(1.8*pi+x)

    a = C*np.cos(H)
    b = C*np.sin(H)
    
    Lab = np.asarray([L,a,b]).transpose()
    rgb = lab2rgb( Lab )

    return rgb

def lapis(x):
    x = np.asarray( x )
    L = 100 * x
    C = 40*((erf((x-0.1)*15)+erfc((x-0.9)*15))-1)/2; 
    H = pi/4*(-0.65*pi + x)

    a = C*np.cos(H)
    b = C*np.sin(H)
    
    Lab = np.asarray([L,a,b]).transpose()
    rgb = lab2rgb( Lab )

    return rgb


def planckian_locus( x ):
    N = len(x)
    lm = np.linspace(350, 750, 1024)*(10**-9)
    dL = lm[1]-lm[0]
    X_L = x_func(lm*10**9)
    Y_L = y_func(lm*10**9)
    Z_L = z_func(lm*10**9)
    
    T_min = 1000; T_max = 10000
    Ts = np.linspace( T_min, T_max, N )

    h = 6.626 * (10**-34) # J*s
    c = 2.998 * (10**8  ) # m/s
    k = 1.381 * (10**-23) # J/K

    c1 = 2*pi*h*(c**2); c2 = h*c/k
    X_T = np.zeros( N )
    Y_T = np.zeros( N )
    Z_T = np.zeros( N )
    for ind_T in range(N):
        T = Ts[ind_T]
        M_L = c1/( (lm**5)*(np.exp(c2/(lm*T))-1) )
        
        X_T[ ind_T ] = dL*np.sum( X_L*M_L )
        Y_T[ ind_T ] = dL*np.sum( Y_L*M_L )
        Z_T[ ind_T ] = dL*np.sum( Z_L*M_L )
    
    XYZ = np.asarray( [X_T/(2*Y_T), Y_T/(2*Y_T), Z_T/(2*Y_T)] ).transpose()
    RGB = xyz2rgb(XYZ)
    return RGB

def x_func( lm ):
    #Color matching function x(lm)
    x = 1.056*g_func( lm, 599.8, 37.9, 31.0) \
        + 0.362*g_func( lm, 442.0, 16.0, 26.7) \
        - 0.065*g_func( lm, 501.1, 20.4, 26.2)
    return x

def y_func( lm ):
    #Color matching function y(lm)
    y = 0.821*g_func( lm, 568.8, 46.9, 40.5) \
        + 0.286*g_func( lm, 530.9, 16.3, 31.1)
    return y

def z_func( lm ):
    #Color matching function z(lm)
    z = 1.217*g_func( lm, 437.0, 11.8, 36.0) \
        + 0.681*g_func( lm, 459.0, 26.0, 13.8)
    return z


def g_func( x, mu, s1, s2 ):
    # Piece-wise Gaussian
    g1 = np.exp( -0.5*( (x-mu)/s1 )**2 )
    g2 = np.exp( -0.5*( (x-mu)/s2 )**2 )

    g = g1
    g[x>=mu] = g2[x>=mu]
    
    return g