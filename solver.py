from scipy.special import airy, itairy
from scipy.integrate import quad

import numpy as np
import numba as nb
from numba import njit, prange

NUMBA_PROGRESS = False
try:
    from numba_progress import ProgressBar
    NUMBA_PROGRESS = True
except ModuleNotFoundError:
    pass

#####################
## World constants ##
#####################

hbar  = 1.0545718170e-27      # Plank's constant
me    = 9.1093837015e-28      # electron mass
c     = 2.9979245800e+10      # speed of light
el    = 4.8032047125e-10      # electron charge
mc2   = me * c * c            # electron rest energy
alpha = el ** 2 / (hbar * c)  # fine structure constant

################################
## Bessel functions from Airy ##
################################

# Modified Bessel function of 2nd kind of order 1/3 divided by \pi \sqrt{3}
def fun_k13(x):
    y = np.power(1.5 * x, 2.0 / 3.0)
    return airy(y)[0] / np.sqrt(y)

# Modified Bessel function of 2nd kind of order 2/3 divided by \pi \sqrt{3}
def fun_k23(x):
    y = np.power(1.5 * x, 2.0 / 3.0)
    return - airy(y)[1] / y

# Integral from x to +inf of modified Bessel function of 2nd kind of order 1/3 divided by \pi \sqrt{3}
def fun_intk13(x):
    y = np.power(1.5 * x, 2.0 / 3.0)
    return 1.0 / 3.0 - itairy(y)[0]

#################################
## Tables for Bessel functions ##
#################################

TAB_BESSEL_N = 100_000 # number of samples in tables
TAB_BESSEL_XMIN = 1e-3 # left boundary of the table. Can't be 0 exactly due to divergence
TAB_BESSEL_XMAX = 50 # right boundary of the table. Should be large enough so that K_1/3, K_2/3, IK_1/3 are very close to 0
TAB_BESSEL_RANGE = TAB_BESSEL_XMAX - TAB_BESSEL_XMIN
TAB_BESSEL_XS = np.linspace(TAB_BESSEL_XMIN, TAB_BESSEL_XMAX, TAB_BESSEL_N)

try:
    TAB_K13 = np.load('tab_k13.npy')
except FileNotFoundError:
    print("Tabulating K13...", end="")
    TAB_K13 = fun_k13(TAB_BESSEL_XS)
    np.save('tab_k13.npy', TAB_K13)
    print("Saved.")

try:
    TAB_K23 = np.load('tab_k23.npy')
except FileNotFoundError:
    print("Tabulating K23...", end="")
    TAB_K23 = fun_k23(TAB_BESSEL_XS)
    np.save('tab_k23.npy', TAB_K23)
    print("Saved.")

try:
    TAB_INTK13 = np.load('tab_intk13.npy')
except FileNotFoundError:
    print("Tabulating IK13...", end="")
    TAB_INTK13 = fun_intk13(TAB_BESSEL_XS)
    np.save('tab_intk13.npy', TAB_INTK13)
    print("Saved.")
    
#########################################
## Tables for integrated probabilities ##
#########################################

TAB_CHI_N = 10_000                # The range should sufficiently cover the range \chi \sim 1.
TAB_CHI_XMIN_LOG = np.log10(3e-4) # For \chi below 3e-4, the numerical integration is unstable and asymptotic expressions can be used.
TAB_CHI_XMAX_LOG = 3              # For \chi larger than 1e3 asymptotic expressions can be used.
TAB_CHI_RANGE = TAB_CHI_XMAX_LOG - TAB_CHI_XMIN_LOG
TAB_CHI_XS = np.linspace(TAB_CHI_XMIN_LOG, TAB_CHI_XMAX_LOG, TAB_CHI_N)

try:
    TAB_WRAD_INT = np.load('tab_Wrad_int.npy')
except FileNotFoundError:
    print("Tabulating Wrad_int...", end="")
    def wrad(u, chi):
        y = 2 * u / (3 * (1 - u) * chi)
        return ( 2 + u * u / (1 - u) ) * fun_k23(y) - fun_intk13(y)

    fun_Wrad_tot = np.vectorize(lambda chi : quad(wrad, 0, 1, chi)[0])
    TAB_WRAD_INT = fun_Wrad_tot(np.power(10, TAB_CHI_XS))
    np.save('tab_Wrad_int.npy', TAB_WRAD_INT)
    print("Saved.")
    
try:
    TAB_GRAD_INT = np.load('tab_Grad_int.npy')
except FileNotFoundError:
    print("Tabulating Grad_int...", end="")
    def grad(u, chi):
        y = 2 * u / (3 * (1 - u) * chi)
        return u * fun_k13(y)

    fun_Grad_tot = np.vectorize(lambda chi : quad(grad, 0, 1, chi)[0])
    TAB_GRAD_INT = fun_Grad_tot(np.power(10, TAB_CHI_XS))
    np.save('tab_Grad_int.npy', TAB_GRAD_INT)
    print("Saved.")
    
#############################################
## Numba functions for calculating Bessels ##
#############################################

@njit(inline='always', fastmath = True)
def calc_bessels(x):
    i = int((x - TAB_BESSEL_XMIN) * TAB_BESSEL_N // TAB_BESSEL_RANGE)
    if x < TAB_BESSEL_XMIN:
        # z << 1
        # K_\nu(z) -> \Gamma(\nu) * 2^{\nu - 1} / z^{\nu}
        # \Gamma(1/3) / 2^{2/3} / \pi / \sqrt{3} = 0.3101455723097432
        # \Gamma(2/3) / 2^{1/3} / \pi / \sqrt{3} = 0.1975161718471919
        # IK_{1/3}(z) / \pi / \sqrt{3} -> 1 / 3 - 1 / (2^{2/3} \Gamma(2/3)) * z^{2/3}
        # 1 / (2^{2/3} \Gamma(2/3)) = 0.4652183584646147
        tmp = np.power(1.0 / x, 0.3333333333333333)
        return (0.3101455723097432 * tmp,
                0.1975161718471919 * tmp * tmp,
                0.3333333333333333 - 0.4652183584646147 * np.power(x, 0.6666666666666667))
    
    elif i >= TAB_BESSEL_N - 1:
        # z >> 1
        # K_\nu(z) -> \sqrt{ \frac{ \pi }{ 2 z } } \ exp{-z}
        # IK_{1/3}(z) -> \sqrt{ \frac{ \pi }{ 2 z } } \ exp{-z}
        # \sqrt{1 / 6 / \pi} = 0.2303294329808903
        asym = 0.2303294329808903 * np.exp(-x) / np.sqrt(x)
        return asym, asym, asym
    
    else:
        x_a = TAB_BESSEL_XS[i]
        x_b = TAB_BESSEL_XS[i + 1]
        
        denom = 1.0 / (x_b - x_a)
        
        y_a = TAB_K13[i]
        y_b = TAB_K13[i + 1]
        fac = (x - x_a) * denom
        k13 = fac * (y_b - y_a) + y_a
        
        y_a = TAB_K23[i]
        y_b = TAB_K23[i + 1]
        fac = (x - x_a) * denom
        k23 = fac * (y_b - y_a) + y_a
        
        y_a = TAB_INTK13[i]
        y_b = TAB_INTK13[i + 1]
        fac = (x - x_a) * denom
        intk13 = fac * (y_b - y_a) + y_a
        
        return k13, k23, intk13
    
@njit(inline='always', fastmath = True)
def calc_u_integrals(chi):
    x = np.log10(chi)
    i = int((x - TAB_CHI_XMIN_LOG) * TAB_CHI_N // TAB_CHI_RANGE)

    if x < TAB_CHI_XMIN_LOG:
        # \chi \ll 1
        # 5 / 2 / \sqrt{3} = 1.443375672974065 - From Baier-Katkov
        # \sqrt{3} / 4 = 0.4330127018922193 - From Mathematica
        return 1.443375672974065*chi, 0.4330127018922193*chi*chi
        # Numerical approximation for Grad is 0.4288507664083324*chi*chi
    
    # TODO: asymptotics for Grad
    elif x > TAB_CHI_XMAX_LOG:
        # \chi \gg 1
        # 14 \Gamma(2/3) / 3^{7/3} = 1.460500129047159 - From Baier-Katkov
        return 1.460500129047159*np.power(chi, 0.6666666667), TAB_GRAD_INT[-1]
        
    else:
        x_a = TAB_CHI_XS[i]
        x_b = TAB_CHI_XS[i + 1]
        
        denom = 1.0 / (x_b - x_a)
        
        y_a = TAB_WRAD_INT[i]
        y_b = TAB_WRAD_INT[i + 1]
        fac = (x - x_a) * denom
        Wrad_tot = fac * (y_b - y_a) + y_a
        
        y_a = TAB_GRAD_INT[i]
        y_b = TAB_GRAD_INT[i + 1]
        fac = (x - x_a) * denom
        Grad_tot = fac * (y_b - y_a) + y_a
        
        return Wrad_tot, Grad_tot

#############
## Pushers ##
#############

@njit(inline='always', fastmath = True)
def push_higuerra(dt, ux, uy, uz, ex, ey, ez, bx, by, bz):
    # Higuera, A. V. and Cary, J. R., Phys. Plasmas 24, 052104 (2017)
    
    uux = ux - ex * dt * 0.5
    uuy = uy - ey * dt * 0.5
    uuz = uz - ez * dt * 0.5
    
    g2 = 1.0 + uux*uux + uuy*uuy + uuz*uuz
    
    bbx = - dt * bx * 0.5
    bby = - dt * by * 0.5
    bbz = - dt * bz * 0.5

    beta2 = bbx*bbx + bby*bby + bbz*bbz
    betau = uux*bbx + uuy*bby + uuz*bbz
    
    diff = g2 - beta2
    gamma_new2 = 0.5 * diff
    
    diff = np.sqrt( diff * diff + 4.0 * ( beta2 + betau * betau ) )
    gamma_new2 += 0.5 * diff
    
    gamma_new2_inv = 1.0 / gamma_new2
    gamma_new2_inv_sqrt = np.sqrt(gamma_new2_inv)
    
    uux = uux - (bby * uuz - bbz * uuy) * gamma_new2_inv_sqrt + bbx * betau * gamma_new2_inv
    uuy = uuy - (bbz * uux - bbx * uuz) * gamma_new2_inv_sqrt + bby * betau * gamma_new2_inv
    uuz = uuz - (bbx * uuy - bby * uux) * gamma_new2_inv_sqrt + bbz * betau * gamma_new2_inv
    
    mult = 1.0 / (1.0 + beta2 * gamma_new2_inv)
    
    uux *= mult
    uuy *= mult
    uuz *= mult

    n_ux = ux - ex * dt + 2.0 * (uuy * bbz - uuz * bby) * gamma_new2_inv_sqrt
    n_uy = uy - ey * dt + 2.0 * (uuz * bbx - uux * bbz) * gamma_new2_inv_sqrt
    n_uz = uz - ez * dt + 2.0 * (uux * bby - uuy * bbx) * gamma_new2_inv_sqrt
        
    return n_ux, n_uy, n_uz

@njit(inline='always', fastmath = True)
def push_tbmt(dt, sx, sy, sz, ux, uy, uz, ex, ey, ez, bx, by, bz):
    # J. Vieira et al. Phys. Rev. ST Accel. Beams 14, 071303 (2011) 
    # Pusher that conserves |S|
    
    g = np.sqrt(1.0 + ux * ux + uy * uy + uz * uz)
    inv_g = 1.0 / g
    
    amm = 0.0011614 # anomalous magnetic moment of the electron

    vx = ux * inv_g
    vy = uy * inv_g # v_{i+1/2}
    vz = uz * inv_g
    
    vB = vx * bx + vy * by + vz * bz
    vxE_x = vy * ez - vz * ey
    vxE_y = vz * ex - vx * ez
    vxE_z = vx * ey - vy * ex
    
    opg_inv = 1.0 / (1.0 + g)
    mp = amm + opg_inv
    amm_g_vb = amm * g * vB * opg_inv
    
    Omega_x = - 0.5 * dt * (bx * (amm + inv_g) - amm_g_vb * vx - mp * vxE_x)
    Omega_y = - 0.5 * dt * (by * (amm + inv_g) - amm_g_vb * vy - mp * vxE_y)
    Omega_z = - 0.5 * dt * (bz * (amm + inv_g) - amm_g_vb * vz - mp * vxE_z)
    
    O2 = Omega_x*Omega_x + Omega_y*Omega_y + Omega_z*Omega_z 
    d = 2.0 / (1.0 + O2)
    
    s1x = sx + sy * Omega_z - sz * Omega_y
    s1y = sy + sz * Omega_x - sx * Omega_z
    s1z = sz + sx * Omega_y - sy * Omega_x

    n_sx = sx + d * (s1y * Omega_z - s1z * Omega_y)
    n_sy = sy + d * (s1z * Omega_x - s1x * Omega_z)
    n_sz = sz + d * (s1x * Omega_y - s1y * Omega_x)
    
    return n_sx, n_sy, n_sz

######################################################################
## Helper function for choosing between spin-up and spin-down state ##
######################################################################

@njit(inline='always', fastmath = True)
def measure_spin(axis_x, axis_y, axis_z, av_prob, rng):
    axis_mod = np.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
    axis_mod_inv = 1.0 / axis_mod

    # Probability that spin is co-directional with axis
    prob_up = 0.5 * (1.0 + axis_mod / av_prob)

    n_sx = axis_x * axis_mod_inv
    n_sy = axis_y * axis_mod_inv
    n_sz = axis_z * axis_mod_inv

    if rng.random() > prob_up:
        n_sx *= -1.0
        n_sy *= -1.0
        n_sz *= -1.0

    return n_sx, n_sy, n_sz

######################
## Single time-step ##
######################

@njit(fastmath = True)
def step_numba(dt, t, x, y, z, ux, uy, uz, sx, sy, sz, rf_scheme, a0s, fields, rng, recoil, selection):
    ex, ey, ez, bx, by, bz = fields(t, x, y, z)
    
    n_ux, n_uy, n_uz = push_higuerra(dt, ux, uy, uz, ex, ey, ez, bx, by, bz)
    n_g = np.sqrt(1.0 + n_ux**2 + n_uy**2 + n_uz**2)
    n_g_inv = 1.0 / n_g
    
    n_x = x + dt * n_ux * n_g_inv
    n_y = y + dt * n_uy * n_g_inv
    n_z = z + dt * n_uz * n_g_inv
    
    n_sx, n_sy, n_sz = sx, sy, sz

    if rf_scheme == 0:
        n_sx, n_sy, n_sz = push_tbmt(dt, sx, sy, sz, ux, uy, uz, ex, ey, ez, bx, by, bz)
        return n_x, n_y, n_z, n_ux, n_uy, n_uz, n_sx, n_sy, n_sz
    
    # Radiation reaction (rf_scheme > 0)
    g = np.sqrt(1.0 + ux * ux + uy * uy + uz * uz)
    
    m_g = 0.5 * (n_g + g) # energy at the time i where coordinates are defined
    m_g_inv = 1.0 / m_g

    m_ux = 0.5 * (n_ux + ux)
    m_uy = 0.5 * (n_uy + uy) # momentum at i
    m_uz = 0.5 * (n_uz + uz)

    fx = - ex - (m_uy * bz - m_uz * by) * m_g_inv 
    fy = - ey - (m_uz * bx - m_ux * bz) * m_g_inv # force at i
    fz = - ez - (m_ux * by - m_uy * bx) * m_g_inv

    chi = np.sqrt(m_g * m_g * (fx * fx + fy * fy + fz * fz) - np.power(m_ux * fx + m_uy * fy + m_uz * fz, 2)) / a0s
    
    if rf_scheme > 3:
        # Semi-classical radiation reaction
        # TODO: Might use tabulated function here as well
        
        frr = dt * 2.0 / 3.0 * alpha * a0s * chi**2 * m_g_inv

        if rf_scheme == 5:
            frr *= np.power( 1 + 18 * chi + 69 * chi**2 + 73 * chi**3 + 5.804 * chi**4 ,-1/3)
        
        n_ux -= frr * m_ux
        n_uy -= frr * m_uy
        n_uz -= frr * m_uz
        
        n_sx, n_sy, n_sz = push_tbmt(dt, sx, sy, sz, ux, uy, uz, ex, ey, ez, bx, by, bz)
        
        return n_x, n_y, n_z, n_ux, n_uy, n_uz, n_sx, n_sy, n_sz

    elif rf_scheme < 4:
        # Quantum radiation reaction
    
        u = rng.random()
        r = u*u*u # ratio of the emitted photon energy to the electron energy, modified event generator from  A. Gonoskov et al. Phys. Rev. E 92, 023305 (2015)
        
        bessel_arg = 2.0 / 3.0 * r / ( 1.0 - r ) / chi # argument of bessel functions
        k13, k23, intk13 = calc_bessels(bessel_arg)
        wrad = ( 2.0 + r * r / ( 1.0 - r ) )  * k23 - intk13
        
        w_tot = 0
        if ( 1.0 - r ) * m_g < 1.0:
            n_sx, n_sy, n_sz = push_tbmt(dt, sx, sy, sz, ux, uy, uz, ex, ey, ez, bx, by, bz)
            return n_x, n_y, n_z, n_ux, n_uy, n_uz, sx, sy, sz
        
        if rf_scheme == 1:
            w_tot = wrad

        if rf_scheme > 1:
            modu_inv = 1.0 / np.sqrt(m_ux * m_ux + m_uy * m_uy + m_uz * m_uz)

            evx = m_ux * modu_inv
            evy = m_uy * modu_inv # direction along the electron velocity
            evz = m_uz * modu_inv

            f2v = fx * evx + fy * evy + fz * evz

            fx_tr = fx - f2v * evx
            fy_tr = fy - f2v * evy # transverse force
            fz_tr = fz - f2v * evz

            modf_inv = 1.0 / np.sqrt(fx_tr * fx_tr + fy_tr * fy_tr + fz_tr * fz_tr)

            e1x = fx_tr * modf_inv
            e1y = fy_tr * modf_inv # direction along transverse acceleration (force)
            e1z = fz_tr * modf_inv

            e2x = evy * e1z - evz * e1y
            e2y = evz * e1x - evx * e1z # ev x e1
            e2z = evx * e1y - evy * e1x

            sie2 = e2x * sx + e2y * sy + e2z * sz # e2x \cdot si
            
            g_rad = - r * k13
            w_tot = wrad + g_rad * sie2
            
                    
        
        multiplier = dt * alpha * a0s * m_g_inv
        # Total probability to emit in time interval dt
        # 3u^2 is the 'weight' of the current sample due to non-uniform sampling
        w_int = w_tot * 3 * u * u * multiplier
        
        # Sub-stepping. Might be not exactly accurate, since momenta and positions are offset with current delta t
        if w_int > 1:
            n_x, n_y, n_z, n_ux, n_uy, n_uz, n_sx, n_sy, n_sz, = step_numba(0.5*dt, t, x, y, z, ux, uy, uz, sx, sy, sz, rf_scheme, a0s, fields, rng, recoil, selection)
            n_x, n_y, n_z, n_ux, n_uy, n_uz, n_sx, n_sy, n_sz, = step_numba(0.5*dt, t+0.5*dt, n_x, n_y, n_z, n_ux, n_uy, n_uz, n_sx, n_sy, n_sz, rf_scheme, a0s, fields, rng, recoil, selection)
            return n_x, n_y, n_z, n_ux, n_uy, n_uz, n_sx, n_sy, n_sz
        
        # Radiation actually happened
        elif w_int > rng.random():
            
            if recoil:
                n_ux -= r * m_ux
                n_uy -= r * m_uy # recoil
                n_uz -= r * m_uz
            
            if rf_scheme > 1:
                e2_coeff = - r / ( 1.0 - r ) * k13
                si_coeff = 2.0 * k23 - intk13
                ev_coeff = r * r / ( 1.0 - r ) * ( k23 - intk13 ) * (sx * evx + sy * evy + sz * evz)
                
                # Mean polarization axis after emission
                q_axis_x = e2_coeff * e2x + si_coeff * sx + ev_coeff * evx
                q_axis_y = e2_coeff * e2y + si_coeff * sy + ev_coeff * evy
                q_axis_z = e2_coeff * e2z + si_coeff * sz + ev_coeff * evz
                        
        # No radiation polarization (selection effect)
        elif rf_scheme > 1:
            if not selection:
                n_sx, n_sy, n_sz = push_tbmt(dt, n_sx, n_sy, n_sz, ux, uy, uz, ex, ey, ez, bx, by, bz)
                return n_x, n_y, n_z, n_ux, n_uy, n_uz, n_sx, n_sy, n_sz
            
            Wrad_tot, Grad_tot = calc_u_integrals(chi)
            
            # total probability that photon is not emitted during this step
            w_tot = 1.0 - multiplier * (Wrad_tot - Grad_tot * sie2)
            
            e2_coeff = Grad_tot * multiplier
            si_coeff = 1.0 - Wrad_tot * multiplier
            
            q_axis_x = e2_coeff * e2x + si_coeff * sx
            q_axis_y = e2_coeff * e2y + si_coeff * sy
            q_axis_z = e2_coeff * e2z + si_coeff * sz
            
        if rf_scheme == 2:
            n_sx, n_sy, n_sz = measure_spin(q_axis_x, q_axis_y, q_axis_z, w_tot, rng)

        elif rf_scheme == 3:
            n_sx, n_sy, n_sz = q_axis_x / w_tot, q_axis_y / w_tot, q_axis_z / w_tot
            
        n_sx, n_sy, n_sz = push_tbmt(dt, n_sx, n_sy, n_sz, ux, uy, uz, ex, ey, ez, bx, by, bz)

        return n_x, n_y, n_z, n_ux, n_uy, n_uz, n_sx, n_sy, n_sz

def jitify_solve(fields):
    """
    A function which returns a jit-compilable function which solves motion equations in given field configuration
    
    ---
    Parameters:

    fields : callable
        A function of type fields(t, x, y, z) -> ex, ey, ez, bx, by, bz

    """
        
    nb_fields = njit(inline='always', fastmath = True)(fields)
        
    def solve(dt, r0, u0, s0, t0, steps, n_particles = None, n_out = None, rf_scheme = 0, recoil = True, selection = True, lamda = 1e-4, seed = None, progress = False, n_threads = None):
        """
        Solves motions equations for given initial conditions. Different particles are processed in parallel
        ---
        Parameters:
        
        r0, u0, s0 : np.array with shape (3, ) or (N, 3), where N is number of particles
            Initial conditions

        steps : int
            Number of time-steps to perform
    
        n_particles : int
            Number of particles if r0 has shape (3, ). Not used otherwise

        n_out : {int | 'None'}, default 'None'
            Number of entries in return array. If set to 'None', number of entries equals to steps

        rf_scheme : {0, 1, 2, 3, 4, 5}, default 0
            Scheme used to account radiation reaction:
            0 - no radiation reaction
            
            Quantum (stochastic) radiation:
            1 - averaged over polarization
            2 - resolved electron spin, pure states, averaged photon polarization
            3 - resolved electron spin, mixed states, averaged photon polarization
            
            Quasiclassical radiation:
            4 - classical syncrotron expressions
            5 - quantum corrections (g-factor)

        recoil : bool
            Whether to subtract emitted energy from the electron. Only used for rf_scheme 1 to 3

        selection : bool
            Whether to account selection effects (no-radiation polarization). Only used for rf_scheme 2 to 3

        lamda : float, default 1e-4
            Wavelength used for normalization

        seed : {int | 'None'}, default 'None'
            Sets rng seed if defined

        seed : bool, default False
            Whether to show a progress bar. Shows number of proccessed particle.
            Requires numba_progress package (```$ pip install numba_progress```)

        n_threads : {int | 'None'}, default 'None'
            Maximum number of threads to use for parallel proccessing
        ---
        Returns

        sol : np.array
            A numpy array with shape (n_particles, n_out, 10) containing the following values: [t, x, y, z, ux, uy, uz, sx, sy, sz]
        """

        NUMBA_THREADS = nb.config.NUMBA_NUM_THREADS
        if n_threads is not None:
            NUMBA_THREADS = n_threads

        nb.set_num_threads(NUMBA_THREADS)

        rng = np.random.default_rng(seed)
        
        if r0.shape != u0.shape or r0.shape != s0.shape:
            raise Exception("Inconsistent shape of initial conditions. Aborting")
        
        if r0.shape == (3, ):
            if n_particles is None:
                N = 1
            else:
                N = n_particles
        
            r = np.broadcast_to(r0, (N, 3))
            u = np.broadcast_to(u0, (N, 3))
            s = np.broadcast_to(s0, (N, 3))
        
        elif len(r0.shape) == 2 and r0.shape[1] == 3:
            r = r0
            u = u0
            s = s0
            N = r0.shape[0]

        else:
            raise Exception("Initial conditions must be an np.array with shape (3, ) or (N, 3), where N is number of particles. Aborting")
        
        if n_out is None or n_out > steps:
            n_out = steps
        
        sol =  np.zeros((n_particles, n_out, 10))

        omega = 2 * np.pi * c / lamda
        a0s = mc2 / hbar / omega
        
        out_step = int(steps / n_out)
                
        @njit(nogil=True, fastmath = True)
        def loop_single(x, y, z, ux, uy, uz, sx, sy, sz, sol, rng):
            try:
                t = t0

                # half-step back for momenta without radiation reaction
                _, _, _, ux, uy, uz, _, _, _ = step_numba(- 0.5 * dt, t, x, y, z, ux, uy, uz, sx, sy, sz, 0, a0s, nb_fields, rng, recoil, selection)

                k = 0
                for j in range(steps):
                    if not (j % out_step) and k < n_out:
                        sol[k, :] = t, x, y, z, ux, uy, uz, sx, sy, sz
                        k += 1

                    x, y, z, ux, uy, uz, sx, sy, sz = step_numba(dt, t, x, y, z, ux, uy, uz, sx, sy, sz, rf_scheme, a0s, nb_fields, rng, recoil, selection)
                    t += dt
            except Exception:
                return
        
        @njit(nogil=True, fastmath = True, parallel=True)
        def loop_particles_progress(r0, u0, s0, sol, rng, pbar):
            for i in prange(n_particles):
                loop_single(r0[i, 0], r0[i, 1], r0[i, 2], u0[i, 0], u0[i, 1], u0[i, 2], s0[i, 0], s0[i, 1], s0[i, 2], sol[i], rng)
                pbar.update(1)
            return sol
        
        @njit(nogil=True, fastmath = True, parallel=True)
        def loop_particles(r0, u0, s0, sol, rng):
            for i in prange(n_particles):
                loop_single(r0[i, 0], r0[i, 1], r0[i, 2], u0[i, 0], u0[i, 1], u0[i, 2], s0[i, 0], s0[i, 1], s0[i, 2], sol[i], rng)
            return sol
        
        if NUMBA_PROGRESS and progress:
            # TODO: show number of performed steps for a single particle
            pbar = ProgressBar(total = N)
            sol = loop_particles_progress(r, u, s, sol, rng, pbar)
            pbar.close()
        else:
            sol = loop_particles(r, u, s, sol, rng)
        
        return sol
    
    # Precompiling ??
    # solve(0.1, np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 0, 10, 10, rf_scheme = 0)
    
    return solve

######################################################################
## Helper functions to get labeled xarray Dataset from numpy array  ##
######################################################################

try:
    import xarray as xr

    def process_solution(sol):
        """
        Returns an xarray Dataset containing the solution
        """
        ts = sol[0, :, 0]
        ns = np.arange(0, sol.shape[0])
        
        sol_dict = {}
        
        x = xr.DataArray(sol[:, :, 1], coords=[('n',ns), ('t',ts)])
        y = xr.DataArray(sol[:, :, 2], coords=[('n',ns), ('t',ts)])
        z = xr.DataArray(sol[:, :, 3], coords=[('n',ns), ('t',ts)])
        sol_dict["x"] = x
        sol_dict["y"] = y
        sol_dict["z"] = z

        ux = xr.DataArray(sol[:, :, 4], coords=[('n',ns), ('t',ts)])
        uy = xr.DataArray(sol[:, :, 5], coords=[('n',ns), ('t',ts)])
        uz = xr.DataArray(sol[:, :, 6], coords=[('n',ns), ('t',ts)])
        g  = np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)
        sol_dict["ux"] = ux
        sol_dict["uy"] = uy
        sol_dict["uz"] = uz
        sol_dict["g"] = g
        
        sol_dict["sx"] = xr.DataArray(sol[:, :, 7], coords=[('n',ns), ('t',ts)])
        sol_dict["sy"] = xr.DataArray(sol[:, :, 8], coords=[('n',ns), ('t',ts)])
        sol_dict["sz"] = xr.DataArray(sol[:, :, 9], coords=[('n',ns), ('t',ts)])
        
        xr_sol = xr.Dataset(sol_dict)

        return xr_sol

    def calculate_chi(xr_sol, lamda, fields):
        """
        Calculates chi along the given trajectory in given fields
        """
        omega = 2 * np.pi * c / lamda
        a0s = mc2 / hbar / omega

        nb_fields = np.vectorize(njit(inline='always', fastmath = True)(fields))

        ts = xr_sol.t.values
        ns = xr_sol.n.values
        ex, ey, ez, bx, by, bz = nb_fields(xr_sol.t.values, xr_sol.x.values, xr_sol.y.values, xr_sol.z.values)

        ex = xr.DataArray(np.array(ex), coords=[('n',ns), ('t',ts)])
        ey = xr.DataArray(np.array(ey), coords=[('n',ns), ('t',ts)])
        ez = xr.DataArray(np.array(ez), coords=[('n',ns), ('t',ts)])
        bx = xr.DataArray(np.array(bx), coords=[('n',ns), ('t',ts)])
        by = xr.DataArray(np.array(by), coords=[('n',ns), ('t',ts)])
        bz = xr.DataArray(np.array(bz), coords=[('n',ns), ('t',ts)])

        fx = -ex - (xr_sol.uy * bz - xr_sol.uz * by) / xr_sol.g
        fy = -ey - (xr_sol.uz * bx - xr_sol.ux * bz) / xr_sol.g
        fz = -ez - (xr_sol.ux * by - xr_sol.uy * bx) / xr_sol.g

        xr_sol["chi"] = np.sqrt(xr_sol.g * xr_sol.g * (fx**2 + fy**2 + fz**2) - np.power(xr_sol.ux * ex + xr_sol.uy * ey + xr_sol.uz * ez, 2)) / a0s
        return xr_sol

except ModuleNotFoundError:
    pass