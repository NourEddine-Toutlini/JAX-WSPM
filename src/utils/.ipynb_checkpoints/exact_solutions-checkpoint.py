# analysis/exact_solutions.py
import numpy as np

def exact_solution(x: np.ndarray, y: np.ndarray, t: float, test_number: int, 
                  L: float, alpha0: float, eps0: float, d: float, nt: int = 500):
    """
    Calculate exact solution for given test case.
    
    Args:
        x, y: Coordinates
        t: Time
        test_number: Test case number (1, 2, or 3)
        L: Domain length
        alpha0: Gardner model parameter
        eps0: Initial condition
        d: Model parameter (alpha0 * (thetas - thetar) / Ks)
        nt: Number of terms in series
        
    Returns:
        hex: Exact pressure head
        Sex: Exact saturation
    """
    def calculate_s(k_range, beta, lind_func, mu_func):
        s = 0
        for k in range(1, k_range + 1):
            lind = lind_func(k)
            mu = mu_func(lind, beta)
            s += ((-1)**k) * lind / mu * np.sin(lind*y) * np.exp(-mu*t)
        return s

    if test_number == 1:
        # Test1
        beta1 = np.sqrt(0.25 * alpha0**2 + (np.pi/L)**2)
        beta3 = np.sqrt(0.25 * alpha0**2 + (3*np.pi/L)**2)

        lind_func = lambda k: np.pi * k / L
        mu_func = lambda lind, beta: (beta**2 + lind**2) / d

        s1 = calculate_s(nt, beta1, lind_func, mu_func)
        s2 = calculate_s(nt, beta3, lind_func, mu_func)

        psi1 = 0.75 * np.sin(np.pi*x/L) * (np.sinh(beta1*y)/np.sinh(beta1*L) + (2/(L*d))*s1)
        psi2 = 0.25 * np.sin(3*np.pi*x/L) * (np.sinh(beta3*y)/np.sinh(beta3*L) + (2/(L*d))*s2)
        psi = psi1 - psi2

    elif test_number == 2:
        # Test2
        beta1 = np.sqrt(0.25 * alpha0**2 + (np.pi/L)**2)
        lind_func = lambda k: np.pi * k / L
        mu_func = lambda lind, beta: (beta**2 + lind**2) / d
        s1 = calculate_s(nt, beta1, lind_func, mu_func)
        psi = (np.sinh(beta1*y) / np.sinh(beta1*L) + (2/(L*d)) * s1)
        psi *= np.sin(np.pi*x/L)

    elif test_number == 3:
        # Test3
        beta1 = np.sqrt(0.25 * alpha0**2 + (2*np.pi/L)**2)
        s1 = 0
        for k1 in range(nt + 1):
            lind = np.pi * k1 / L
            mu1 = (0.25 * alpha0**2 + lind**2) / d
            mu2 = (0.25 * alpha0**2 + lind**2 + (2*np.pi/L)**2) / d
            s1 += ((-1)**k1) * lind * (1/mu1 * np.exp(-mu1*t) - 1/mu2 * np.cos(2*np.pi*x/L) * np.exp(-mu2*t)) * np.sin(lind*y)
        psi = np.sinh(0.5*alpha0*y) / np.sinh(0.5*alpha0*L) - np.cos(2*np.pi*x/L) * (np.sinh(beta1*y) / np.sinh(beta1*L)) + (2/(L*d)) * s1

    else:
        raise ValueError("Invalid test number. Please choose 1, 2, or 3.")

    # Calculate hb and then hex, Sex
    if test_number in [1, 2]:
        hb = (1 - eps0) * np.exp(0.5*alpha0*(L-y)) * psi
    else:  # Test3
        hb = 0.5 * (1-eps0) * np.exp(0.5*alpha0*(L-y)) * psi

    log_arg = np.maximum(eps0 + hb, 1e-10)
    hex = np.log(log_arg) / alpha0
    Sex = np.exp(alpha0 * hex)

    return hex, Sex