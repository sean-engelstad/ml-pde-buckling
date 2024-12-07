def d1_fact(xbar, L):
    return L**(-1.0) * -xbar

def d2_fact(xbar, L):
    return L**(-2.0) * (-1.0 + xbar**2)

def d3_fact(xbar, L):
    return L**(-3.0) * (-1.0 * xbar**3 + 3.0 * xbar)

def d4_fact(xbar,L):
    return L**(-4.0) * (3.0 - 6.0 * xbar**2 + xbar**4)

def d5_fact(xbar, L):
    return L**(-5.0) * (-15.0 * xbar + 10.0 * xbar**3 - xbar**5)

def d6_fact(xbar,L):
    return L**(-6.0) * (-15 + 45 * xbar**2 - 15 * xbar**4 + xbar**6)

def d8_fact(xbar,L):
    return L**(-8.0) * (105 - 420 * xbar**2 + 210 * xbar**4 - 28 * xbar**6 + xbar**8)

def d10_fact(xbar, L):
    return L**(-10.0) * (-945 + 4725 * xbar**2 - 3150 * xbar**4 + 630 * xbar**6 - 45 * xbar**8 + xbar**10)

def d12_fact(xbar, L):
    return L**(-12.0) * (10395 - 62370 * xbar**2 + 51975 * xbar**4 - 13860 * xbar**6 + 1485 * xbar**8 - 66 * xbar**10 + xbar**12)

def d14_fact(xbar, L):
    return L**(-14.0) * (-135135 + 945945 * xbar**2 - 945945 * xbar**4 + 315315 * xbar**6 - 45045 * xbar**8 + 3003 * xbar**10 - 91 * xbar**12 + xbar**14)

def d16_fact(xbar, L):
    return L**(-16.0) * (2027025 - 16216200 * xbar**2 + 18918900 * xbar**4 - 7567500 * xbar**6 + 1351350 * xbar**8 - 120120*xbar**10+ 5460 * xbar**12 - 120 * xbar**14 + xbar**16)