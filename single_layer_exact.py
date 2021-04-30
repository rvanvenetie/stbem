import numpy as np
from math import sqrt
from scipy.special import expi, erf, expn, erfc

FPI_INV = (4 * np.pi)**-1
PI_SQRT = sqrt(np.pi)


def sign(x, y):
    """ Returns the sign of x -y. """
    if x == y: return 0
    if x < y: return -1
    return 1


def gint_1(z, h):
    """ Returns integral of g_z(x) for [0, h]. """
    return -(
        (2 * sqrt(np.pi) * sqrt(z) * erf(h /
                                         (2 * sqrt(z))) - h * expi(-h**2 /
                                                                   (4 * z))) /
        (4 * np.pi))


def gint_2(z, h, k):
    """ Returns integral of g_z(x) for [h, k] for 0 < h < k. """
    assert h < k
    z_sqrt = 2 * sqrt(z)
    return FPI_INV * (PI_SQRT * z_sqrt * (erf(h / z_sqrt) - erf(k / z_sqrt)) -
                      h * expi(-(h**2 / (4 * z))) + k * expi(-(k**2 /
                                                               (4 * z))))


def fint_1(a, b, h):
    """ Returns integrate f_z(x-y) for x,y in [0,h]^2. """
    if a <= b:
        return 0
    else:
        z = a - b
        return (4 * z * (np.exp(-(h**2 / (4 * z))) *
                         (h**2 - 12 * z) + 12 * z) -
                64 * h * np.sqrt(np.pi) * z**(3 / 2) * erf(h /
                                                           (2 * np.sqrt(z))) +
                (h**4 + 24 * h**2 * z) * expi(-(h**2 /
                                                (4 * z)))) / (96 * np.pi)


def fint_2(a, b, h, k):
    """ Returns integrate f_z(x-y) for x,y in [-h,0]x[0,k]. """
    if a <= b:
        return 0
    else:
        z = a - b
        val = (1 / (192 * np.pi)) * (
            (4 * z * ((-(-1 + np.exp(
                (k * (2 * h + k)) / (4 * z)))) * h**2 + k**2 + 2 * h *
                      (k - 8 * np.exp(
                          (h + k)**2 /
                          (4 * z)) * np.sqrt(np.pi) * np.sqrt(z)) - np.exp(
                              (h * (h + 2 * k)) / (4 * z)) *
                      (k**2 - 12 * z) - 12 * z - 12 * np.exp(
                          (h + k)**2 / (4 * z)) * z + 12 * np.exp(
                              (k *
                               (2 * h + k)) / (4 * z)) * z)) / np.exp(
                                   (h + k)**2 /
                                   (4 * z)) + 64 * h * np.sqrt(np.pi) * z**
            (3 / 2) * erf(h / (2 * np.sqrt(z))) - 64 * k * np.sqrt(np.pi) * z**
            (3 / 2) * erfc(k / (2 * np.sqrt(z))) + 64 *
            (h + k) * np.sqrt(np.pi) * z**(3 / 2) * erfc(
                (h + k) /
                (2 * np.sqrt(z))) + 24 * h**2 * z * expn(1, h**2 / (4 * z)) -
            h**4 * expi(-(h**2 / (4 * z))) - k**4 * expi(-(k**2 / (4 * z))) -
            24 * k**2 * z * expi(-(k**2 / (4 * z))) + (h + k)**2 *
            ((h + k)**2 + 24 * z) * expi(-((h + k)**2 / (4 * z))))
        return val


def fint_3(a, b, h, k):
    """ Returns integrate f_z(x-y) for x,y in [0,h]x[0,k]. """
    if a <= b:
        return 0
    else:
        z = a - b
        return (1 / (192 * np.pi)) * (
            (-4 * np.exp((h * k) / (2 * z)) *
             ((h - k)**2 - 12 * z) * z - 4 * np.exp(h**2 / (4 * z)) * z *
             (-k**2 + 12 * z) + np.exp(k**2 / (4 * z)) *
             (4 * (h**2 - 12 * z) * z + np.exp(h**2 / (4 * z)) *
              (48 * z**2 - 64 * h * np.sqrt(np.pi) * z**
               (3 / 2) * erf(h / (2 * np.sqrt(z))) + 16 *
               (4 * h - 3 * k) * np.sqrt(np.pi) * z**(3 / 2) * erf(
                   (h - k) / (2 * np.sqrt(z))) - 64 * k * np.sqrt(np.pi) * z**
               (3 / 2) * erf(k / (2 * np.sqrt(z))) + h**4 * expi(-(h**2 /
                                                                   (4 * z))) +
               24 * h**2 * z * expi(-(h**2 / (4 * z))) - h**4 * expi(-(
                   (h - k)**2 / (4 * z))) + 4 * h**3 * k * expi(-(
                       (h - k)**2 / (4 * z))) - 6 * h**2 * k**2 * expi(-(
                           (h - k)**2 / (4 * z))) + 4 * h * k**3 * expi(-(
                               (h - k)**2 / (4 * z))) - k**4 * expi(-(
                                   (h - k)**2 /
                                   (4 * z))) - 24 * h**2 * z * expi(-(
                                       (h - k)**2 /
                                       (4 * z))) + 48 * h * k * z * expi(-(
                                           (h - k)**2 /
                                           (4 * z))) - 24 * k**2 * z * expi(-(
                                               (h - k)**2 / (4 * z))) + k**2 *
               (k**2 + 24 * z) * expi(-(k**2 / (4 * z)))))) / np.exp(
                   (h**2 + k**2) /
                   (4 * z)) + 16 * k * np.sqrt(np.pi) * z**(3 / 2) *
            (-1 + erfc(np.abs(h - k) / (2 * np.sqrt(z)))) * np.sign(h - k))


def fint_4(a, b, h, k, l):
    """ Returns integrate f_z(x-y) for x,y in [0,h]x[k,l] with 0 < h < k < l """
    if a <= b: return 0
    else:
        z = a - b
        hk = h - k
        hl = h - l
        expihk = expi(-((h - k)**2 / (4 * z)))
        expihl = expi(-((h - l)**2 / (4 * z)))
        sqrtz = np.sqrt(z)
        sqrtpi = np.sqrt(np.pi)
        z32 = z**(3 / 2)
        return (1 / (192 * np.pi)) * (
            4 * z * ((hk**2 - 12 * z) / np.exp(hk**2 / (4 * z)) -
                     (k**2 - 12 * z) / np.exp(k**2 / (4 * z)) +
                     (l**2 - 12 * z) / np.exp(l**2 / (4 * z)) +
                     (-hl**2 + 12 * z) / np.exp(hl**2 / (4 * z))) + 64 *
            (-h + k) * sqrtpi * z32 * erf(hk / (2 * sqrtz)) +
            64 * k * sqrtpi * z32 * erf(k / (2 * sqrtz)) +
            64 * h * sqrtpi * z32 * erf(hl / (2 * sqrtz)) -
            64 * l * sqrtpi * z32 * erf(hl / (2 * sqrtz)) -
            64 * l * sqrtpi * z32 * erf(l / (2 * sqrtz)) + h**4 * expihk -
            4 * h**3 * k * expihk + 6 * h**2 * k**2 * expihk -
            4 * h * k**3 * expihk + k**4 * expihk + 24 * h**2 * z * expihk -
            48 * h * k * z * expihk + 24 * k**2 * z * expihk -
            k**4 * expi(-(k**2 / (4 * z))) - 24 * k**2 * z * expi(-(k**2 /
                                                                    (4 * z))) -
            h**4 * expihl + 4 * h**3 * l * expihl - 6 * h**2 * l**2 * expihl +
            4 * h * l**3 * expihl - l**4 * expihl - 24 * h**2 * z * expihl +
            48 * h * l * z * expihl - 24 * l**2 * z * expihl + l**2 *
            (l**2 + 24 * z) * expi(-(l**2 / (4 * z))))


#def fint_5(a, b, h, k, l):
#    """ Returns integrate f_z(x-y) for x,y in [0,h]x[k,l] with 0 < h,  k < l """
#    if a <= b: return 0
#    else:
#        z = a - b
#        return(1/(192*np.pi))*(4*z*(((h-k)**2-12*z)/np.exp((h-k)**2/(4*z))-(k**2-12*z)/np.exp(k**2/(4*z))+(l**2-12*z)/np.exp(l**2/(4*z))+(-(h-l)**2+12*z)/np.exp((h-l)**2/(4*z)))+16*(-4*h+3*k)*np.sqrt(np.pi)*z**(3/2)*erf((h-k)/(2*np.sqrt(z)))+48*k*np.sqrt(np.pi)*z**(3/2)*erf(k/(2*np.sqrt(z)))+64*h*np.sqrt(np.pi)*z**(3/2)*erf((h-l)/(2*np.sqrt(z)))-48*l*np.sqrt(np.pi)*z**(3/2)*erf((h-l)/(2*np.sqrt(z)))-48*l*np.sqrt(np.pi)*z**(3/2)*erf(l/(2*np.sqrt(z)))+h**4*expi(-((h-k)**2/(4*z)))-4*h**3*k*expi(-((h-k)**2/(4*z)))+6*h**2*k**2*expi(-((h-k)**2/(4*z)))-4*h*k**3*expi(-((h-k)**2/(4*z)))+k**4*expi(-((h-k)**2/(4*z)))+24*h**2*z*expi(-((h-k)**2/(4*z)))-48*h*k*z*expi(-((h-k)**2/(4*z)))+24*k**2*z*expi(-((h-k)**2/(4*z)))-k**4*expi(-(k**2/(4*z)))-24*k**2*z*expi(-(k**2/(4*z)))-h**4*expi(-((h-l)**2/(4*z)))+4*h**3*l*expi(-((h-l)**2/(4*z)))-6*h**2*l**2*expi(-((h-l)**2/(4*z)))+4*h*l**3*expi(-((h-l)**2/(4*z)))-l**4*expi(-((h-l)**2/(4*z)))-24*h**2*z*expi(-((h-l)**2/(4*z)))+48*h*l*z*expi(-((h-l)**2/(4*z)))-24*l**2*z*expi(-((h-l)**2/(4*z)))+l**2*(l**2+24*z)*expi(-(l**2/(4*z)))+16*np.sqrt(np.pi)*z**(3/2)*((k-k*erfc(np.abs(h-k)/(2*np.sqrt(z))))*np.sign(h-k)+(k-k*erfc(np.abs(k)/(2*np.sqrt(z))))*np.sign(k)+l*(-1+erfc(np.abs(h-l)/(2*np.sqrt(z))))*np.sign(h-l)+l*(-1+erfc(np.abs(l)/(2*np.sqrt(z))))*np.sign(l)))

#def fint(a, b, h, i, j, k):
#    """ Returns integrate f_z(x-y) for z = a-b and
#        x,y in [h,i]x[j,k] with h < i and j < k"""
#    if a <= b: return 0
#    else:
#        z = a - b
#        expihj = expi(-((h-j)**2/(4*z)))
#        expiij = expi(-((i-j)**2/(4*z)))
#        expihk = expi(-((h-k)**2/(4*z)))
#        expiik = expi(-((i-k)**2/(4*z)))
#        print(expihj, expiij, expihk, expiik)
#        return(1/(192*np.pi))*(4*z*(((i-j)**2-12*z)/np.exp((i-j)**2/(4*z))+((h-k)**2-12*z)/np.exp((h-k)**2/(4*z))+(-(h-j)**2+12*z)/np.exp((h-j)**2/(4*z))+(-(i-k)**2+12*z)/np.exp((i-k)**2/(4*z)))+16*(4*h-3*j)*np.sqrt(np.pi)*z**(3/2)*erf((h-j)/(2*np.sqrt(z)))+16*(-4*i+3*j)*np.sqrt(np.pi)*z**(3/2)*erf((i-j)/(2*np.sqrt(z)))-64*h*np.sqrt(np.pi)*z**(3/2)*erf((h-k)/(2*np.sqrt(z)))+48*k*np.sqrt(np.pi)*z**(3/2)*erf((h-k)/(2*np.sqrt(z)))+64*i*np.sqrt(np.pi)*z**(3/2)*erf((i-k)/(2*np.sqrt(z)))-48*k*np.sqrt(np.pi)*z**(3/2)*erf((i-k)/(2*np.sqrt(z)))-h**4*expihj+4*h**3*j*expihj-6*h**2*j**2*expihj+4*h*j**3*expihj-j**4*expihj-24*h**2*z*expihj+48*h*j*z*expihj-24*j**2*z*expihj+i**4*expiij-4*i**3*j*expiij+6*i**2*j**2*expiij-4*i*j**3*expiij+j**4*expiij+24*i**2*z*expiij-48*i*j*z*expiij+24*j**2*z*expiij+h**4*expihk-4*h**3*k*expihk+6*h**2*k**2*expihk-4*h*k**3*expihk+k**4*expihk+24*h**2*z*expihk-48*h*k*z*expihk+24*k**2*z*expihk-(i-k)**2*((i-k)**2+24*z)*expiik+16*np.sqrt(np.pi)*z**(3/2)*(j*(-1+erfc(np.abs(h-j)/(2*np.sqrt(z))))*np.sign(h-j)+(j-j*erfc(np.abs(i-j)/(2*np.sqrt(z))))*np.sign(i-j)-k*(-1+erfc(np.abs(h-k)/(2*np.sqrt(z))))*np.sign(h-k)+k*(-1+erfc(np.abs(i-k)/(2*np.sqrt(z))))*np.sign(i-k)))


def spacetime_integrated_kernel_1(a, b, c, d, h):
    """ Returns kernel integrated in time over [a,b] x [c, d], and in space 
    over [0,h]^2.
    """
    assert a < b and c < d and h > 0
    f_bd = fint_1(b, d, h)
    f_bc = fint_1(b, c, h)
    f_ca = fint_1(a, c, h)
    f_da = fint_1(a, d, h)
    return f_bd - f_bc + f_ca - f_da


def spacetime_integrated_kernel_2(a, b, c, d, h, k):
    """ Returns kernel integrated in time over [a,b] x [c, d], and in space 
    over [-h,0] x [0,k]
    """
    assert a < b and c < d and h > 0
    f_bd = fint_2(b, d, h, k)
    f_bc = fint_2(b, c, h, k)
    f_ca = fint_2(a, c, h, k)
    f_da = fint_2(a, d, h, k)
    return f_bd - f_bc + f_ca - f_da


def spacetime_integrated_kernel_3(a, b, c, d, h, k):
    """ Returns kernel integrated in time over [a,b] x [c, d], and in space 
    over [0,h] x [0,k]
    """
    assert a < b and c < d and h > 0
    f_bd = fint_3(b, d, h, k)
    f_bc = fint_3(b, c, h, k)
    f_ca = fint_3(a, c, h, k)
    f_da = fint_3(a, d, h, k)
    return f_bd - f_bc + f_ca - f_da


def spacetime_integrated_kernel_4(a, b, c, d, h, k, l):
    """ Returns kernel integrated in time over [a,b] x [c, d], and in space 
    over [0,h] x [k,l] with 0 < h < k < l """
    assert a < b and c < d and h > 0
    f_bd = fint_4(b, d, h, k, l)
    f_bc = fint_4(b, c, h, k, l)
    f_ca = fint_4(a, c, h, k, l)
    f_da = fint_4(a, d, h, k, l)
    return f_bd - f_bc + f_ca - f_da


#def spacetime_integrated_kernel(a, b, c, d, h, i, j, k):
#    """ Returns kernel integrated in time over [a,b] x [c, d], and in space
#    over [h,i] x [j,k]
#    """
#    assert a < b and c < d and h > 0
#    f_bd = fint(b, d, h, i, j, k)
#    f_bc = fint(b, c, h, i, j, k)
#    f_ca = fint(a, c, h, i, j, k)
#    f_da = fint(a, d, h, i, j, k)
#    return f_bd - f_bc + f_ca - f_da


def spacetime_evaluated_1(t, a, b, h):
    result = 0
    if t > a: result -= gint_1(t - a, h)
    if t > b: result += gint_1(t - b, h)
    return result


def spacetime_evaluated_2(t, a, b, h, k):
    result = 0
    if t > a: result -= gint_2(t - a, h, k)
    if t > b: result += gint_2(t - b, h, k)
    return result
