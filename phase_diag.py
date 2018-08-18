import numpy as np
from matplotlib import pyplot as plt
from scipy.special import lambertw as W

g = 0.1

dc = -4/g*W(-g/(4*np.sqrt(2)),-1).real

Lc0 = g**2/16
Lc1 = Lc0/(W(-g/(np.exp(1)*4*np.sqrt(2)),-1).real)**2

print('dc =', dc)
print('Lc0 =', Lc0)
print('Lc1 =', Lc1)
print('Lc1/Lc0 =', Lc1/Lc0)
print('log(10) =', np.log(10))

L = np.linspace(1e-10,1,200)*Lc1
d_inv = np.linspace(1e-10,1,100)/dc
l, d =np.meshgrid(L, d_inv)

G = 4*np.sqrt(l) - 4*d*(1+np.sqrt(l)/d)*np.log(d*(1+np.sqrt(l)/d)/np.sqrt(2))

print(G.shape)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size = 13)

plt.contour(l/Lc1,d*dc,G, [.1], colors=('k'), linewidths = (2))
plt.xlim(-0.5,1)
plt.xlabel('$\Lambda/\Lambda_{c1}$',fontsize = 16)
plt.ylim(0,1.1)
plt.ylabel('$d_{c0}/d$',fontsize = 16)
plt.tick_params(direction='in', length = 10)
plt.title('$g = 0.1 \kappa$', fontsize = 16)
plt.show()
