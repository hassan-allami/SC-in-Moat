import numpy as np
from matplotlib import pyplot as plt
from scipy.special import lambertw as W

g = .1

dc = -4/g*W(-g/(4*np.sqrt(2)),-1).real

Ec0 = -np.pi*(g)/W(-(g)/(4*np.sqrt(2)*np.exp(1)),-1).real

print('dc =', dc)
print('Ec0 =', Ec0)

d = np.linspace(1,20,1000)*dc

Ec =  -np.pi*(g+ 4/d)/W(-(g + 4/d)/(4*np.sqrt(2)*np.exp(1)),-1).real

nc = (Ec/(4*np.pi)-1/d)**2/(d*Ec)

plt.plot(d/dc, nc, 'k', lw = 2)
plt.xlabel('$d/d_{c0}$', fontsize = 14)
plt.xlim(1,20)
plt.ylabel('$n_c/p_0^2$', fontsize = 14)
plt.ylim(0,2e-7)
plt.show()
