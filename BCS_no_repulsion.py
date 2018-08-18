import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integ

def fun(L,D):
	integrand = lambda x: 2/np.sqrt(D**2 + (x**2 + L)**2)/np.pi
	result = integ.quad(integrand, 0, np.inf)
	return result[0]

l = np.linspace(-15,1,50)
d = np.linspace(0,2,20)
L, D = np.meshgrid(l,d)
z = np.zeros((len(d),len(l)))

print('l size is',l.shape)

for i in range(len(d)):
	for j in range(len(l)):
		z[i,j] = fun(L[i,j], D[i,j])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.rc('font', size=14)

plt.contour(L,D,z,[1],linewidths = 2, colors = 'k')
plt.plot([0,-1.42406,-8.48527],[1.3932,1.65505,0.6966], 'ko', ms = 7)
plt.text(0.1,1.45, 'A', weight = 'bold', size = 14)
plt.text(-1.45,1.7, 'B', weight = 'bold', size = 14)
plt.text(-8.8,.75, 'C', weight = 'bold', size = 14)
plt.xlim((-15,2))
plt.ylim((0,2))
plt.xticks([1,0,-1.42406, -5, -8.48527, -10, -15],
[1, 0, r'$\tilde{\Lambda}_m$',-5, r'$\tilde{\Lambda}_{\text{cros}}$', -10, -15])
plt.axvline(x = 0, ls ='--', color = 'k')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.axvline(x = -1.42406, ymax = .82, ymin = 0, ls ='--', lw = 1, color = 'k')
plt.axvline(x = -8.48527, ymax = .35, ymin = 0, ls ='--', lw = 1, color = 'k')
plt.xlabel(r'$\tilde{\Lambda}$', fontsize = 16)
plt.ylabel(r'$\tilde{\Delta}$', fontsize = 16)
plt.tick_params(direction='in', length = 10, pad = 8)

# plt.plot(l, np.sqrt(8/3*(1-l)))
# plt.plot(l, 8.47*np.abs(l)*np.exp(-np.pi/2*np.sqrt(np.abs(l))))

plt.show()

