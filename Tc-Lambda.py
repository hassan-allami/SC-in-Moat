import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integ

def fun(L,T):
	integrand = lambda x: 2*np.tanh((x**2 + L)/(2*T))/(x**2 + L)/np.pi
	result = integ.quad(integrand, 0, np.inf)
	return result[0]

l = np.linspace(-15,1,50)
t = np.linspace(0,1,20)
L, T = np.meshgrid(l,t)
z = np.zeros((len(t),len(l)))

print('l size is',l.shape)

for i in range(len(t)):
	for j in range(len(l)):
		z[i,j] = fun(L[i,j], T[i,j])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.rc('font', size=14)

plt.contour(L,T,z,[1],linewidths = 2, colors = 'k')
plt.plot([0,-1.78432,-8.97095],[0.73582,0.888789,0.367914], 'ko', ms = 7)
plt.text(0.1,0.77, 'A', weight = 'bold', size = 14)
plt.text(-1.78,.92, 'B', weight = 'bold', size = 14)
plt.text(-9,.41, 'C', weight = 'bold', size = 14)
plt.xlim((-15,2))
plt.ylim((0,1))
plt.xticks([1,0,-1.78432, -5, -8.97095, -15],
[1, 0, r'$\tilde{\Lambda}_{m,T_c}$',-5, r'$\tilde{\Lambda}_{{\rm cros},T_c}$', -15])
plt.axvline(x = 0, ls ='--', color = 'k')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.axvline(x = -1.78432, ymax = .88, ymin = 0, ls ='--', lw = 1, color = 'k')
plt.axvline(x = -8.97095, ymax = .35, ymin = 0, ls ='--', lw = 1, color = 'k')
plt.xlabel(r'$\tilde{\Lambda}$', fontsize = 16)
plt.ylabel(r'$\tilde{T}_c$', fontsize = 16)
plt.tick_params(direction='in', length = 10, pad = 8)

# plt.plot(l, np.sqrt(8/3*(1-l)))
# plt.plot(l, 8.47*np.abs(l)*np.exp(-np.pi/2*np.sqrt(np.abs(l))))

plt.show()

