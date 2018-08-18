import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


N = 3*50/np.pi # is a/pi/M
x, y = np.mgrid[-3:3:50j, -3:3:50j]
X = x.flatten()
Y = y.flatten()
Rx1 = np.outer(np.ones(X.size), X)
Rx2 = Rx1.T
Ry1 = np.outer(np.ones(Y.size), Y)
Ry2 = Ry1.T
R = np.sqrt((Rx1-Rx2)**2 + (Ry1 -Ry2)**2) + 1e-10


def energy(g, L, Ec, d):
	xi = (np.sqrt(x**2 + y**2) - 1)**2 + L
	Xi = xi.flatten()
	delta = 2*np.diag(Xi)
	M = delta -g/(N**2)
	la, v = linalg.eigh(M)
	E = np.min(la)
	return E, v, xi

E, v, xi = energy(1.27,.1,0,0)
print('bound state energy =',E,'\n' 
	'approximation says E = 2L - g^2/8 ='
	, .2 - (1.27**2)/8)

m = int(X.size*np.random.random(1)[0])
V = np.reshape(v[:, m], (50,50))

print('the random eig vector is #', m)


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
cs = ax1.contourf(x,y,V,50)
cbar = plt.colorbar(cs)
ax1.set_aspect('equal')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
fig1.show()


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection = '3d')
ax2.plot_wireframe(x,y,xi, rstride = 1, cstride = 1)
fig2.show()
input()
