import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as intg

def Coulomb(a, d):
	integrand = lambda t: np.cos(t)/np.sqrt(1-a*np.cos(t))*(
				1-np.exp(-d*np.sqrt(1-a*np.cos(t))))
	I, err = intg.quad(integrand, 0, 2*np.pi)
	return I



N = 100 #size in readial direction
m = 3
a = 0.7
p = np.linspace(1e-15,m,N)
p = (a*p - np.tanh(a*(p-1)) - np.tanh(a))/(a - np.tanh(a))
dp = np.diff(p)
dp = np.r_[dp, dp[N-2]]

print('index', p[33])

Lambda = 0
g = .135
Ec = .1

Lc = g**2/16
d = 3000

xi = (p -1)**2 + Lambda #the band

p1, p2 = np.meshgrid(p, p)

print('p1 size is', p1.shape)


colmb = np.zeros(p1.shape)
P = np.sqrt(p1**2 + p2**2)
alpha = 2*p1*p2/(P**2)

for i in range(len(p)):
	j = 0
	while j <= i:
		colmb[i, j] = Coulomb(alpha[i,j], d*P[i,j])/(2*np.pi*P[i,j])
		j = j + 1


colmb = colmb.T + colmb - np.diag(np.diag(colmb))


plt.figure(0)
plt.contourf(p1, p2, colmb,30)
plt.axis('equal')
plt.colorbar()


delta0 = Lc*np.ones(len(p))/100
delta = np.zeros(len(p))

j = 0
err = 1
while err > 1e-5 and j <2000:
	oldmean = np.mean(delta0)
	for i in range(len(p)):
		integ1 = p*delta0/np.sqrt(xi**2 + delta0**2)
		integ2 = (g - Ec*colmb[:, i])/(4*np.pi)
		delta[i] = sum(dp*integ1*integ2)

	newmean = np.mean(delta)
	err = np.abs(oldmean - newmean)/newmean
	
	j += 1
	delta0 = delta
		

	plt.figure(3)
	plt.plot([j], [np.log(err)] , 'ro')


plt.figure(1)
plt.plot(p, integ1, 'x', p, delta[33]/np.sqrt(xi**2 + delta[33]**2))
plt.xlim((0, 3))

plt.figure(2)
plt.plot(p,delta/Lc, 'x')
#plt.xlim((0.5, 2))
#plt.ylim((-.01,.1))
plt.axhline(y = 0, color = 'r')
	
plt.show()

