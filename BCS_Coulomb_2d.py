import numpy as np
from matplotlib import pyplot as plt



# making the mesh
m = 3 # size of the BZ
a = 1 # invert width of the dense area
NR = 150 #size in readial direction
Nth = 60 #angular steps


p1 = np.linspace(0,m,NR)
p1 = (a*p1 - np.tanh(a*(p1-1)) - np.tanh(a))/(a - np.tanh(a))
dp = np.diff(p1)
dp = np.r_[dp, dp[NR-2]]

theta1 = np.linspace(0,2*np.pi,Nth)
dth = 2*np.pi/Nth * np.ones(Nth)

P1, Theta1 = np.meshgrid(p1, theta1)
dP, dTh = np.meshgrid(dp,dth)

#compute |P1 - P2|
def dist(P2, Theta2):
	R = np.sqrt(P2**2 + P1**2 -2*P2*P1*np.cos(Theta2 - Theta1))
	return R


#the parameters

Lambda = 0
g = .1
Ec = 0.45
Ps = 1
Lc = g**2/16

xi = (P1 -1)**2 + Lambda #the band

#preparing starting point
delta0 = Lc*(np.ones(P1.shape) + 0*np.cos(2*Theta1))
delta = np.zeros(P1.shape)

n = 1

plt.figure(n)
plt.contourf(P1,Theta1,delta0/Lc,30)
plt.colorbar()





while n < 2:
	for i in range(30):
		for j in range(NR):
			R = dist(P1[i,j], Theta1[i,j])
			integ1 = delta0/np.sqrt(xi**2 + delta0**2)
			integ2 = g - Ec*np.cos(Theta1 - Theta1[i,j])/(R + Ps)
			integ = integ1 * integ2 /(4*(np.pi)*Nth)
			delta[i,j] = sum(sum(dP*P1*integ))
	
	delta = np.r_[delta[:30,:],np.flipud(delta)[30:,:]]

	delta0 = delta	
	n = n+1
	plt.figure(n)
	plt.contourf(P1,Theta1,delta0/Lc,30)
	plt.colorbar()

plt.figure(21)
plt.plot(theta1, delta0[:,70]/max(delta0[:,70]))

plt.show()

