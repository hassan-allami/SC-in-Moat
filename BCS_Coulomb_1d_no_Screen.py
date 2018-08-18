import numpy as np
from matplotlib import pyplot as plt
from scipy import special as esp

#Ec is the Coulomb energy
#L is the Lambda
#g is the attraction
# every energy is normalized to E_R = \kappa*p0^2
# every length is nomalized to 1/p0 
c = 2 - 4*np.sqrt(2)/np.pi #is the constant in the coulomb aprox
A = np.exp(1)*np.pi**(3/2)/(np.sqrt(2)*esp.gamma(5/4)**2)
A = A.real #is the constant we need for L = 0


'''DEFINING COULOMB FOR SPECIAL CASE OF NO SCREENING'''

def Coul(a):
	I = 4/(a*np.sqrt(1+a))*(esp.ellipk(2*a/(1+a+1e-15))
				-(1+a)*esp.ellipe(2*a/(1+a+1e-15)))
	return I/(2*np.pi)


''' BUILDING THE SOLVER '''

def solver(g, Ec, L):

	# this is critcal L in absence of Coulomb. here I'm using it as energy scale.
	Lc0 = g**2/16 
	N = 2000 #determines fineness

	#forming the grid

	if L >= 0:
		logx = np.linspace(-10, 0, N)
		xl = 1 - 10**logx
		xl = np.flip(xl,0)
		xr = 1 + 10**logx
		x = np.r_[xl, [1], xr]
		x = x[1:]
	else:
		logx = np.linspace(-10, 0, int(N/2))
		xl = 1 -np.sqrt(-L) - 10**logx*(1 -np.sqrt(-L))
		xl = np.flip(xl, 0)
		xcl = 1 -np.sqrt(-L) + 10**logx*(np.sqrt(-L))
		xcr = 1 +np.sqrt(-L) - 10**logx*(np.sqrt(-L))
		xcr = np.flip(xcr, 0)
		xr = 1 +np.sqrt(-L) + 10**logx*(1 -np.sqrt(-L))
		x = np.r_[xl, [1 - np.sqrt(-L)], xcl, xcr, [1 + np.sqrt(-L)], xr]
		x = x[1:]

	#forming dx
	dx = np.diff(x)
	dx = np.r_[dx, dx[len(x)-2]]

	#the band
	xi = (x-1)**2 + L 

	#forming Coulomb term
	X, Y = np.meshgrid(x, x)
	colmb = Coul(2*X*Y/(X**2 + Y**2))/np.sqrt(X**2 + Y**2)

	#setting initial values

	delta0 = Lc0*np.ones(len(x)) #flat initial value of delta
	phi0 = np.zeros(len(x)) #flat zero for initial phi
	err = 1 #error
	j = 0 #counter


	while err > 1e-5 and j < 50:
		#new delta
		delta = np.zeros(len(x))
	
		for i in range(len(x)):
			integ1 = x*delta0/np.sqrt((xi + phi)**2 + delta0**2)
			integ2 = (g - Ec*colmb[:, i])/(4*np.pi)
			delta[i] = sum(dx*integ1*integ2)
	
		#this is the error that's supposed to converge if there is a solution
		err = sum(np.abs(delta - delta0))/sum(np.abs(delta))
		delta0 = delta
	
		j += 1
		print('#',j)
		print(np.log(err))
		

		plt.figure(0)#this is keeping track of error
		plt.plot([j], [np.log(err)] , 'ro')
	
	#plt.show()

	return x, delta


'''PLOTTING TWO EXAMPLES FOR g = 0.1 '''

g = 0.1 #setting g

# plot for L = 0 and Ec = 0.05 

x, delta = solver(g, 0.05, 0)
x, d = solver(g, 0, 0)#this is without Coulomb to be used as normalization scale
d0 = min(d)

plt.figure(1)

plt.subplot(2,1,1)
plt.title('$\Lambda = 0$,  $E_c = 0.05 E_R$', fontsize = 14)
plt.plot(x, delta/d0, 'k', lw = 2)
plt.xlim((x[0],2))
plt.xlabel('$p/p_0$', fontsize = 14)
plt.ylabel('$\Delta / \Delta_{0}$', fontsize = 14)


# plot for L = -0.01 and Ec = 0.02

x, delta = solver(g, 0.02, -0.01)
x, d = solver(g, 0, -0.01)#this is without Coulomb to be used as normalization scale
d0 = min(d)

plt.figure(1)

plt.subplot(2,1,2)
plt.title('$\Lambda = -0.01 E_R$,  $E_c = 0.02 E_R$ ', fontsize = 14)
plt.plot(x, delta/d0, 'k', lw = 2)
plt.xlim((x[0],2))
plt.xlabel('$p/p_0$', fontsize = 14)
plt.ylabel('$\Delta / \Delta_{0}$', fontsize = 14)

plt.show()
'''



g = 0.1
Lc0 = g**2/16
x, delta = solver(g, .0634, 0)

print('min delta =', min(delta)/Lc0)
'''
