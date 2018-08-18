import numpy as np
from scipy import special as esp
from scipy import linalg
from matplotlib import pyplot as plt
from scipy import integrate as intg
from scipy.special import gamma


#Ec is the Coulomb energy
#L is the Lambda
#g is the attraction
# every energy is normalized to E_R = \kappa*p0^2
# every length is nomalized to 1/p0 
c = 2 - 4*np.sqrt(2)/np.pi #is the constant in the coulomb aprox
A = 4*gamma(5/4)**2/np.sqrt(np.pi)
A = A.real #is the constant we need for L = 0


'''DEFINING THE GENERAL COULOMB FUNCTION'''

def Coulomb(a, d):
	integrand = lambda t: np.cos(t)/np.sqrt(1-a*np.cos(t))*(
				1-np.exp(-d*np.sqrt(1-a*np.cos(t))))
	I, err = intg.quad(integrand, 0, 2*np.pi)
	return I/(2*np.pi)

def coulomb(x,d):
	X, Y = np.meshgrid(x,x)
	R = np.sqrt(X**2 + Y**2)
	a = 2*X*Y/R**2	
	A = np.zeros(R.shape)
	for i in range(len(x)):
		for j in range(len(x)):
			A[i, j] = Coulomb(a[i, j], d*R[i,j])/R[i, j]
	return A


'''DEFINING COULOMB FOR SPECIAL CASE OF NO SCREENING'''

def Coul(a):
	I = 4/(a*np.sqrt(1+a))*(esp.ellipk(2*a/(1+a+1e-15))
				-(1+a)*esp.ellipe(2*a/(1+a+1e-15)))
	return I/(2*np.pi)



'''MAKING THE MESH'''

m = 5 # means that size of the mesh is m*p0
dx = 0.1 # tells how fine the mesh is
x = np.arange(1e-15,m,dx)
X, Y = np.meshgrid(x,x) # this makes the grid


''' FINDING THE BOUND STATE ENERGY FOR GENERAL CASE'''

def Bound_E(g,L,Ec,d):
	xi = (x-1)**2 + L # this is the band
	M = 2*np.diag(xi) + dx*X*(Ec*coulomb(x,d) - g)/(2*np.pi)
	#M is the reduced Hamiltonain matrix
	la, v = linalg.eigh(M) #solve it
	E = min(la)
	return E


''' FINDING THE BOUND STATE ENERGY FOR NO-SCREENING'''

def bound_energy(g,L,Ec):
	xi = (x-1)**2 + L # this is the band
	A = Ec*Coul(2*X*Y/(X**2+Y**2))/np.sqrt(X**2+Y**2) #this is the Coulomb part
	M = 2*np.diag(xi) + dx*X*(A - g)/(2*np.pi)
	#M is the reduced Hamiltonain matrix
	la, v = linalg.eigh(M) #solve it
	E = min(la)
	return E



''' FINDING THE CRTICAL g for GENERAL CASE'''

def Crit_g(L,Ec,d):
	g = np.linspace(0,.03,30)
	E = np.zeros(len(g))
	for i in range(len(g)):
		E[i] = Bound_E(g[i],L,Ec,d) 
	gc = g[np.argmin(np.abs(E))] 
	#find the g for which "absolute E" is the smallest
	return gc


''' FINDING THE CRTICAL g for NO SCREENING CASE'''

def critical_g(L,Ec):
	g0L = 4*np.sqrt(L)
	E = Ec + 1e-10
	g0E = np.sqrt(2)*c*E*np.log(A/(np.sqrt(2)*c*E))
	print('lower limit = ', (g0L + g0E/2))
	print('upper limit = ', 2*(g0L + g0E))
	g = np.linspace(g0L + g0E, 2*(g0L + g0E), 60)
	E = np.zeros(len(g))
	for i in range(len(g)):
		E[i] = bound_energy(g[i],L,Ec) 
	gc = g[np.argmin(np.abs(E))] 
	#find the g for which "absolute E" is the smallest
	return gc



''' PLOT gc VS L FOR A FIXED Ec for NO-SCREENING CASE'''

#this is for Ec = 1

f = open('gc_L_Ec1.txt')

data = f.read()
data = data.split()
data = np.array(list(map(float, data)))
m = int(len(data)/2)
data = np.reshape(data, (m, 2))
gc = data[:, 1]
L = data[:, 0]
f.close()

gc_aprox = (np.log(64/L)-4)/(2*np.pi) #analytic aproximation
gc_aprox_1 = (np.log(2/L))/(2*np.pi) #analytic aproximation

# we plot it here below

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=14)

plt.figure()
plt.axvline(x = 0, color = 'k' ,lw = 0.5 , ls = '--')
#plt.plot(L, gc_aprox, '-k', lw =2, label = 'analytic approximate result')
plt.plot(L, gc_aprox_1, '-k', lw =2, label = 'analytic approximate result')
plt.plot(L, gc, '--k', lw =2, label = 'numerical result')
plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
plt.xlabel('$\Lambda/E_R$', fontsize = 16)
plt.ylabel('$(g_c - g_{c0})/\kappa$', fontsize = 16)
plt.ylim((0,2))
plt.xlim((-.005,.1))
plt.tick_params(direction='in', length = 10)
plt.legend(loc = 1, fontsize = 16)



''' PLOT gc VS Ec at L = 0 for NO-SCREENING CASE'''

#read from the file

f = open('gc_Ec_L0.txt')

data = f.read()
data = data.split()
data = np.array(list(map(float, data)))
m = int(len(data)/2)
data = np.reshape(data, (m, 2))
gc = data[:, 1]
Ec = data[:, 0]
f.close()

#this is the analytic expression we get from BCS

gc_aprox = Ec*(np.log(16*np.pi**2/(A*Ec))-1)/np.pi

#we plot them together

plt.figure()

plt.plot(Ec, gc, '-k', lw = 2, label = 'numerical result' + '\n' + 'derived from pair wave-function')
plt.plot(Ec, gc_aprox, '--k', lw = 2, label = 'analytic expression' + '\n' + 'derived from BCS wave-function')
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
plt.xlabel('$E_c/E_R$', fontsize = 16)
plt.ylabel('$g_c/\kappa$', fontsize = 16)
plt.ylim((0,3.2))
plt.xlim((0,5))
plt.tick_params(direction='in', length = 10)
plt.legend(loc = 4, fontsize = 16)


#here I'm ploting correction to gc vs d for strong screening case for 3 defferen Lambda from the data I collected in the .txt files below
# I collected the data below by adjusting the code manually for each case to get enough accuracy in reasonable time

# this one is for Ec = 10 and Lambda = 0.1
f = open('Ec10L01.txt')

data = f.read()
data = data.split()
data = np.array(list(map(float, data)))
m = int(len(data)/2)
data = np.reshape(data, (m, 2))
gc1 = data[:, 1]
g01 = gc1[0]

f.close()

#this one is for Ec = 10 and Lambda = 0.05 
f = open('Ec10L005.txt')

data = f.read()
data = data.split()
data = np.array(list(map(float, data)))
m = int(len(data)/2)
data = np.reshape(data, (m, 2))
gc2 = data[:, 1]
g02 = gc2[0]

f.close()

#this one is for Ec = 10 and Lambda = 0
f = open('Ec10L0.txt')

data = f.read()
data = data.split()
data = np.array(list(map(float, data)))
m = int(len(data)/2)
data = np.reshape(data, (m, 2))
gc3 = data[:, 1]

f.close()


#here I'm plotting them together and wiht the analytic approximation

d = np.linspace(0, .1, 10)
Ec = 10

plt.figure()
plt.plot(d, 2*Ec/3/np.pi*d**2, 'k', lw = 2, label = 'analytic approximate result')
plt.plot(d, gc1 - g01, 'ok', color = '0.5', markersize = 10,
 label = 'numerical result for $\Lambda = 0.1 E_R$')
plt.plot(d, gc2 - g02, 's k', markersize = 8, markerfacecolor = 'w',
 label = 'numerical result for $\Lambda = 0.05 E_R$')
plt.plot(d, gc3, '< k', markersize = 8,
 label = 'numerical result for $\Lambda = 0$')
plt.axvline(x = 0, lw = 0.5 , ls = '--', c = 'k')
plt.axhline(y = 0, lw = 0.5 , ls = '--', c = 'k')
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
plt.xlabel('$dp_0$', fontsize = 16)
plt.ylabel('$(g_c - g_{c0})/\kappa$', fontsize = 16)
plt.tick_params(direction='in', length = 10)
plt.legend(loc = 2, fontsize = 15)

plt.show()
