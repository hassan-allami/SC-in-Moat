import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as intg

#Ec is the Coulomb energy
#L is the Lambda
#g is the attraction
# every energy is normalized to E_R = \kappa*p0^2
# every length is nomalized to 1/p0


'''READING THE GRID AND COULOMB FUNCTIONS'''

d = 100
N = 200

#reading the grid info

f = open('grid_d'+str(d)+'_N_'+str(N)+'.txt')

data = f.read()
data = data.split()
data = np.array(list(map(float, data)))
m = int(len(data)/2)
data = np.reshape(data, (m, 2))
x = data[:, 0]
dx = data[:, 1]

f.close()

#reading the Coulomb functions info

f_d = open('angle_delta_d'+str(d)+'_N_'+str(N)+'.txt')
f_p = open('angle_phi_d'+str(d)+'_N_'+str(N)+'.txt')

angle_delta = f_d.read()
angle_delta = angle_delta.split()
angle_delta = np.array(list(map(float, angle_delta)))
angle_delta = np.reshape(angle_delta, (len(x), len(x)))


angle_phi = f_p.read()
angle_phi = angle_phi.split()
angle_phi = np.array(list(map(float, angle_phi)))
angle_phi = np.reshape(angle_phi, (len(x), len(x)))

f_d.close()
f_p.close()

print(sum(sum(np.isnan(angle_delta))))

'''SOLVER'''

def solve(g, L, Ec):

	#the band
	xi = (x-1)**2 + L
	
	Lc0 = g**2/16 #critcal L in absence of Coulomb that we use as the initial value of delta

	#setting initial values

	delta0 = Lc0*np.ones(len(x)) #flat initial value of delta
	phi0 = np.zeros(len(x)) #flat zero for initial phi

	err = 1 #error
	j = 0 #counter
	while err > 1e-6 and j < 1000:
		delta = np.zeros(len(x)) #new delta
		phi = np.zeros(len(x)) #new phi
		for i in range(len(x)):
			integ1 = x*delta0/np.sqrt((xi + phi0)**2 + delta0**2)
			integ2 = (g - Ec*angle_delta[i, :])/(4*np.pi)
			integ3 = x*(1- (xi + phi0)/np.sqrt((xi + phi0)**2 + delta0**2))
			integ4 = Ec*(d - angle_phi[i, :])/(4*np.pi)			
			delta[i] = sum(dx*integ1*integ2)
			phi[i] = sum(dx*integ3*integ4)

		#this is the error that's supposed to converge if there is a solution
		err = sum(np.abs(delta - delta0))/sum(np.abs(delta))
		delta0 = (delta + delta0)/2
		phi0 = (phi + phi0)/2

		j += 1
		print('#',j)
		print(np.log(err))
	
	return delta, phi, err

'''COMPUTE AND PLOT'''

g = 0.1

Lc0 = g**2/16

#xi = (x-1)**2 + L
#plt.plot(x, (delta/np.sqrt((xi + phi)**2 + delta**2)), '.')
#plt.plot(x, (1- (xi + phi)/np.sqrt((xi + phi)**2 + delta**2)), '.')

#print('min_phi = ', min(phi))
#print('min_delta = ', min(delta))

delta, phi, err = solve(g, 0, 0.05)
#delta1, phi1, err1 = solve(g, Lc0/10, 0.05)
#delta3, phi3, err3 = solve(g, -Lc0/10, 0.05)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size = 13)

plt.subplot(2,1,1)
plt.plot(x,delta/Lc0, '-k', lw = 2)
plt.xlim(0,2)
plt.ylabel('$\Delta / \Lambda_{c0}$',fontsize = 16)
plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
plt.tick_params(direction='in', length = 10)

plt.subplot(2,1,2)
plt.plot(x,phi/Lc0, '-k', lw = 2)
plt.xlim(0,2)
plt.xlabel('$p/p_0$',fontsize = 16)
plt.ylabel('$\Phi / \Lambda_{c0}$',fontsize = 16)
plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
plt.tick_params(direction='in', length = 10)

'''
Ec = np.linspace(3.14, 3.15, 50)
err = np.zeros(len(Ec))
min_delta = np.zeros(len(Ec))
min_phi = np.zeros(len(Ec))
ukvk = np.zeros(len(Ec))
delta = np.zeros((len(Ec),len(x)))
phi = np.zeros((len(Ec),len(x)))

for i in range(len(Ec)):
	delta[i, :], phi[i, :], err[i] = solve(g, L, Ec[i])
	ukvk[i] = sum(dx*x*delta[i,:]/np.sqrt((xi + phi[i, :])**2 + delta[i,:]**2))/(4*np.pi)
	min_delta[i] = min(delta[i,:])
	min_phi[i] = min(phi[i,:])


plt.figure(1)

plt.subplot(1,2,1)
plt.title('$\Lambda = 0$,  $g = 0.33\kappa$, $dp_0$ = 10', fontsize = 12)
#plt.plot(Ec, ukvk, label = '$\sum_k u_k v_k$')
plt.plot(Ec, min_phi, '.r', label = 'min $\Phi$')
plt.plot(Ec, min_delta, '.g', label = 'min $\Delta$')
plt.axhline(y = 0, ls ='--', color = 'k')
plt.xlabel('$Ec/E_R$', fontsize = 14)
plt.legend(loc = 1, fontsize = 14)

plt.subplot(1,2,2)
plt.title('$\Lambda = 0$,  $g = 0.33\kappa$, $dp_0$ = 10', fontsize = 12)
plt.plot(Ec, err,'.')
#plt.axvline(x = 0.3, ls ='--', color = 'k')
plt.xlabel('$Ec/E_R$', fontsize = 14)
plt.ylabel('error', fontsize = 14)



#for Ec = 0.3
Lc0 = g**2/16
L = -0.04*Lc0
Ec = .02

delta, phi, err = solve(g, L, Ec)

xi = (x-1)**2 + L
A = x*delta/np.sqrt((xi + phi)**2 + delta**2)

plt.figure()
plt.title('$\Lambda = 0$,  $g = 0.33\kappa$, $dp_0$ = 10, $E_c = 0.3 E_R$', fontsize = 12)
plt.plot(x, delta, label = '$\Delta/E_R$')
plt.plot(x, phi, label = '$\Phi/E_R$')
plt.xlabel('$p/p_0$',fontsize = 14)
plt.legend(loc = 3, fontsize = 14)

plt.figure()
plt.plot(x, A)



Ec = 0.02
Lc0 = g**2/16
L = np.linspace(-0.05, 1, 50)*Lc0
err = np.zeros(len(L))
min_delta = np.zeros(len(L))
min_phi = np.zeros(len(L))
delta = np.zeros((len(L),len(x)))
phi = np.zeros((len(L),len(x)))

for i in range(len(L)):
	delta[i, :], phi[i, :], err[i] = solve(g, L[i], Ec)
	min_delta[i] = min(delta[i,:])
	min_phi[i] = min(phi[i,:])

plt.figure(1)

plt.subplot(1,2,1)
plt.title('$E_c = E_R$,  $g = 0.33\kappa$, $dp_0$ = 1000', fontsize = 12)
plt.plot(L/Lc0, min_delta, label = 'min $\Delta$')
plt.plot(L/Lc0, min_phi, 'r', label = 'min $\Phi$')
plt.axvline(x = -0.02, ls ='--', color = 'k')
plt.axvline(x = 0.85, ls ='--', color = 'k')
plt.xlabel('$\Lambda/\Lambda_{c0}$', fontsize = 14)
plt.legend(loc = 1, fontsize = 14)


plt.subplot(1,2,2)
plt.title('$E_c = E_R$,  $g = 0.33\kappa$, $dp_0$ = 1000', fontsize = 12)
plt.plot(L/Lc0, err)
plt.axvline(x = -0.02, ls ='--', color = 'k')
plt.axvline(x = 0.85, ls ='--', color = 'k')
plt.xlabel('$\Lambda/\Lambda_{c0}$', fontsize = 14)
plt.ylabel('error', fontsize = 14)

'''
plt.show()
