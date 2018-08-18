import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as intg

#Ec is the Coulomb energy
#L is the Lambda
#g is the attraction
# every energy is normalized to E_R = \kappa*p0^2
# every length is nomalized to 1/p0

'''DEFINING ANGULAR FUNCTIONS'''

# angle_1 is the one appears in the Delta equation

def angle_1(a, b):
	integrand = lambda t: np.cos(t)/np.sqrt(1-a*np.cos(t))*(
				1-np.exp(-b*np.sqrt(1-a*np.cos(t))))
	I, err = intg.quad(integrand, 0, 2*np.pi)
	return I/(2*np.pi)

# angle_2 is the one appears in the Phi equation

def angle_2(a, b):
	integrand = lambda t: (1-np.exp(-b*np.sqrt(1-a*np.cos(t))))/np.sqrt(1-a*np.cos(t))
	I, err = intg.quad(integrand, 0, 2*np.pi)
	return I/(2*np.pi)


N = 200 #determines fineness
d = 2000 #gate separation in unit of 1/p0




'''MAKING THE GRID AND COULOMB FUNCTIONS'''

	
logx = np.linspace(-10, 0, N)
xl = 1 - 10**logx
xl = np.flip(xl,0)
xr = 1 + 10**logx
x = np.r_[xl, [1], xr]
#x = np.linspace(0,2,N)
x[1] = 1e-12


#forming dx
dx = np.diff(x)
dx = np.r_[dx, dx[len(x)-2]]

f = open('grid_d'+str(d)+'_N_'+str(N)+'.txt', 'w')

for i in range(len(x)):
	f.write(str(x[i]) + ' ' + str(dx[i]) + '\n')
	
f.close()




#angular part for the delta equation

f_d = open('angle_delta_d'+str(d)+'_N_'+str(N)+'.txt', 'w')
f_p = open('angle_phi_d'+str(d)+'_N_'+str(N)+'.txt', 'w')

angle_delta = np.zeros((len(x), len(x)))
angle_phi = np.zeros((len(x), len(x)))
for i in range(len(x)):
	print(i)
	for j in range(len(x)):
		angle_delta[i,j] = angle_1(2*x[i]*x[j]/(x[i]**2 + x[j]**2), 
				d*np.sqrt(x[i]**2 + x[j]**2))/np.sqrt(x[i]**2 + x[j]**2)
		angle_phi[i,j] = angle_2(2*x[i]*x[j]/(x[i]**2 + x[j]**2), 
				d*np.sqrt(x[i]**2 + x[j]**2))/np.sqrt(x[i]**2 + x[j]**2)

		f_d.write(str(angle_delta[i,j]) + '\n')
		f_p.write(str(angle_phi[i,j]) + '\n')		

f_d.close()
f_p.close()


f = open('angle_delta_d'+str(d)+'_N_'+str(N)+'.txt')

data = f.read()
data = data.split()
data = np.array(list(map(float, data)))
data = np.reshape(data, (len(x), len(x)))

f.close()


plt.contourf(x,x, data)
plt.colorbar()

plt.show()
