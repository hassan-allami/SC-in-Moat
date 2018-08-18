import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as intg
from scipy.special import lambertw as W
from scipy.special import gamma

#Ec is the Coulomb energy
#L is the Lambda
#g is the attraction
# every energy is normalized to E_R = \kappa*p0^2
# every length is nomalized to 1/p0 
a = 8/np.exp(2) #is the constant in the coulomb aprox
A = 4*gamma(5/4)**2/np.sqrt(np.pi)
A = A.real #is the constant we need for L = 0



'''======================================'''
'''                                      '''
'''        MAKING THE FUNCTIONS          '''
'''                                      '''
'''======================================'''



   ################################
   #   L < 0 (ABOVE BAND CASE)    #
   ################################


'''DEFINE THE INTEGRAL'''

def f0(a):
	if a > 100:
		I = (2*np.log(a) +  2.07944)/a
	else:
		integrand = lambda x: 1/np.sqrt(((x**2-a**2)**2)+1)
		I, err = intg.quad(integrand, 0, np.inf)
	
	return I


'''DEFFINE THE FUNCTION THAT WE ARE MAXIMIZING'''

def Fn(x, L, Ec, g):
	M = x*f0(x*np.sqrt(L))*(g - Ec/(2*np.pi)*np.log(a**2*x**2))
	return M/(2*np.pi)


'''FINDING ITS MAXIMUM'''

def max_Fn(L,Ec,g):
	x0 = np.exp(g*np.pi/Ec)/a
	#x0 is the zero crossing point
	D0 = 64*np.exp(2*W(-g*A*np.exp(1)/(16*np.pi), -1)-2).real
	#D0 is Dc at L = 0
	logx = np.linspace(-1, 0, 250)
	x = x0*10**logx
	#x = np.linspace(1/np.sqrt(2.5*D0),1/np.sqrt(D0),300)	
	F = np.zeros(len(x))
	for i in range(len(x)):
		F[i] = Fn(x[i], L, Ec, g)

	xm = x[np.argmax(F)]
	Fm = F[np.argmax(F)]
	return xm, Fm


'''FINDING THE CRITICAL POINT'''

def crit_Ecn(L, g):
	Ec0 = -g*np.pi/(W(-g*A*np.exp(1)/(16*np.pi), -1)).real
	Ec = Ec0*np.linspace(0.15,1,200)
	Fm = np.zeros(len(Ec))
	xm = np.zeros(len(Ec))
	for i in range(len(Ec)):
		xm[i], Fm[i] = max_Fn(L, Ec[i], g)
	m = np.argmin(np.abs(1 - Fm))
	Ec2 = Ec[m]
	Xm2 = xm[m]
	return Ec2, Xm2





g = 0.01

Lc0 = g**2/16
Lc1 = ((g/3/W(-g*np.exp(2/3)/24, -1))**2).real
D0 = 64*np.exp(2*W(-g*A*np.exp(1)/(16*np.pi), -1)-2).real
Ec0 = -g*np.pi/(W(-g*A*np.exp(1)/(16*np.pi), -1)).real
x0 = np.exp(g*np.pi/Ec0)/a

print('Lc0 =', Lc0)
print('Lc1 =', Lc1)
print('D0 =', D0)
print('Ec0 =', Ec0, '\n')
print('X0/x0 = ', np.log(1/np.sqrt(D0)/x0)/np.log(10))

'''
L = float(input('L = '))

Ec, Xc = crit_Ecn(L, g)
Dc = 1/Xc**2


print('Ec = ', Ec/Ec0)
print('Dc = ', Dc/D0)



L = np.linspace(20*Lc1, 5*Lc0, 50)

fE = open('Ecn3_g001.txt', 'w')
fD = open('Dcn3_g001.txt', 'w')

for i in range(len(L)):
	Ec, Xc = crit_Ecn(L[i], g)
	Dc = 1/Xc**2

	print('Ec =',Ec/Ec0)
	print('Dc =',Dc/D0)
	print(i, '\n')

	fE.write(str(L[i]) + ' ' + str(Ec) + '\n')
	fD.write(str(L[i]) + ' ' + str(Dc) + '\n')

fE.close()
fD.close()

'''
fE = open('Ecn3_g001.txt')
fD = open('Dcn3_g001.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L = dataE[:, 0]

Ec = dataE[:, 1]/Ec0

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc = dataD[:, 1]/D0

fE.close()
fD.close()

plt.plot(L/Lc1, Dc)
plt.figure()
plt.plot(L/Lc1, Ec)
plt.show()


#'''

