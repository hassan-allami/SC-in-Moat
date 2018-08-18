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
	logx = np.linspace(-1, 0, 200)
	x = x0*10**logx
	F = np.zeros(len(x))
	for i in range(len(x)):
		F[i] = Fn(x[i], L, Ec, g)

	xm = x[np.argmax(F)]
	Fm = F[np.argmax(F)]
	return xm, Fm


'''FINDING THE CRITICAL POINT'''

def crit_Ecn(L, g):
	Ec0 = -g*np.pi/(W(-g*A*np.exp(1)/(16*np.pi), -1)).real
	Ec = Ec0*np.linspace(0.45,1.05,300)
	Fm = np.zeros(len(Ec))
	xm = np.zeros(len(Ec))
	for i in range(len(Ec)):
		xm[i], Fm[i] = max_Fn(L, Ec[i], g)
	m = np.argmin(np.abs(1 - Fm))
	Ec2 = Ec[m]
	Xm2 = xm[m]
	return Ec2, Xm2




   ################################
   #   L > 0 (BELOW BAND CASE)    #
   ################################


'''DEFINE THE FUNCTION THAT WE ARE MAXIMIZING'''

def Fp(D, L, Ec, g):
	integrand = lambda x: (g + Ec/np.pi*np.log(x/a))/(
			np.sqrt((x**2 + L)**2 + D**2))
	I, err = intg.quad(integrand, 0, 1)
	return I/(2*np.pi)

'''FINDING ITS MAXIMUM'''

def max_Fp(L, Ec, g):
	D0 = 64*np.exp(2*W(-g*A*np.exp(1)/(16*np.pi), -1)-2).real
	D = np.linspace(1e-11, 2*D0, 200)
	F = np.zeros(len(D))

	for i in range(len(D)):
		F[i] = Fp(D[i], L, Ec, g)
		
	Dm = D[np.argmax(F)]
	Fm = F[np.argmax(F)]

	return Dm, Fm


'''FINDING THE CRITICAL POINT'''

def crit_Ecp(L, g):
	Ec0 = -g*np.pi/(W(-g*A*np.exp(1)/(16*np.pi), -1)).real
	Ec = Ec0*np.linspace(0.95,1.05,100)

	Fm = np.zeros(len(Ec))
	Dm = np.zeros(len(Ec))
	for i in range(len(Ec)):
		Dm[i], Fm[i] = max_Fp(L, Ec[i], g)

	m2 = np.argmin(np.abs(1 - Fm))
	Ec2 = Ec[m2]
	Dm2 = Dm[m2]
		
	return Ec2, Dm2



'''in order to collect accurate enough data in a reasonable amount of time I had to change the range and numbers in the functions accordingly. a sample data colloecting code can be found in the file 'collecting.py'. here below I'm reading out from the already collected data and plot them.'''


'''======================================'''
'''                                      '''
'''               PLOTTING               '''
'''                                      '''
'''======================================'''


'''READING DATA FOR THE CASE g = 0.1'''

g = 0.1

#important values#
Lc0_1 = g**2/16 #first critical L
Lc1_1 = ((g/3/W(-g*np.exp(2/3)/24, -1))**2).real #second critical L
D0_1 = 64*np.exp(2*W(-g*A*np.exp(1)/(16*np.pi), -1)-2).real #critical D at L = 0
Ec0_1 = -g*np.pi/(W(-g*A*np.exp(1)/(16*np.pi), -1)).real #critical Ec at L = 0

''' -5Lc0 < L < -20Lc1 '''

fE = open('Ecn3_g01.txt')
fD = open('Dcn3_g01.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L3 = dataE[:, 0]
L3 = -np.flip(L3, 0)
Ec3 = dataE[:, 1]/Ec0_1
Ec3 = np.flip(Ec3, 0)

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc3 = dataD[:, 1]/D0_1
Dc3 = np.flip(Dc3, 0)

fE.close()
fD.close()


''' -20Lc1 < L < -2Lc1 '''

fE = open('Ecn2_g01.txt')
fD = open('Dcn2_g01.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L2 = dataE[:, 0]
L2 = -np.flip(L2, 0)
Ec2 = dataE[:, 1]/Ec0_1
Ec2 = np.flip(Ec2, 0)

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc2 = dataD[:, 1]/D0_1
Dc2 = np.flip(Dc2, 0)

for i in range(len(Dc2) - 2):
	Dc2[i+1] = (Dc2[i] + Dc2[i+1] + Dc2[i+2])/3 

fE.close()
fD.close()


''' -2Lc1 < L < 0 '''

fE = open('Ecn1_g01.txt')
fD = open('Dcn1_g01.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L1 = dataE[:, 0]
L1 = -np.flip(L1, 0)
Ec1 = dataE[:, 1]/Ec0_1
Ec1 = np.flip(Ec1, 0)

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc1 = dataD[:, 1]/D0_1
Dc1 = np.flip(Dc1, 0)

fE.close()
fD.close()

''' 0 < L < Lc1 '''

fE = open('Ecp_g01.txt')
fD = open('Dcp_g01.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L = dataE[:, 0]
Ec = dataE[:, 1]/dataE[0,1]

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc = dataD[:, 1]/dataD[0,1]

fE.close()
fD.close()

''' Lc1 < L < Lc0 '''

L4 = np.linspace(Lc1_1, Lc0_1, 50)
Ec4 = 2*np.pi*(g- 4*np.sqrt(L4))/np.log(a**2/L4) #this we got analytically
Ec4 = Ec4/Ec0_1

'''combining the data'''

L_1 = np.concatenate((L3, L2, L1, L))
LL_1 = np.concatenate((L3, L2, L1, L, L4))
Ec_1 = np.concatenate((Ec3, Ec2, Ec1, Ec, Ec4))
Dc_1 = np.concatenate((Dc3, Dc2, Dc1, Dc))



'''READING DATA FOR THE CASE g = 0.01'''

g = 0.01

#important values#
Lc0_2 = g**2/16 #first critical L
Lc1_2 = ((g/3/W(-g*np.exp(2/3)/24, -1))**2).real #second critical L
D0_2 = 64*np.exp(2*W(-g*A*np.exp(1)/(16*np.pi), -1)-2).real #critical D at L = 0
Ec0_2 = -g*np.pi/(W(-g*A*np.exp(1)/(16*np.pi), -1)).real #critical Ec at L = 0


''' -5Lc0 < L < -20Lc1 '''

fE = open('Ecn3_g001.txt')
fD = open('Dcn3_g001.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L3 = dataE[:, 0]
L3 = -np.flip(L3, 0)
Ec3 = dataE[:, 1]/Ec0_2
Ec3 = np.flip(Ec3, 0)

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc3 = dataD[:, 1]/D0_2
Dc3 = np.flip(Dc3, 0)

fE.close()
fD.close()


''' -20Lc1 < L < -2Lc1 '''

fE = open('Ecn2_g001.txt')
fD = open('Dcn2_g001.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L2 = dataE[:, 0]
L2 = -np.flip(L2, 0)
Ec2 = dataE[:, 1]/Ec0_2
Ec2 = np.flip(Ec2, 0)

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc2 = dataD[:, 1]/D0_2
Dc2 = np.flip(Dc2, 0)

for i in range(len(Dc2) - 2):
	Dc2[i+1] = (Dc2[i] + Dc2[i+1] + Dc2[i+2])/3 

fE.close()
fD.close()


''' -2Lc1 < L < 0 '''

fE = open('Ecn1_g001.txt')
fD = open('Dcn1_g001.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L1 = dataE[:, 0]
L1 = -np.flip(L1, 0)
Ec1 = dataE[:, 1]/Ec0_2
Ec1 = np.flip(Ec1, 0)

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc1 = dataD[:, 1]/D0_2
Dc1 = np.flip(Dc1, 0)

fE.close()
fD.close()


''' 0 < L < Lc1 '''

fE = open('Ecp_g001.txt')
fD = open('Dcp_g001.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
L = dataE[:, 0]
Ec = dataE[:, 1]/dataE[0,1]

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dc = dataD[:, 1]/dataD[0,1]

fE.close()
fD.close()

''' Lc1 < L < Lc0 '''

L4 = np.linspace(Lc1_2, Lc0_2, 50)
Ec4 = 2*np.pi*(g- 4*np.sqrt(L4))/np.log(a**2/L4) #this we got analytically
Ec4 = Ec4/Ec0_2

'''combining the data'''

L_2 = np.concatenate((L3, L2, L1, L))
LL_2 = np.concatenate((L3, L2, L1, L, L4))
Ec_2 = np.concatenate((Ec3, Ec2, Ec1, Ec, Ec4))
Dc_2 = np.concatenate((Dc3, Dc2, Dc1, Dc))



'''PLOTTING Ec VS LAMBDA'''

plt.plot(LL_1/Lc0_1,Ec_1, 'k', lw = 2, label = '$g = 0.1 \kappa$')
plt.plot(LL_2/Lc0_2,Ec_2, 'k--', lw = 2, label = '$g = 0.01 \kappa$')
plt.axvline(x = 0, ls = '--', color = 'k', lw = 1.5)
plt.xlim((-5, 1))
plt.ylim((0, 1.2))
plt.xlabel('$\Lambda/\Lambda_{c0}$', fontsize = 14)
plt.ylabel(r'$E_c^* / E_{c0}^*$', fontsize = 14)
plt.legend(loc = 3, fontsize = 14)

plt.axes([0.2, 0.6, .25, .25])
plt.plot(LL_1/Lc1_1,Ec_1, 'k', LL_2/Lc1_2, Ec_2, 'k--')
plt.axvline(x = 0, ls = '--', color = 'k', lw = 1)
plt.xlim((-2, .9))
plt.ylim((0.96, 1.03))
plt.xticks([-1, 0],['$-\Lambda_{c1}$', 0])
plt.yticks([0.98, 1])
plt.tick_params(direction='in')

'''PLOTTING Dc VS LAMBDA'''

plt.figure()

plt.plot(L_1/Lc1_1, Dc_1, 'k', lw = 2, label = '$g = 0.1 \kappa$')
plt.plot(L_2/Lc1_2, Dc_2, 'k--', lw = 2, label = '$g = 0.01 \kappa$')
plt.axvline(x = 0, ls = '--', color = 'k', lw = 1.5)
plt.xlim((-100, 1))
plt.ylim((0, 2.5))
plt.xticks([-Lc0_2/Lc1_2, -Lc0_1/Lc1_1, 0, -10, -70, -80, -90, -100], [r'-$\left.\Lambda_{c0}/\Lambda_{c1}\right|_{g = 0.01\kappa}$', r'-$\left.\Lambda_{c0}/\Lambda_{c1}\right|_{g = 0.1\kappa}$', 0, -10, -70, -80, -90, -100])
plt.xlabel('$\Lambda/\Lambda_{c1}$', fontsize = 14)
plt.ylabel(r'$\Delta_c / \Delta_{c0}$', fontsize = 14)
plt.legend(loc = 2, fontsize = 14)

plt.axes([0.2, 0.3, .3, .3])
plt.plot(L_1/Lc1_1, Dc_1, 'k', L_2/Lc1_2, Dc_2, 'k--')
plt.axvline(x = 0, ls = '--', color = 'k', lw = 1)
plt.xlim((-2, 1))
plt.ylim((0, 2))
plt.yticks([0, 1, 2])


'''PLOTTING Lc1/Lc0 VS g'''

logg = np.linspace(-15,0,100)
g = 10**logg
Lc1toLc0 = Lc1_2 = ((4/3/W(-g*np.exp(2/3)/24, -1))**2).real
X = np.concatenate(([-.1, -.1, 0], Lc1toLc0))
Y = np.concatenate(([1, 0, 0], g))

plt.figure()
plt.fill(X, Y, '.7')
plt.plot(Lc1toLc0, g, 'k', lw = 2)
plt.axhline(y = 0, lw = 2, color = 'k')
plt.axvline(x = 0, ls = '--', color = 'k')
plt.xlim((-.05,.1))
plt.ylim((0,0.78))
plt.xlabel('$\Lambda/ \Lambda_{c0}$', fontsize = 14)
plt.ylabel('$g / \kappa$', fontsize = 14)

plt.show()
