''' FOR g = 0.1, 0 > L > -2*L2 '''

g = 0.1

D0 = 2*np.exp(2+2*W(-g/A, -1)).real
L2 = (g/(3*W(-g/(3*np.sqrt(2)*np.exp(4/3)), -1)))**2
L2 = L2.real
L0 = g**2/16
Ec20 = -g/(c*np.sqrt(2)*W(-g/A, -1)).real

print('D0 =', 2*np.exp(2+2*W(-g/A, -1)).real, '\n')
print('L2 =', L2,'\n')
print('L0 =', L0,'\n')
print('Ec20 = ', Ec20, '\n')

L = np.linspace(0, 2*L2, 50)

fE = open('Ecn_g01_L2.txt', 'w')
fD = open('D2n_g01_L2.txt', 'w')

for i in range(len(L)):
	Ec2, Xm2 = crit_Ecn(L[i], g)
	Dm2 = 1/Xm2**2	
	print('Ec =',Ec2/Ec20)
	print('Dc =',Dm2/D0)
	print(i, '\n')

	fE.write(str(L[i]) + ' ' + str(Ec2) + '\n')
	fD.write(str(L[i]) + ' ' + str(Dm2) + '\n')

fE.close()
fD.close()


''' FOR g = 0.1, 0 < L < L2 '''

g = 0.1

D0 = 2*np.exp(2+2*W(-g/A, -1)).real
L2 = (g/(3*W(-g/(3*np.sqrt(2)*np.exp(4/3)), -1)))**2
L2 = L2.real
L0 = g**2/16
Ec20 = -g/(c*np.sqrt(2)*W(-g/A, -1)).real

print('D0 =', 2*np.exp(2+2*W(-g/A, -1)).real, '\n')
print('L2 =', L2,'\n')
print('L0 =', L0,'\n')
print('Ec20 = ', Ec20, '\n')


L = np.linspace(0,L2,20)

fE = open('Ecp_g01_L2.txt', 'w')
fD = open('D2p_g01_L2.txt', 'w')

for i in range(len(L)):
	Ec2, Dm2 = crit_Ecp (L[i], g)
	print('Ec =',Ec2/Ec20)
	print('Dc =',Dm2/D0)
	print(i, '\n')

	fE.write(str(L[i]) + '  ' + str(Ec2) + '\n')
	fD.write(str(L[i]) + '  ' + str(Dm2) + '\n')

fE.close()
fD.close()

'''reading'''
fE = open('Ecp_g01_L2.txt')
fD = open('D2p_g01_L2.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
Lp = dataE[:, 0]/L2
Ecp = dataE[:, 1]/Ec20

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dp = dataD[:, 1]/D0

fE.close()
fD.close()


fE = open('Ecn_g01_L2.txt')
fD = open('D2n_g01_L2.txt')

dataE = fE.read()
dataE = dataE.split()
dataE = np.array(list(map(float, dataE)))
m = int(len(dataE)/2)
dataE = np.reshape(dataE, (m, 2))
Ln = -dataE[:, 0]/L2
Ln = np.flip(Ln, 0)
Ecn = dataE[:, 1]/Ec20
Ecn = np.flip(Ecn, 0)

dataD = fD.read()
dataD = dataD.split()
dataD = np.array(list(map(float, dataD)))
m = int(len(dataD)/2)
dataD = np.reshape(dataD, (m, 2))
Dn = dataD[:, 1]/D0
Dn = np.flip(Dn, 0)

fE.close()
fD.close()

L = np.concatenate((Ln, Lp))
Dc = np.concatenate((Dn, Dp))
Ec = np.concatenate((Ecn, Ecp))

plt.plot(L, Dc)
plt.axvline(x = 0, ls = '--')

plt.figure()
plt.plot(L, Ec)
plt.axvline(x = 0, ls = '--')

plt.show()

