import numpy as np
from scipy import sparse
from scipy import integrate
from itertools import combinations


alpha = 1; #% exponent of capillary pressure
beta = 2; #% saturation flux exponent

#% Functions
def k(p):
    return p**3
def kr(s,beta=beta):
    return s**beta
def pc(s,alpha=alpha):
    return s**-alpha
def kr_pc_prime(s,alpha=alpha,beta=beta):
    return - alpha*(s**(beta-(alpha+1)))
def Abar(tau):
    return AccumulationRate
def EbarFun(tau):
    Qbar-np.cos(2*np.pi*tau)


def solve_matrix(A,b):
    num_vars = A.shape[1]
    rank = np.linalg.matrix_rank(A)
    print(rank)
    if rank == num_vars:              
        sol = np.linalg.lstsq(A, b)[0]    # not under-determined
    else:
        for nz in combinations(range(num_vars), rank):    # the variables not set to zero
            try: 
                sol = np.zeros((num_vars, 1))  
                sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
                print(sol)
            except np.linalg.LinAlgError:     
                pass     
    return sol


def sparse_compress(i, j, v, m, n):
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values 
            Size n1
        j: 1-D array representing the index 2 values 
            Size n1
        v: 1-D array representing the values 
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return sparse.csr_matrix((v, (i, j)), shape=(m, n)).toarray()



def conversiontotemperature(H,W,Stefan):
	#% This function takes in the enthalpy and total water/ice mass and spits
	#% out the temperature, porosity, and saturation
	#% [T,phi,S] = conversiontotemperature(H,W,Stefan)
	T = np.minimum(0.0,(H/W));
	phi = np.maximum(1-W+(1/Stefan)*np.maximum(0.0,H),0.0);
	#% Sc = max(0,(H./(Stefan*phi)));
	#% S = max(0,H)./(max(0,H)+Stefan*(1-W));
	S = np.maximum(np.minimum(1,np.maximum(0,H)/(np.maximum(0,H)+Stefan*(1-W))),0);
	#S = np.maximum(np.minimum(1,np.maximum(0.0,H))/(np.maximum(0,H))+Stefan*(1-W),0.0);
	return T,phi,S

def conversiontoenthalpy(T,phi,S,Stefan):
	#% This function takes in the enthalpy and total water/ice mass and spits
	#% out the temperature, porosity, and saturation
	#% [T,phi,S] = conversiontotemperature(H,W,Stefan)
	W = 1-phi+phi*S;
	H = W*T+Stefan*phi*S;
	return H,W

def temperaturediffusiveflux(TotalWater,Temperature,dt,dx,Constants):
	#% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
	#% [Fp,Fm] = TemperatureDiffusiveFlux(TotalWater,Temperature,dt,dx,Constants)
	#% Diffusive flux with input arguments: TotalWater which is W and 
	#% temperature, which comes from the function
	#% [T,phi,S]=conversiontotemperature(H,W,Stefan).
	#% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

	Temperaturem = np.append(Temperature[0],Temperature[0:-1])
	Temperaturep = np.append(Temperature[1:], Temperature[-1])

	TotalWaterm = np.append(TotalWater[0], TotalWater[0:-1])
	TotalWaterp = np.append(TotalWater[1:], TotalWater[-1])

	Dm = TotalWaterm;
	D = TotalWater;
	Dp = TotalWaterp;

	fim = (2/dx)*(((1./Dm)+(1./D))**(-1));
	fip = (2/dx)*(((1./D)+(1./Dp))**(-1));
	Fm = -Constants*fim*(Temperature-Temperaturem);
	Fp = -Constants*fip*(Temperaturep-Temperature);
	return Fp,Fm 

def saturationdiffusiveflux(Porosity,Saturation,dt,dx,Constants):
	#% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
	#% [Fp,Fm] = SaturationDiffusiveFlux(Diffusivity,Porosity,Saturation,dt,dx,Constants)
	#% Diffusive flux with input arguments: Diffusivity as a function handle of
	#% phi and S (two input arguments) and Porosity and
	#% Saturation, which come from the function
	#% [T,phi,S]=conversiontotemperature(H,W,Stefan).
	#% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

	Porositym = np.append(Porosity[0],Porosity[0:-1]);
	Porosityp = np.append(Porosity[1:], Porosity[-1]);

	#print(len(Porosity))
	#print(len(Porositym))
	#print(len(Porosityp))
	Saturationm = np.append(Saturation[0], Saturation[0:-1]);
	Saturationp = np.append(Saturation[1:], Saturation[-1]);

	Dm = k(Porositym)*kr_pc_prime(Saturationm);
	D = k(Porosity)*kr_pc_prime(Saturation);
	Dp = k(Porosityp)*kr_pc_prime(Saturationp);

	fim = (2/dx)*(((1./Dm)+(1./D))**(-1));
	fip = (2/dx)*(((1./D)+(1./Dp))**(-1));
	Fm = Constants*fim*(Saturation-Saturationm);
	Fp = Constants*fip*(Saturationp-Saturation);

	return Fp,Fm

def fullysaturatedwaterpressure(U,pressure,dx,xgrid,N,A,phi,W,Theta,nglen,Indicator,Accumulation,solver_type):
	# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
	# Compute water pressure when the snowpack is fully saturated. I is an
	# indicator function that determines whether to solve for the water pressure
	# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

	phimin = 10**(-4);
	Indicator[phi<phimin]=0;

	phim = np.append(phi[0], phi[0:-1]); #% cell values to the left of cell i
	phip = np.append(phi[1:], phi[-1]); #% cell values to the right of cell i

	Dm = U*k(phim);
	D = U*k(phi);
	Dp = U*k(phip);

	fim = (2/dx)*(((1/Dm)+(1/D))**(-1));
	fip = (2/dx)*(((1/D)+(1/Dp))**(-1));
	fip[N-1]=0; #% no pressure gradient
	# full difference
	S = np.append(np.append(fim[1:].conj().T, -(fim+fip).conj().T),fip[0:-1].conj().T);
	Ind = np.append(np.append(np.arange(1,N), np.arange(0,N)), np.arange(0,(N-1)));
	Jnd = np.append(np.append(np.arange(0,(N-1)), np.arange(0,N)), np.arange(1,N));

	#print(len(N))
	M = sparse_compress(Ind,Jnd,S,N,N);
	# positive flux
	S = np.append(np.append(np.zeros(N-1),-fip.conj().T),fip[0:-1].conj().T);
	Ind = np.append(np.append(np.arange(1,N), np.arange(0,N)), np.arange(0,(N-1)));
	Jnd = np.append(np.append(np.arange(0,(N-1)), np.arange(0,N)), np.arange(1,N));


	Mp = sparse_compress(Ind,Jnd,S,N,N);
	# negative flux
	S = np.append(np.append(-fim[1:].conj().T,fim.conj().T),np.zeros(N-1));
	Ind = np.append(np.append(np.arange(1,N), np.arange(0,N)), np.arange(0,(N-1)));
	Jnd = np.append(np.append(np.arange(0,(N-1)), np.arange(0,N)), np.arange(1,N));

	Mm = sparse_compress(Ind,Jnd,S,N,N);

	# compaction right hand side
	Shear = compactionfunction(xgrid,W,phi,Theta,A,nglen,Accumulation,solver_type);

	# saturation advection components
	SaturationVelocity = np.ones(N+1);
	[fadvSp,fadvSm] = advectiveflux(N,SaturationVelocity,U*k(phi));
	fdifS = fadvSp-fadvSm;

	# water pressure computation


	print((dx*Shear[Indicator]-fdifS[Indicator]+np.dot(M[Indicator][:,~Indicator],pressure[~Indicator])))
	indicatorX,indicatorY=np.meshgrid(Indicator,Indicator)
	indicatorXtilda,indicatorYtilda=np.meshgrid(~Indicator,~Indicator)
	print(np.array(np.expand_dims(-M[Indicator,Indicator],axis=0)))

	#print(solve_matrix(np.expand_dims(-M[Indicator,Indicator],axis=0), dx*Shear[Indicator]-fdifS[Indicator]+np.dot(M[Indicator][:,~Indicator],pressure[~Indicator])).shape)
	#pressure[Indicator]= np.linalg.solve(((-M[Indicator,Indicator]).T.dot(-M[Indicator,Indicator])), (-M[Indicator,Indicator]).T.dot(dx*Shear[Indicator]-fdifS[Indicator]+M[Indicator,~Indicator]*pressure[~Indicator]));
	pressure[Indicator]= solve_matrix(-M[indicatorX*indicatorY].reshape(Indicator.sum(),Indicator.sum()), dx*Shear[Indicator]-fdifS[Indicator]+np.dot(M[indicatorY*indicatorXtilda].reshape(Indicator.sum(),~Indicator.sum()),pressure[~Indicator]))[0];
	#if len(pressure[Indicator])==1:
#		pressure[Indicator]= solve_matrix(np.expand_dims(-M[Indicator,Indicator],axis=0), (dx*Shear[Indicator]-fdifS[Indicator]+np.dot(M[Indicator,~Indicator],pressure[~Indicator])));
#	else:
#		print(M.dtype)
#		print(M.shape)
#		print(Indicator.dtype)
#		print(Indicator.shape)
#		print((~Indicator).shape)
#		print(-M[Indicator,Indicator])
#		print(dx*np.array(Shear[Indicator]))
#		print(M[Indicator][:,~Indicator].shape)
#		print(pressure[(~Indicator)].shape)
#		print(np.dot(M[Indicator][:,~Indicator],pressure[~Indicator]).shape)
#		pressure[Indicator]= solve_matrix(-M[Indicator,Indicator], dx*Shear[Indicator]-fdifS[Indicator]+np.dot(M[Indicator][:,~Indicator],pressure[~Indicator]));
#	print(pressure)

	fpD = -Mp*pressure; fmD = -Mm*pressure;
	qp = fpD+fadvSp;
	qm = fmD+fadvSm;
	return qp[0],qm[0],pressure

def enthalpydiffusiveflux(Temperature,Porosity,Saturation,dt,dx,Constants,Stefan):
	
	#[Fp,Fm] = EnthalpyDiffusiveFlux(Diffusivity,Temperature,Porosity,Saturation,dt,dx,Constants,Stefan)
	#Diffusive flux with input arguments: Diffusivity as a function handle of
	#phi and S (two input arguments) and temperature, porosity, and
	#Saturation, come from the function
	# [T,phi,S]=conversiontotemperature(H,W,Stefan).
	# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
	# Here the diffusivity includes the (T+Stefan) term.
	# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

	#% Temperaturem = [Temperature(end); Temperature(1:(end-1))];
	#% Temperaturep = [Temperature(2:end); Temperature(1)];
	#% Porositym = [Porosity(end); Porosity(1:(end-1))];
	#% Porosityp = [Porosity(2:end); Porosity(1)];
	#% Saturationm = [Saturation(end); Saturation(1:(end-1))];
	#% Saturationp = [Saturation(2:end); Saturation(1)];

	Temperaturem = np.append(Temperature[0], Temperature[0:-1]);
	Temperaturep = np.append(Temperature[1:], Temperature[-1])

	Porositym = np.append(Porosity[0], Porosity[0:-1]);
	Porosityp = np.append(Porosity[1:], Porosity[-1]);

	Saturationm = np.append(Saturation[0], Saturation[0:-1]);
	Saturationp = np.append(Saturation[1:], Saturation[-1]);

	Dm = k(Porositym)*kr_pc_prime(Saturationm)*(Temperaturem+Stefan);
	D = k(Porosity)*kr_pc_prime(Saturation)*(Temperature+Stefan);
	Dp = k(Porosityp)*kr_pc_prime(Saturationp)*(Temperaturep+Stefan);

	fim = (2/dx)*(((1./Dm)+(1./D))**(-1));
	fip = (2/dx)*(((1./D)+(1./Dp))**(-1));
	Fm = Constants*fim*(Saturation-Saturationm);
	Fp = Constants*fip*(Saturationp-Saturation);
	return Fp,Fm

def compactionfunction(xgrid,W,phi,Theta,A,nglen,Accumulation,solver_type):
# This function takes in W, phi, and theta and outputs the compaction rate
# xcelledges are the cell edge values
# phi is the snow porosity
# W is the total water, i.e. sum of ice and liquid water
# Theta is the snow temperature
# A is the snow viscosity softness
# nglen is the snow viscosity exponent
    if solver_type=='none':
        Shear = np.zeros(len(phi)); # dw/dz
        
    elif solver_type=='poreclosure':
        IcePressure = integrate.cumtrapz(W,xgrid);
        Shear = 2.*A*(phi/(1-phi))*IcePressure**nglen; # dw/dz
        
    else:
        rhoi = 917; # ice density
        rhow = 1000; # water density
        f = 1-(550/rhoi); # transition porosity
        R = 8.314; # gas constant
        E0 = 10160; # activation energy
        E1 = 21400; # activation energy
        T = 273.15+(200/14.8)*Theta; # temperature in kelvin
        omega = 1/(3600*24*365); # 1/(seconds in a year)
        ell = 200/(omega*917*334000); # lengthscale
        k0 = 11; # H&L1980 constant
        k1 = 575; # H&L1980 constant
        A0 = (rhoi/rhow)*Accumulation*ell; # water equivalent accumulation per year % % ACCUMULATION TO MASS RATE
        A1 = (rhoi/rhow)*Accumulation*ell; # water equivalent accumulation per year % % ACCUMULATION TO MASS RATE
        a = 1; # H&L1980 constant
        b = 1/2; # H&L1980 constant
        c0 = (k0*(A0**a))*np.exp(-E0/(R*T));
        c1 = (k1*(A1**b))*np.exp(-E1/(R*T));
        C = c0*(phi>=f)+c1*(phi<f);
        Shear = C*(phi/(1-phi)); # dw/dz
    return Shear

def advectiveflux(N,velocity,field):
    #% velocity is (N+1)x1
    #% field is Nx1
    fadv = np.zeros(N+1);
    for i in range(0,N+1):
        if (velocity[i]>0) and (i==0):
            #% fadv(i) = 0; # will be fixed by the boundary condition
            fadv[i] = velocity[i]*field[i]; # will usually be fixed by the boundary condition
        elif (velocity[i]>0) and (i!=0):
        	fadv[i] = velocity[i]*field[i-1];
        elif (velocity[i]<=0) and (i==(N)):
            #% fadv[i] = 0; # will be fixed by the boundary condition
            fadv[i] = velocity[i]*field[i-1]; # natural inflow
        elif (velocity[i]<=0) and (i!=(N)): # i.e velocity(i)<=0 and not at the end point
            fadv[i] = velocity[i]*field[i];
    fadvp = fadv[1:(N+1)];
    fadvm = fadv[0:N];
    return fadvp,fadvm




