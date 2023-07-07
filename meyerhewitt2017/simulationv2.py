import numpy as np
from scipy import integrate
from model import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from constants import *

#% % Simulation Parameters % % % % % % % % % % % % % %
T = 10 * 31536000
#% total simulation time (yr)
plot_amount = 1000
#% time between each plot
save_freq = 100
#% frequency at which plots are saved
phi0 = 0.64
#% surface porosity
AccumulationRate = 0.25 * (1 - phi0) * 31536000.0
#% accumulation rate
Qbar = -0.5
#% surface energy flux
solver_type = "empirical"

#% % Discretization % % % % % % % % % % % % % %
dx = .25
dt = 3600.0
#% % % % % % % % % % % % % % % % % % % % % % % % % % %

#% % Physical Parameters % % % % % % % % % % % % % % %
B = 260
#% bond number
Stefan = 12
#% stefan number
U = 100
#% darcy velocity to advection
A = 1
#% compaction pressure / viscosity scale
nglen = 1
#% Glens law exponent for viscosity
Pe = 11
#% Peclet number
ell = 20.6
#% firn melting lengthscale
#% % % % % % % % % % % % % % % % % % % % % % % % % % %


# read in climate data


#% Mesh Information
#% domain from from x=a to x=b
a = 0
b = 100
N = int((b - a) / dx)
#% Number of grid cells
xcelledges = np.linspace(a, b, N + 1)  #% Cell edges
xgrid = (xcelledges[0:N] + xcelledges[1 : (N + 1)]) / 2
#% Cell centers


#% Time step information
Nt = int(T / dt)
#% Number of timesteps


phiwithz = np.zeros((N, np.int(Nt / save_freq + 2)))
Swithz = np.zeros((N, np.int(Nt / save_freq + 2)))
Thetawithz = np.zeros((N, np.int(Nt / save_freq + 2)))
MeltRateSave = np.zeros((np.int(Nt / save_freq + 2)))
RunOffSave = np.zeros((np.int(Nt / save_freq + 2)))
time = np.zeros((np.int(Nt / save_freq) + 2))
SIV = np.zeros((np.int(Nt / save_freq + 2)))
BIV = np.zeros((np.int(Nt / save_freq + 2)))
BottomIceFlux = np.zeros((np.int(Nt / save_freq + 2)))
IceFlux = np.zeros((np.int(Nt / save_freq + 2)))
WaterFlux = np.zeros((np.int(Nt / save_freq + 2)))
BottomWaterFlux = np.zeros((np.int(Nt / save_freq + 2)))
TotalFlux = np.zeros((np.int(Nt / save_freq + 2)))


#% Functions
def k(p):
    return dp**2 / 180* p ** 3


def kr(s, β=β):
    return s ** β


def pc(s, α=α):
    return γ / dp * s ** (-α)


def kr_pc_prime(s, α=α, β=β):
    return -α * γ / dp * (s ** (β - (α + 1)))


def Abar(τ):
    return AccumulationRate


def EbarFun(τ, Qbar=Qbar):
    return Qbar - np.cos(2 * np.pi * τ)

def ustar(u,C,density):
    r"""
    calculate the friction velocity
    
    NOTE: likely overestimates ustar in the area covered case.
    """

    return u*np.sqrt((ρ_a/density)*C)


def sensible(T,T_a,u,C=C_o):
    r"""
    calculate the sensible heat flux
    
    """
    return ρ_a*c_a*C*u*(T-T_a)

def latent(T,P,q,u,C=C_o,L=L_v):
    r"""
    claculate the latent heat flux
    
    """
    return ρ_a*L*C*u*(qsat(T,P)-q)


def energy_balance_ice(temperature_surface,temperature_ambient,h_i,h_s,Q_sw,Q_lw,u,P,q):
    r"""
    calculate ice surface energy balance
    
    """

    Q_sens = sensible(temperature_surface,temperature_ambient,u,C_i)
    Q_lat = latent(temperature_surface,P,q,u,C_i,L_s)

    return Q_sens + Q_lat + ϵ_i*σ*temperature_surface**4 - (1-α_s)*Q_sw - ϵ_i*Q_lw  



# k = @(p) p.^(3); #% Simple Carmen-Kozeny
# kr = @(s) s.^β; #% relative permeability with saturation
# pc = @(s) s.^(-α); #% capillary pressure
# kr_pc_prime = @(s) -α.*(s.^(β-(α+1))); #% combined function
# Abar = @(τ)AccumulationRate; #% Accumulation
# EbarFun = @(τ)Qbar-np.cos(2*pi*τ);

#% Initial values
Rbar = 0
#% fixed surface water flux (rain)
pressure0 = -(pc(1.0) * γ / dp) * np.ones(N)
#% initial pressure
zs = 0
#% zero initial surface height
W = (1 - phi0) * np.ones(N) # initially saturation is zero.
#% take in W from above


H = W * energy_balance(Tm, Ta, h_i, h_s, Q_sw, Q_lw, u, P, q)
#% take in H from above

#% Initialize variables
Tsurf = np.zeros(Nt)
MeltRate = np.zeros(Nt)
n_plot = 1
RunOff = np.zeros(Nt)
RO = np.zeros(Nt)

LiquidWater = np.zeros(Nt)
TotalIce = np.zeros(Nt)
for n in range(Nt):  # range(Nt):
    #% assign values from previous timestep
    print("timestamp is: " + str(n * dt))
    W_nm1 = W
    H_nm1 = H
    # print(H)
    #% convert to Theta, phi, and S
    Theta_nm1, phi_nm1, S_nm1 = conversiontotemperature(H_nm1, W_nm1, Stefan)
    #% % Theta is temperature % phi is porosity % S is saturation % %
    # print(S_nm1)
    #% Compute Compaction Velocity
    # print(phi_nm1)
    # print(W_nm1)
    # print(Theta_nm1)
    Shear = compactionfunction(
        xgrid, W_nm1, phi_nm1, Theta_nm1, A, nglen, Abar(n * dt), solver_type
    )
    CompactionVelocity = integrate.cumtrapz(
        np.append(Shear, Shear[-1]), xcelledges, initial=0
    )
    SurfaceIceVelocity = (Abar(n * dt) / (1 - phi0)) - MeltRate[np.maximum(n - 1, 0)]
    if SurfaceIceVelocity < 0:
        SurfaceIceVelocity = (Abar(n * dt) / (1 - phi_nm1[0])) - MeltRate[
            np.maximum(n - 1, 0)
        ]

    IceVelocity = SurfaceIceVelocity * np.ones(N + 1) - CompactionVelocity
    #% % Total Water % %
    #% Ice Advective flux;
    FadvIpW, FadvImW = advectiveflux(N, IceVelocity, W_nm1)

    #% Saturation Advective flux;
    SaturationVelocity = np.ones(N + 1)
    fadvSp, fadvSm = advectiveflux(N, SaturationVelocity, k(phi_nm1) * kr(S_nm1))
    FadvSp = ρ * g / μ * fadvSp
    FadvSm = ρ * g / μ * fadvSm

    #% Diffusive flux
    FpD, FmD = saturationdiffusiveflux(phi_nm1, S_nm1, dt, dx, U / B)

    #% No diffusive flux boundary condition
    FpD[-1] = 0

    #% Total water saturation flux
    FpWS = FadvSp + FpD
    FmWS = FadvSm + FmD

    #% Total fluxes
    FpW = FadvIpW + FpWS
    FmW = FadvImW + FmWS
    FmW[0] = Abar(n * dt) + Rbar
    # % accumulation and rain :: - RO[np.max(n-1,1)]
    Fdif = FpW - FmW
    W = W_nm1 - (dt / dx) * Fdif

    #% % Enthalpy % %
    #% Ice Advective flux;
    FadvIpH, FadvImH = advectiveflux(N, IceVelocity, H_nm1)

    #% Saturation Advective flux;
    SaturationVelocity = np.ones(N + 1)
    fadvSp, fadvSm = advectiveflux(N, SaturationVelocity, k(phi_nm1) * kr(S_nm1))
    FadvSp = ρ * g / μ * ρ * L * fadvSp
    FadvSm = ρ * g / μ * ρ * L * fadvSm
    # print(phi_nm1)
    # print(S_nm1)
    # print(k(phi_nm1)*kr(S_nm1))
    # print(FadvSp)
    # print(FadvSm)
    #% Enthalpy/Saturation Diffusive flux
    FpE, FmE = enthalpydiffusiveflux(Theta_nm1, phi_nm1, S_nm1, dt, dx, U / B, ρ * L)

    #% Temperature Diffusive flux
    FpT, FmT = temperaturediffusiveflux((1.0-phi_nm1), Theta_nm1, dt, dx, K)

    #% No diffusive flux boundary condition
    FpT[-1] = 0
    FpE[-1] = 0

    #% Total saturation flux
    FpS = FpE + FadvSp
    FmS = FmE + FadvSm

    #% Total Saturation and Temperature fluxes
    Fp = FadvIpH + FpS + FpT
    Fm = FadvImH + FmS + FmT
    # print(EbarFun(n*dt))
    Fm[0] = Q - h*Theta_nm1 + ρ*L*R - ρ*L*r#Stefan * (EbarFun(n * dt) - Theta_nm1[0]) + Stefan * Rbar
    # % Enthalpy neumann conditions
    Fdif = Fp - Fm
    H = H_nm1 - (dt / dx) * Fdif

    #% Compute fully saturated water pressure
    ind = S_nm1 >= 1
    # print(ind)
    if ind[ind].any():
        print("We are fully_saturated..")
        qp, qm, pressure = fullysaturatedwaterpressure(
            U,
            pressure0,
            dx,
            xgrid,
            N,
            A,
            phi_nm1,
            W_nm1,
            Theta_nm1,
            nglen,
            ind,
            Abar(n * dt),
            solver_type,
        )
        slocations = np.argwhere(ind[1 : (N - 1)]) + 1
        for i in slocations:
            if (S_nm1[i - 1] != 1) and (S_nm1[i] == 1):  #% unsat(left)/sat(right):
                #% Total Water
                FpWS[i - 1] = np.minimum(qp[i - 1], FpWS[i - 1])
                #% use minimum
                FmWS[i] = np.minimum(qp[i - 1], FpWS[i - 1])
                #% use minimum
                #% Enthalpy
                FpS[i - 1] = np.minimum(Stefan * qp[i - 1], FpS[i - 1])
                #% use minimum
                FmS[i] = np.minimum(Stefan * qp[i - 1], FpS[i - 1])
                #% use minimum
            elif (S_nm1[i] == 1) and (S_nm1[i + 1] == 1):  #% sat[(eft)/sat[right]:
                #% Total Water
                FpWS[i] = qp[i]
                #% use q
                FmWS[i + 1] = qp[i]
                #% use q
                #% Enthalpy
                FpS[i] = Stefan * qp[i]
                #% use q
                FmS[i + 1] = Stefan * qp[i]
                #% use q
            elif (S_nm1[i] == 1) and (S_nm1[i + 1] != 1):  #% sat[left]/unsat[right]:
                #% Total Water
                FpWS[i] = np.maximum(qp[i], FpWS[i])
                #% use maximum
                FmWS[i + 1] = np.maximum(qp[i], FpWS[i])
                #% use maximum
                #% Enthalpy
                FpS[i] = np.maximum(Stefan * qp[i], FpS[i])
                #% use maximum
                FmS[i + 1] = np.maximum(Stefan * qp[i], FpS[i])
                #% use maximum

        # Fix flux at surface for full saturation
        if ind[0]:
            FmWS[0] = qm[0]
            FmS[0] = Stefan * qm[0]
        elif ind[-1]:
            FpWS[-1] = qp[-1]
            FpS[-1] = Stefan * qp[-1]

        #% Total Water
        if ind[0]:
            FpW = FadvIpW + FpWS
            FmW = FadvImW + FmWS
        else:
            FpW = FadvIpW + FpWS
            # Don't doctor FmW(1) if first node not saturated
            # to keep boundary condition imposed earlier
            FmW[1:] = FadvImW[1:] + FmWS[1:]

        #% Compute run off
        RO[n] = Abar(n * dt) + Rbar - FmW[0]
        if RO[n] > 0:
            RunOff[n] = RO[n]
            RunOff_flag = 1
            #% FmW(1) = FpW(1);
        else:
            FmW[0] = Abar(n * dt) + Rbar
            #% Fixed rain flux
            RunOff_flag = 0
        Fdif = FpW - FmW
        W = W_nm1 - (dt / dx) * Fdif
        if RunOff_flag:
            W[0] = 1

        #% Enthalpy
        Fp = FadvIpH + FpS + FpT
        Fm = FadvImH + FmS + FmT
        if ~RunOff_flag:
            Fm[0] = Stefan * (EbarFun(n * dt) - Theta_nm1[0] + Rbar)
        else:
            Fm[0] = Stefan * (EbarFun(n * dt) - Theta_nm1[0] + Rbar - RunOff[n])
        Fdif = Fp - Fm
        H = H_nm1 - (dt / dx) * Fdif
    else:
        pressure = pressure0
    Theta, phi, S = conversiontotemperature(H, W, Stefan)
    H, W = conversiontoenthalpy(Theta, phi, S, Stefan)

    #% Compute surface melt rate
    Tsurf[n] = Theta[0]
    DiffusiveFlux = (
        -(2 / (Pe * dx)) * (((1 / W[0]) + (1.0 / W[1])) ** (-1)) * (Theta[1] - Theta[0])
    )
    MR = (EbarFun(n * dt) - (DiffusiveFlux / Stefan)) / (1 - phi[0])
    ThetaSurface = (3 / 2) * Theta[0] - (1 / 2) * Theta[1]
    #%     if and(Theta[0]==0,MR>0)
    if (ThetaSurface >= 0) and (MR > 0):
        MeltRate[n] = MR
    else:
        MeltRate[n] = 0
    """
    if ~mod(n,plot_amount)
        plot(H,xgrid,'k','linewidth',2)
        hold on;
        plot(Theta,xgrid,'y','linewidth',2)
        plot(S,xgrid,'r','linewidth',2)
        plot(W,xgrid,'b','linewidth',2)
        plot(phi,xgrid,'g','linewidth',2)
        plot(pressure,xgrid,'m','linewidth',2)
        plot(FmWS,xgrid,'c','linewidth',2)
        title(num2str(n*dt))
        set(gca,'fontsize',18,'ydir','reverse')
        axis([-1 2 a b])
        drawnow;
        hold off;
    """
    zs = zs + SurfaceIceVelocity * dt

    if np.mod(n, save_freq) == 0:
        phiwithz[:, n_plot] = phi
        Swithz[:, n_plot] = S
        Thetawithz[:, n_plot] = Theta
        MeltRateSave[n_plot] = MeltRate[n]
        RunOffSave[n_plot] = RunOff[n]
        time[n_plot] = n * dt
        SIV[n_plot] = IceVelocity[0]
        #% surface ice velocity
        BIV[n_plot] = IceVelocity[-1]
        #% bottom ice velocity
        BottomIceFlux[n_plot] = FadvImW[-1]
        IceFlux[n_plot] = FadvImW[0]
        WaterFlux[n_plot] = FmWS[0]
        BottomWaterFlux[n_plot] = FpWS[-1]
        TotalFlux[n_plot] = FmW[0]
        n_plot = n_plot + 1

    LiquidWater[n] = np.trapz(phi * S, xgrid)
    TotalIce[n] = np.trapz(1 - phi, xgrid)

    if np.mod(n, plot_amount) == 0:
        fig, ax = plt.subplots()
        ax.plot(H, xgrid, "k", linewidth=2)
        ax.plot(Theta, xgrid, "y", linewidth=2)
        ax.plot(S, xgrid, "r", linewidth=2)
        ax.plot(W, xgrid, "b", linewidth=2)
        ax.plot(phi, xgrid, "g", linewidth=2)
        ax.plot(pressure, xgrid, "m", linewidth=2)
        ax.plot(FmWS, xgrid, "c", linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_xlim(-1, 2)
        plt.gca().invert_yaxis()
        fig.savefig(
            "/Users/andrew/projects/thwaites_melt_event/figures/fig01_t{}.png".format(
                n * dt
            )
        )

    zs = zs + SurfaceIceVelocity * dt


#% Save solution
tvec = time
MR = MeltRateSave
RO = RunOffSave
phimat = phiwithz
smat = Swithz
thetamat = Thetawithz
drainage = -BIV * phimat[-1, :] * smat[-1, :] + BottomWaterFlux


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8), dpi=200)
im1 = ax1.imshow(phimat, extent=[0, T, 0, ell], aspect="auto")
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax, orientation="vertical")


im2 = ax2.imshow(smat, extent=[0, T, 0, ell], aspect="auto")
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax, orientation="vertical")

deltaT = 200 / 14.8
im3 = ax3.imshow(deltaT * thetamat, extent=[0, T, 0, ell], aspect="auto")
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im3, cax=cax, orientation="vertical")

ax1.get_shared_x_axes().join(ax2, ax3)


fig.savefig("/Users/andrew/projects/thwaites_melt_event/figures/fig02.png")

"""
% Save solution
tvec = time;
MR = MeltRateSave;
RO = RunOffSave;
phimat = phiwithz;
smat = Swithz;
thetamat = Thetawithz;
drainage = -BIV.*phimat(end,:).*smat(end,:) + BottomWaterFlux;

tmat = repmat(time,length(xgrid),1);
zmat = ell*repmat(xgrid,1,length(time));

figure(2)
subplot(3,1,1)
surf(tmat,zmat,phimat,'EdgeColor','none'); view(2);
hc = colorbar; set(hc,'ylim',[0 phi0])
ylabel(hc,'$\phi$','interpreter','latex','fontsize',20)
set(gca,'fontsize',18,'ydir','reverse','layer','top'); grid off;
ylabel('$Z$ (m)','interpreter','latex','fontsize',20)
axis([0 T 0 ell])

subplot(3,1,2)
surf(tmat,zmat,smat,'EdgeColor','none'); view(2);
hc = colorbar; set(hc,'ylim',[0 1])
ylabel(hc,'$S$','interpreter','latex','fontsize',20)
set(gca,'fontsize',18,'ydir','reverse','layer','top'); grid off;
ylabel('$Z$ (m)','interpreter','latex','fontsize',20)
axis([0 T 0 ell])

subplot(3,1,3)
deltaT = 200/14.8;
surf(tmat,zmat,deltaT*thetamat,'EdgeColor','none'); view(2);
hc = colorbar; set(hc,'ylim',[-20,0])
ylabel(hc,'$T$ $^{\circ}$C','interpreter','latex','fontsize',20)
set(gca,'fontsize',18,'ydir','reverse','layer','top'); grid off;
xlabel('$t$ (yr)','interpreter','latex','fontsize',20)
ylabel('$Z$ (m)','interpreter','latex','fontsize',20)
axis([0 T 0 ell])
Footer
"""
