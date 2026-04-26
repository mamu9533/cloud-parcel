import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyrcel as pm
import warnings
warnings.filterwarnings('ignore')


class Cloud_Parcel:
    """
    End-to-end wrapper for running a pyrcel adiabatic parcel model and
    computing cloud microphysical and optical properties.

    Chains together three components:
        1. pyrcel parcel model (run_pyrcel)
        2. Microphysics_Analysis for r_eff, LWC, and optical depth (add_cloud_microphysics)
        3. Monte Carlo radiative transfer model (compute_optical_properties)

    Parameters
    ----------
    T0 : float
        Initial parcel temperature, K.
    P0 : float
        Initial parcel pressure, Pa.
    S0 : float
        Initial parcel supersaturation, fraction (e.g. -0.02 for 2% subsaturated).
    H : float
        Maximum parcel ascent height, m.
    V : float
        Updraft velocity, m/s. Must be <= 3 (characteristic of low cloud).
    aerosols : list of pyrcel.AerosolSpecies
        Aerosol populations to activate within the parcel.
    parcel_trace : pd.DataFrame, optional
        Pre-computed parcel thermodynamic output. If None, populated by run_pyrcel().
    aerosol_traces : dict of pd.DataFrame, optional
        Pre-computed aerosol size output. If None, populated by run_pyrcel().
    max_cloud_thickness : float, optional
        Maximum cloud thickness in metres. Microphysical quantities above
        cloud_base + max_cloud_thickness are zeroed out. If None, the full
        parcel column is used (default 300).

    Attributes
    ----------
    parcel_trace : pd.DataFrame
        Parcel thermodynamic output with appended microphysics columns
        (reff, LWC, tau) after add_cloud_microphysics() is called.
    aerosol_traces : dict of pd.DataFrame
        Per-species droplet radius output from pyrcel.
    total_tau : float
        Column-integrated cloud optical depth.
    cloud_base_z : float
        Height of cloud base in metres, set when max_cloud_thickness is not None.
    cloud_top_z : float
        Height of cloud top in metres, set when max_cloud_thickness is not None.
    transmittance : float
        Fraction of photons transmitted through the cloud.
    reflectance : float
        Fraction of photons reflected by the cloud.
    absorbance : float or None
        Fraction of photons absorbed; None if with_absorbance=False.
    """

    def __init__(
        self,
        T0=None,
        P0=None,
        S0=None,
        H=None,
        V=None,
        aerosols=None,
        parcel_trace=None,
        aerosol_traces=None,
        max_cloud_thickness=300
    ):
        if V > 3:
            raise Exception("Selected updraft not characteristic of low cloud; select value < 3")

        self.T0 = T0
        self.P0 = P0
        self.S0 = S0
        self.H = H
        self.V = V
        self.aerosols = aerosols
        self.parcel_trace = parcel_trace
        self.aerosol_traces = aerosol_traces
        self.max_cloud_thickness = max_cloud_thickness

    def __repr__(self):
        species = [a.species for a in self.aerosols]
        return f"Cloud_Parcel(H={self.H}, V={self.V}, species={species})"

    def run_pyrcel(self, accom=0.3):
        """
        Run the pyrcel adiabatic parcel model.

        Integrates the parcel from the surface to height H at updraft
        velocity V, using a timestep of 1 second and the CVODE solver.

        Parameters
        ----------
        accom : float, optional
            Accommodation coefficient for condensation (default 0.3).

        Sets
        ----
        self.parcel_trace : pd.DataFrame
            Parcel thermodynamic state (T, P, S, z, etc.) at each timestep.
        self.aerosol_traces : dict of pd.DataFrame
            Per-species droplet radius at each timestep and size bin.
        """
        dt = 1.0 
        t_end = self.H / self.V
        model = pm.ParcelModel(self.aerosols, self.V, self.T0, self.S0, self.P0, console=False, accom=accom)
        self.parcel_trace, self.aerosol_traces = model.run(t_end, dt, solver='cvode')

    def add_cloud_microphysics(self):
        """
        Compute height-resolved microphysical properties and append to parcel_trace.

        Uses Microphysics_Analysis to compute effective radius, liquid water
        content, and optical depth at each model level, accounting for droplet
        activation state and optional cloud thickness truncation.

        If max_cloud_thickness is set, microphysical quantities above
        cloud_base + max_cloud_thickness are zeroed, and cloud_base_z /
        cloud_top_z are stored as attributes.

        Sets
        ----
        self.parcel_trace['reff'] : float
            Droplet effective radius, m.
        self.parcel_trace['LWC'] : float
            Liquid water volume concentration, m^3/m^3. NaN where zero.
        self.parcel_trace['tau'] : float
            Layer optical depth.
        self.total_tau : float
            Column-integrated optical depth.
        self.cloud_base_z : float
            Cloud base height, m (only set if max_cloud_thickness is not None).
        self.cloud_top_z : float
            Cloud top height, m (only set if max_cloud_thickness is not None).
        """
        from cloud_parcel.microphysics import Microphysics_Analysis

        microphys = Microphysics_Analysis(self.aerosols, self.parcel_trace, self.aerosol_traces, max_cloud_thickness=self.max_cloud_thickness)
        self.r_effs = microphys.height_resolved_r_effs()
        self.lwcs = microphys.height_resolved_LWCs()
        self.taus = microphys.height_resolved_taus()

        self.parcel_trace['reff'] = self.r_effs
        self.parcel_trace['LWC'] = np.where(self.lwcs == 0, np.nan, self.lwcs)
        self.parcel_trace['tau'] = self.taus

        self.total_tau = np.sum(self.parcel_trace['tau'])

        if self.max_cloud_thickness is not None:
            self.cloud_base_z = microphys.cloud_base_z
            self.cloud_top_z = microphys.cloud_top_z

    def compute_optical_properties(self, g=0.85, omega=0.99, N=100, with_absorbance=False):
        """
        Estimate cloud transmittance, reflectance, and optionally absorbance
        using a plane-parallel Monte Carlo radiative transfer model.

        Parameters
        ----------
        g : float, optional
            Henyey-Greenstein asymmetry parameter (default 0.85).
        omega : float, optional
            Single scattering albedo (default 0.99).
        N : int, optional
            Number of photons to simulate (default 100). Increase for
            lower variance at the cost of compute time.
        with_absorbance : bool, optional
            If True, enables photon absorption (default False).

        Sets
        ----
        self.transmittance : float
        self.reflectance : float
        self.absorbance : float or None
        """
        from cloud_parcel.monte_carlo import monte_carlo

        if self.total_tau is None:
            self.add_cloud_microphysics()

        model = monte_carlo(g=g, omega=omega, N=N, with_absorbance=with_absorbance)
        model.run(tau=self.total_tau, return_values=True)

        T = model.transmittance()
        R = model.reflectance()
        A = model.absorbance() if with_absorbance else None

        self.absorbance = A
        self.transmittance = T
        self.reflectance = R

    def summarize_and_visualize(self, **kwargs):
        """
        Produce a 2x2 summary figure of the parcel run.

        Runs add_cloud_microphysics() and compute_optical_properties() first
        if they have not already been called. Any keyword arguments are
        forwarded to compute_optical_properties() (e.g. N, g, omega,
        with_absorbance).

        The four panels show:
            - Top left:     Supersaturation and temperature profiles.
            - Top right:    Per-species droplet size distributions.
            - Bottom left:  Effective radius and liquid water content profiles.
            - Bottom right: Cumulative optical depth profile with a summary
                            text box of bulk optical properties.

        A gray shaded band marks the cloud layer in all panels. If
        max_cloud_thickness is set, the band spans cloud_base_z to
        cloud_top_z; otherwise it spans cloud base to the parcel top.

        Parameters
        ----------
        **kwargs
            Passed to compute_optical_properties().
        """
        if self.parcel_trace is None:
            raise Exception("Must run parcel model first. Use .run()")

        if self.total_tau is None:
            self.add_cloud_microphysics()

        if self.reflectance is None:
            self.compute_optical_properties(**kwargs)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row')
        ax_thermo = axes[0, 0]
        ax_aerosol = axes[0, 1]
        ax_microphys = axes[1, 0]
        ax_optical_depth = axes[1, 1]

        # --- Thermodynamics panel ---
        ax_thermo.plot(self.parcel_trace['S']*100., self.parcel_trace['z'], color='k', lw=2)
        axT = ax_thermo.twiny()
        axT.plot(self.parcel_trace['T'], self.parcel_trace['z'], color='r', lw=1.5)
        ax_thermo.set_ylim(0, self.H)

        axT.set_xticks(np.round(np.linspace(self.parcel_trace['T'].min(), self.T0, 5), 1))
        axT.xaxis.label.set_color('red')
        axT.tick_params(axis='x', colors='red')

        ax_thermo.set_xlabel("Supersaturation, %")
        axT.set_xlabel("Temperature, K")
        ax_thermo.set_ylabel("Height, m")

        # --- Aerosol size distribution panel ---
        colors = cm.tab10(np.linspace(0, 1, len(self.aerosols)))
        lines = []
        for i, aer in enumerate(self.aerosol_traces.keys()):
            aer_array = self.aerosol_traces[aer].values
            # Plot every 5th bin for visual clarity
            line_plot = ax_aerosol.plot(aer_array[:, ::5]*1e6, self.parcel_trace['z'], color=colors[i], label=str(aer))
            lines.append(line_plot[0])
        
        ax_aerosol.semilogx()
        ax_aerosol.legend([l for l in lines], [k for k in self.aerosol_traces.keys()], loc='upper right')
        ax_aerosol.set_xlabel("Droplet radius, micron")

        # --- Microphysics panel ---
        ax_microphys.plot(self.parcel_trace['reff']*10**6, self.parcel_trace['z'], color='b', lw=2)
        ax_LWC = ax_microphys.twiny()
        ax_LWC.plot(self.parcel_trace['LWC']*10**6, self.parcel_trace['z'], color='g', lw=1.5)

        ax_microphys.set_ylim([0, self.H])
        ax_LWC.set_ylim([0, self.H])

        ax_microphys.set_xlabel("Effective Radius, micron")
        ax_microphys.xaxis.label.set_color('blue')
        ax_microphys.set_ylabel("Height, m")
        ax_microphys.tick_params(axis='x', colors='blue')
        
        ax_LWC.set_xlabel("Liquid Water Volume ($cm^3$/$cm^3$)")
        ax_LWC.xaxis.label.set_color('green')
        ax_LWC.tick_params(axis='x', colors='green')

        # --- Optical depth panel ---
        ax_optical_depth.plot(np.cumsum(self.parcel_trace['tau']), self.parcel_trace['z'], color='purple', lw=2)
        ax_optical_depth.xaxis.label.set_color('purple')
        ax_optical_depth.set_xlabel("Integrated Optical Depth")
        ax_optical_depth.tick_params(axis='x', colors='purple')
        
        textstr = f"Total Optical Depth: {self.total_tau:.2f}\nReflectance: {self.reflectance:.2f}\nTransmittance: {self.transmittance:.2f}"
        if self.absorbance is not None:
            textstr += f"\nAbsorbance: {self.absorbance:.2f}"
            
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_optical_depth.text(0.2, 0.9, textstr, transform=ax_optical_depth.transAxes, fontsize=10,
                              verticalalignment='center', horizontalalignment='center', bbox=props)

        # --- Cloud layer shading ---
        # If max_cloud_thickness is set, shade between the computed cloud base
        # and top; otherwise shade from cloud base (first height with S > 0)
        # up to the parcel top.
        for ax in [ax_thermo, ax_aerosol, axT, ax_microphys, ax_LWC, ax_optical_depth]:
            ax.grid(False, 'both', 'both')
            if self.max_cloud_thickness is not None:
                ax.axhspan(self.cloud_base_z, self.cloud_top_z, color='gray', alpha=0.3)
            else:
                cloud_base_idx = np.argmax(self.parcel_trace['S'].values > 0)
                cloud_base_z = self.parcel_trace['z'].values[cloud_base_idx]
                ax.axhspan(cloud_base_z, self.parcel_trace['z'].values[-1], color='gray', alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ammonium_sulfate = pm.AerosolSpecies('Ammonium Sulfate',
                                pm.Lognorm(mu=0.015, sigma=1.6, N=850.),
                                kappa=0.8, bins=200)
    sea_salt = pm.AerosolSpecies('sea salt',
                                pm.Lognorm(mu=0.85, sigma=1.2, N=10.),
                                kappa=1.2, bins=40)

    cloud_test = Cloud_Parcel(T0=275, P0=77000, S0=-0.02, H=500, V=1, aerosols=[ammonium_sulfate, sea_salt], max_cloud_thickness=100)
    cloud_test.run_pyrcel()
    print("Model run completed\n"+"="*100)

    cloud_test.add_cloud_microphysics()
    print("Microphysics computed")
    print(f"Sample output: integrated optical depth is {cloud_test.total_tau}\n"+"="*100)

    cloud_test.compute_optical_properties(N=1000, with_absorbance=True)
    print("Monte Carlo model run completed")
    print(f"Transmittance: {cloud_test.transmittance}")
    print(f"Reflectance: {cloud_test.reflectance}")
    print(f"Absorbance: {cloud_test.absorbance}")

    assert cloud_test.total_tau > 0, "Optical depth should be positive"
    assert abs(cloud_test.transmittance + cloud_test.reflectance + cloud_test.absorbance - 1.0) < 1e-5, "T + R + A should equal 1"
    print("\nAll checks passed.")