import numpy as np
import pyrcel as pm

from utils import compute_effective_radius, compute_volume, compute_tau


class Microphysics_Analysis:
    """
    Post-processing class for pyrcel parcel model output.

    Computes height-resolved microphysical quantities including effective
    radius and liquid water content, accounting for droplet activation state.

    Parameters
    ----------
    initial_aerosol : list of AerosolSpecies
        Initial aerosol population passed to the parcel model.
    parcel_trace : pd.DataFrame
        Parcel thermodynamic output from pyrcel.
    aerosol_traces : dict of pd.DataFrame
        Aerosol radii output from pyrcel, keyed by species name.
    max_cloud_thickness : float, optional
        Maximum cloud thickness in metres. Microphysical quantities above
        cloud_base + max_cloud_thickness are zeroed. If None, the full
        parcel column is used (default 300).

    Attributes
    ----------
    activation_matrices : dict
        Per-species binary activation matrices (n_heights x n_bins).
        Set after calling process_that_shiiii().
    cloud_base_idx : int
        Index of cloud base in parcel_trace. Set by _apply_cloud_thickness_truncation().
    cloud_base_z : float
        Height of cloud base, m. Set by _apply_cloud_thickness_truncation().
    cloud_top_idx : int
        Index of cloud top in parcel_trace. Set by _apply_cloud_thickness_truncation().
    cloud_top_z : float
        Height of cloud top, m. Set by _apply_cloud_thickness_truncation().
    reffs : np.ndarray
        Height-resolved effective radii, m. Set after calling height_resolved_r_effs().
    lwcs : np.ndarray
        Height-resolved liquid water content, m^3/m^3. Set after calling height_resolved_LWCs().
    taus : np.ndarray
        Height-resolved layer optical depths. Set after calling height_resolved_taus().
    """

    def __init__(self, initial_aerosol, parcel_trace, aerosol_traces, max_cloud_thickness=300):

        self.initial_aerosol = initial_aerosol
        self.parcel_trace = parcel_trace
        self.aerosol_traces = aerosol_traces
        self.max_cloud_thickness = max_cloud_thickness
        self.activation_matrices = None
        self.reffs = None
        self.lwcs = None
        self.taus = None
        self.masked_arrays = None

    def process_that_shiiii(self):
        """
        Construct per-species activation matrices from parcel output.

        For each aerosol species and height, determines whether each size
        bin has activated based on the Kohler critical supersaturation
        compared to the parcel supersaturation. Once the parcel reaches
        maximum supersaturation, activation state is held constant for
        all subsequent heights.

        Sets
        ----
        self.activation_matrices : dict
            Binary matrices (n_heights x n_bins) per species, where 1
            indicates an activated droplet and 0 indicates unactivated.
        """
        parcel_trace = self.parcel_trace
        initial_aerosol = self.initial_aerosol

        activation_matrices = {aer.species: None for aer in initial_aerosol}

        for aer in initial_aerosol:
            r_drys = aer.r_drys
            super_mask = np.zeros([len(parcel_trace['z']), len(r_drys)])
            smax = parcel_trace['S'].max()
            s = parcel_trace['S'][0]
            j = 0
            for j, (_, row) in enumerate(parcel_trace.iterrows()):
                s = row['S']
                activation_row = np.zeros(len(r_drys)).reshape(1, -1)
                for i, r in enumerate(r_drys):
                    # A bin is activated if the parcel supersaturation exceeds
                    # the Kohler critical supersaturation for that dry radius
                    _, scrit = pm.thermo.kohler_crit(row['T'], r, aer.kappa)
                    if scrit >= row['S']:
                        activation_row[:, i] = 0
                    else:
                        activation_row[:, i] = 1
                    super_mask[j] = activation_row
                if s == smax:
                    # Freeze activation state at peak supersaturation and
                    # propagate it to all remaining heights
                    super_mask[j:,] = activation_row
                    break

            activation_matrices[aer.species] = super_mask

        self.activation_matrices = activation_matrices

    def _apply_cloud_thickness_truncation(self, radii, Nis):
        """
        Zero out rows in radii and Nis arrays that fall above the cloud top.

        Cloud base is the first height where supersaturation exceeds zero.
        Cloud top is the first height where z exceeds cloud base by more than
        max_cloud_thickness metres. If the parcel never reaches that height,
        cloud_top_idx is set to len(z) and nothing is zeroed.

        Parameters
        ----------
        radii : np.ndarray
            Masked radii array (n_heights x n_bins), m.
        Nis : np.ndarray
            Masked number concentration array (n_heights x n_bins), m^-3.

        Returns
        -------
        radii : np.ndarray
            Truncated radii array.
        Nis : np.ndarray
            Truncated number concentration array.

        Sets
        ----
        self.cloud_base_idx, self.cloud_base_z
        self.cloud_top_idx, self.cloud_top_z
        """
        z = self.parcel_trace['z'].values
        S = self.parcel_trace['S'].values

        self.cloud_base_idx = int(np.argmax(S > 0))
        cloud_base_z = z[self.cloud_base_idx]
        self.cloud_base_z = cloud_base_z

        # If the parcel top is below cloud_base_z + max_cloud_thickness,
        # cloud_sel will be empty and cloud_top_idx defaults to len(z),
        # meaning nothing gets zeroed out.
        cloud_sel = np.where(z >= cloud_base_z + self.max_cloud_thickness)[0]
        self.cloud_top_idx = int(cloud_sel[0]) if len(cloud_sel) > 0 else len(z)
        self.cloud_top_z = self.parcel_trace['z'].iloc[self.cloud_top_idx - 1]

        radii = radii.copy()
        Nis = Nis.copy()
        radii[self.cloud_top_idx:] = 0.0
        Nis[self.cloud_top_idx:] = 0.0

        return radii, Nis

    def _get_masked_arrays(self):
        """
        Build concatenated masked radii and number concentration arrays.

        Calls process_that_shiiii() if activation matrices have not yet been
        computed. Results are cached in self.masked_arrays so subsequent calls
        are free.

        Returns concatenated arrays across all aerosol species with unactivated
        bins zeroed out. If max_cloud_thickness is set, rows above the cloud
        top are also zeroed via _apply_cloud_thickness_truncation().

        Returns
        -------
        concat_radii : np.ndarray
            Masked radii array (n_heights x total_bins), m.
        concat_Nis : np.ndarray
            Masked number concentration array (n_heights x total_bins), m^-3.
        """
        if self.activation_matrices is None:
            self.process_that_shiiii()
        
        if self.masked_arrays is not None:
            return self.masked_arrays

        masked_aer_traces = []
        masked_Ns = []

        for aer in self.initial_aerosol:
            # Tile Nis to match the height dimension for broadcasting
            aer_Ns_matrix = np.tile(aer.Nis, (len(self.parcel_trace['z']), 1))
            activation_matrix = self.activation_matrices[aer.species]
            aerosol_trace = np.array(self.aerosol_traces[aer.species])

            # Zero out unactivated bins in both radii and number concentration
            masked_aer_traces.append(np.where(activation_matrix == 1, aerosol_trace, 0))
            masked_Ns.append(np.where(activation_matrix == 1, aer_Ns_matrix, 0))

        radii = np.concatenate(masked_aer_traces, axis=1)
        Nis = np.concatenate(masked_Ns, axis=1)

        if self.max_cloud_thickness is not None:
            radii, Nis = self._apply_cloud_thickness_truncation(radii, Nis)

        self.masked_arrays = (radii, Nis)
        return self.masked_arrays

    def height_resolved_r_effs(self):
        """
        Compute height-resolved effective radius across all aerosol species.

        Uses the Hansen & Travis (1974) definition (ratio of third to second
        moment of the size distribution), implemented in compute_effective_radius.

        Returns
        -------
        np.ndarray
            1D array of effective radii at each height, m.
        """
        radii, Nis = self._get_masked_arrays()
        self.reffs = compute_effective_radius(radii, Nis)
        return self.reffs

    def height_resolved_LWCs(self):
        """
        Compute height-resolved liquid water content across all aerosol species.

        Returns
        -------
        np.ndarray
            1D array of total droplet volume concentration at each height, m^3/m^3.
        """
        radii, Nis = self._get_masked_arrays()
        self.lwcs = compute_volume(radii, Nis)
        return self.lwcs

    def height_resolved_taus(self):
        """
        Compute height-resolved layer optical depth across all aerosol species.

        Ensures LWC and r_eff have been computed first, then applies the
        geometric optics approximation: tau = 1.5 * (LWC / r_eff) * dZ.

        Returns
        -------
        np.ndarray
            1D array of layer optical depths at each height.
            Returns None if taus have already been computed (cached).
        """
        if self.lwcs is None:
            self.height_resolved_LWCs()
        if self.reffs is None:
            self.height_resolved_r_effs()
        if self.taus is None:
            self.taus = compute_tau(self.lwcs, self.reffs, self.parcel_trace['z'])
            return self.taus

        return self.taus