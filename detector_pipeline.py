import time
import logging
from pathlib import Path
import numpy as np
import h5py
from scipy.ndimage import uniform_filter, median_filter, shift
from scipy.optimize import curve_fit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

__version__ = "1.0.0"

class AutomatedDetectorPipeline:
    """
    Real-time data reduction pipeline for 2D detectors.
    Features include sparse despiking, sub-pixel alignment, scale-invariant 
    shape tracking, and parallel Welford statistics for absolute and normalized intensities.
    """
    
    def __init__(self, config_path="config.yaml"):
        # --- Logging Setup ---
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[logging.FileHandler("pipeline_operations.log"), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        # --- Configuration ---
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.watch_dir = Path(self.config['directories']['watch_dir'])
        self.watch_dir.mkdir(exist_ok=True)
        self.processed_files = set()
        
        # --- Pipeline Parameters ---
        self.integration_axis = self.config['processing'].get('integration_axis', 'vertical')
        self.use_dark = self.config['directories'].get('use_dark', False)
        self.dark_filename = self.config['directories'].get('dark_filename', 'dark_background.h5')
        self.min_init_scans = self.config['initialization'].get('min_init_scans', 3)
        self.alignment_mode = self.config['processing'].get('alignment_mode', 'dynamic')
        
        # --- State Machine & Buffers ---
        self.is_initialized = False
        self.reference_center = 0.0
        
        self.init_buffer_arrays = []
        self.init_buffer_arrays_2d = []
        self.init_buffer_intensities = []
        self.init_buffer_files = []
        self.init_buffer_i0 = []
        self.init_buffer_metadata = []
        
        # --- Statistical Analysis (Parallel) ---
        self.n_valid_scans = 0
        self.n_total_scans = 0
        self.correlation_history = []
        
        self.running_mean_raw = None
        self.running_M2_raw = None
        self.running_mean_2d_raw = None
        
        self.running_mean_norm = None
        self.running_M2_norm = None
        self.running_mean_2d_norm = None
        
        self.running_raw_intensity = 0.0
        
        # --- QA & Metadata ---
        self.track_integrity = self.config.get('qa', {}).get('enable_integrity_tracking', True)
        self.integrity_window = self.config.get('qa', {}).get('rolling_window_size', 200)
        self.qa_R = []
        
        self.metadata_config = self.config.get('export', {}).get('metadata_paths', 'none')
        self.metadata_history = {}
        
        # --- Export Control ---
        self.export_throttle = self.config.get('export', {}).get('throttle_sec', 0.5)
        self.file_poll_interval = self.config.get('directories', {}).get('poll_interval_sec', 0.01)
        self.output_h5_name = self.config.get('export', {}).get('output_h5_filename', 'CURRENT_AVERAGE.h5')
        self.output_dash_name = self.config.get('export', {}).get('output_dashboard_filename', 'PIPELINE_DASHBOARD.png')
        self.last_export_time = 0.0
        self.needs_export = False
        
        # --- Hardware Calibration ---
        self.master_dark = None
        self.dark_1d = None
        self._initialize_dark()

    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """Detects and patches NaNs and Infs with the local array median."""
        bad_mask = ~np.isfinite(data)
        if bad_mask.any():
            clean_data = np.copy(data)
            safe_median = np.nanmedian(clean_data) if not np.all(bad_mask) else 0.0
            clean_data[bad_mask] = safe_median
            return clean_data
        return data

    def _extract_metadata(self, hf: h5py.File) -> dict:
        """Extracts environmental/motor metadata from the HDF5 file."""
        extracted = {}
        if self.metadata_config == 'none':
            return extracted
            
        if isinstance(self.metadata_config, list):
            for path in self.metadata_config:
                if path in hf:
                    extracted[path] = hf[path][()]
                    
        elif self.metadata_config == 'all':
            def safe_visitor(name, node):
                if isinstance(node, h5py.Dataset):
                    # Safety to prevent loading large arrays as metadata
                    if node.size < 100: 
                        extracted[name] = node[()]
            hf.visititems(safe_visitor)
            
        return extracted

    def _initialize_dark(self):
        """Locates, cleans, and slices the master dark frame."""
        if not self.use_dark: return
        dark_path = self.watch_dir / self.dark_filename
        
        if dark_path.is_file():
            try:
                if dark_path.suffix == '.sif':
                    try:
                        import sif_parser
                        data, _ = sif_parser.np_open(str(dark_path))
                        raw_dark = np.mean(data, axis=0) if data.shape[0] > 1 else data[0]
                    except ImportError:
                        self.logger.error(f"Cannot process {file_path.name}: 'sif_parser' is not installed.")
                        return
                elif dark_path.suffix == '.npy':
                    raw_dark = np.load(str(dark_path))
                elif dark_path.suffix in ['.h5', '.hdf5']:
                    h5_path = self.config.get('processing', {}).get('h5_data_path', 'entry/data/counts')
                    with h5py.File(dark_path, 'r') as hf:
                        raw_dark = hf[h5_path][:].astype(np.float64)
                elif dark_path.suffix in ['.tif', '.tiff']:
                    try:
                        import tifffile
                        raw_dark = tifffile.imread(str(dark_path)).astype(np.float64)
                        if raw_dark.ndim > 2:
                            raw_dark = np.mean(raw_dark, axis=0)
                    except ImportError:
                        self.logger.error(f"Cannot load dark {dark_path.name}: 'tifffile' is not installed.")
                        return
                else:
                    return

                raw_dark = self._clean_data(raw_dark)
                
                r_start, r_end = self.config['processing'].get('row_bounds', [0, raw_dark.shape[0]])
                c_start, c_end = self.config['processing'].get('col_bounds', [0, raw_dark.shape[1]])
                
                self.master_dark = self._remove_cosmic_rays(raw_dark[r_start:r_end, c_start:c_end])
                self.processed_files.add(dark_path)
                self.logger.info(f"Loaded and sliced master dark: {dark_path.name}")
                
                if self.integration_axis == 'vertical':
                    self.dark_1d = np.sum(self.master_dark, axis=0)
                else:
                    self.dark_1d = np.sum(self.master_dark, axis=1)
                    
            except Exception as e:
                self.logger.error(f"Failed to load dark ({dark_path.name}): {e}")

    def _remove_cosmic_rays(self, z_grid: np.ndarray) -> np.ndarray:
        """
        Sparse despiking algorithm. Uses a rapid uniform filter to identify 
        anomalies, falling back to a heavy median filter only if necessary.
        """
        filter_size = self.config['filtering']['cosmic_ray_size']
        sigma = self.config['filtering']['cosmic_ray_sigma']
        
        local_mean = uniform_filter(z_grid, size=filter_size)
        diff = z_grid - local_mean
        mad = np.median(np.abs(diff)) + 1e-8
        
        glitch_mask = diff > (sigma * mad)  
        cleaned = np.copy(z_grid)
        bad_y, bad_x = np.nonzero(glitch_mask)
        num_glitches = len(bad_y)
        
        if 0 < num_glitches < (z_grid.size * 0.05):
            pad = filter_size // 2
            padded_grid = np.pad(z_grid, pad, mode='reflect')
            for y, x in zip(bad_y, bad_x):
                neighborhood = padded_grid[y : y + filter_size, x : x + filter_size]
                cleaned[y, x] = np.median(neighborhood)
        elif num_glitches >= (z_grid.size * 0.05):
            cleaned = median_filter(z_grid, size=filter_size)

        return cleaned

    def _find_elastic_params(self, profile: np.ndarray):
        """Fits a 1D Gaussian to locate the spectral center of mass."""
        x = np.arange(len(profile))
        max_idx = np.argmax(profile)
        amp_guess = profile[max_idx] - np.min(profile)
        
        expected_center = self.config['processing'].get('expected_peak_center', None)
        fit_window = self.config['processing'].get('fit_window', None)
        w_guess = self.config['processing'].get('alignment_width_guess', 5.0)

        # Determine the mathematical bounds
        if expected_center is not None and fit_window is not None:
            c_target = int(expected_center)
            w_half = int(fit_window)
            
            # Calculate the slice indices 
            start_idx = max(0, c_target - w_half)
            end_idx = min(len(profile), c_target + w_half)
        else:
            # Fallback to the whole array if YAML is missing the constraints
            start_idx = 0
            end_idx = len(profile)

        # Isolate the data 
        fit_profile = profile[start_idx:end_idx]
        x_fit = np.arange(start_idx, end_idx) 
        
        # Generate local guesses based on the isolated window
        local_max_idx = np.argmax(fit_profile)
        amp_guess = fit_profile[local_max_idx] - np.min(fit_profile)
        c_guess_global = x_fit[local_max_idx]

        def _gaussian(x_val, c, a, w, o):
            return a * np.exp(-((x_val - c)**2) / (2 * w**2)) + o
            
        try:
            fit_bounds = ([x_fit[0], 0, 0.1, -np.inf], [x_fit[-1], np.inf, np.inf, np.inf])
            
            popt, _ = curve_fit(_gaussian, x_fit, fit_profile, 
                                p0=[c_guess_global, amp_guess, w_guess, np.min(fit_profile)],
                                bounds=fit_bounds)
            return popt[0], popt[1], popt[3], popt[2] 
        except RuntimeError:
            return float(c_guess_global), amp_guess, np.min(fit_profile), w_guess

    def _welford_update(self, raw_1d: np.ndarray, raw_2d: np.ndarray, 
                        norm_1d: np.ndarray, norm_2d: np.ndarray, raw_intensity: float):
        """Updates rolling statistics for both absolute and normalized tracks."""
        if self.n_valid_scans == 0:
            self.running_mean_raw = np.copy(raw_1d)
            self.running_M2_raw = np.zeros_like(raw_1d)
            self.running_mean_2d_raw = np.copy(raw_2d)
            
            self.running_mean_norm = np.copy(norm_1d)
            self.running_M2_norm = np.zeros_like(norm_1d)
            self.running_mean_2d_norm = np.copy(norm_2d)
            
            self.running_raw_intensity = raw_intensity
            self.n_valid_scans = 1
        else:
            self.n_valid_scans += 1
            self.running_raw_intensity += (raw_intensity - self.running_raw_intensity) / self.n_valid_scans
            
            # Raw Math
            delta_raw = raw_1d - self.running_mean_raw
            self.running_mean_raw += delta_raw / self.n_valid_scans
            self.running_M2_raw += delta_raw * (raw_1d - self.running_mean_raw)
            delta_2d_raw = raw_2d - self.running_mean_2d_raw
            self.running_mean_2d_raw += delta_2d_raw / self.n_valid_scans
            
            # Normalized Math
            delta_norm = norm_1d - self.running_mean_norm
            self.running_mean_norm += delta_norm / self.n_valid_scans
            self.running_M2_norm += delta_norm * (norm_1d - self.running_mean_norm)
            delta_2d_norm = norm_2d - self.running_mean_2d_norm
            self.running_mean_2d_norm += delta_2d_norm / self.n_valid_scans

    def _attempt_initialization(self):
        """Processes the cold-start buffer and establishes the reference frame."""
        self.logger.info("Sufficient files buffered...")
        n_scans = len(self.init_buffer_arrays)
        
        centers = []
        for raw_1d in self.init_buffer_arrays:
            c, _, _, _ = self._find_elastic_params(raw_1d)
            centers.append(c)
        
        self.reference_center = np.median(centers)
        
        for i in range(n_scans):
            raw_1d = self.init_buffer_arrays[i]
            raw_2d = self.init_buffer_arrays_2d[i]
            i0_val = self.init_buffer_i0[i]
            current_metadata = self.init_buffer_metadata[i]
            
            c, _, _, _ = self._find_elastic_params(raw_1d)
            shift_val = self.reference_center - c if self.alignment_mode == 'dynamic' else 0.0
            
            if shift_val == 0.0:
                aligned_1d = raw_1d
                aligned_2d = raw_2d
            else:
                aligned_1d = shift(raw_1d, shift_val, mode='nearest', order=1)
                shift_2d_vec = [0, shift_val] if self.integration_axis == 'vertical' else [shift_val, 0]
                aligned_2d = shift(raw_2d, shift_2d_vec, mode='nearest', order=1)
            
            if i0_val is not None and i0_val > 0.0:
                safe_denominator = i0_val
            else:
                total_area = np.sum(aligned_1d)
                safe_denominator = total_area if total_area != 0 else 1e-8 
            
            normed_1d = aligned_1d / safe_denominator
            normed_2d = aligned_2d / safe_denominator
            
            self._welford_update(aligned_1d, aligned_2d, normed_1d, normed_2d, self.init_buffer_intensities[i])
            self.logger.info(f"[INIT] {self.init_buffer_files[i]}")
            
            # Archive metadata for initialization scans
            for key, val in current_metadata.items():
                if key not in self.metadata_history:
                    self.metadata_history[key] = []
                self.metadata_history[key].append(val)
            
        self.is_initialized = True
        self.needs_export = True

    def process_file(self, file_path: Path):
        """Main processing loop for detector files."""
        if file_path.name == self.output_h5_name: return
        if file_path in self.processed_files: return
        
        self.n_total_scans += 1
        if not self._wait_for_write_completion(file_path): return

        try:
            i0_val = None
            current_metadata = {}
            
            # Read File
            if file_path.suffix == '.sif':
                try:
                    import sif_parser
                    data, _ = sif_parser.np_open(str(file_path))
                    raw_2d = np.mean(data, axis=0) if data.shape[0] > 1 else data[0]
                    raw_2d = raw_2d.astype(np.float64)
                except ImportError:
                    self.logger.error(f"Cannot process {file_path.name}: 'sif_parser' is not installed.")
                    return
            elif file_path.suffix == '.npy':
                raw_2d = np.load(str(file_path)).astype(np.float64)
            elif file_path.suffix in ['.tif', '.tiff']:
                try:
                    import tifffile
                    raw_2d = tifffile.imread(str(file_path)).astype(np.float64)
                except ImportError:
                    self.logger.error(f"Cannot process {file_path.name}: 'tifffile' is not installed.")
                    return
            elif file_path.suffix in ['.h5', '.hdf5']:
                h5_path = self.config.get('processing', {}).get('h5_data_path', 'entry/data/counts')
                i0_path = self.config.get('processing', {}).get('h5_i0_path', None)
                with h5py.File(file_path, 'r') as hf:
                    raw_2d = hf[h5_path][:].astype(np.float64)
                    if i0_path and i0_path in hf:
                        i0_val = float(np.mean(hf[i0_path][()]))
                    current_metadata = self._extract_metadata(hf)
            else:
                return
                
            raw_2d = self._clean_data(raw_2d)
            
            # Application of Bounds & Background Subtraction
            r_start, r_end = self.config['processing'].get('row_bounds', [0, raw_2d.shape[0]])
            c_start, c_end = self.config['processing'].get('col_bounds', [0, raw_2d.shape[1]])
            grid = raw_2d[r_start:r_end, c_start:c_end]
            current_raw_int = np.sum(grid)
            
            if self.master_dark is not None:
                grid = grid - self.master_dark
            clean_grid = self._remove_cosmic_rays(grid)
            
            raw_1d = np.sum(clean_grid, axis=0) if self.integration_axis == 'vertical' else np.sum(clean_grid, axis=1)

            # Route to Buffer or Main 
            if not self.is_initialized:
                self.init_buffer_arrays.append(raw_1d)
                self.init_buffer_arrays_2d.append(clean_grid)
                self.init_buffer_intensities.append(current_raw_int)
                self.init_buffer_files.append(file_path.name)
                self.init_buffer_i0.append(i0_val)
                self.init_buffer_metadata.append(current_metadata)
                
                if len(self.init_buffer_arrays) == self.min_init_scans: 
                    self._attempt_initialization()
            else:
                c, _, _, _ = self._find_elastic_params(raw_1d)
                shift_val = self.reference_center - c if self.alignment_mode == 'dynamic' else 0.0
                
                if shift_val == 0.0:
                    aligned_1d = raw_1d
                    aligned_2d = clean_grid
                else:
                    aligned_1d = shift(raw_1d, shift_val, mode='nearest', order=1)
                    shift_2d_vec = [0, shift_val] if self.integration_axis == 'vertical' else [shift_val, 0]
                    aligned_2d = shift(clean_grid, shift_2d_vec, mode='nearest', order=1)
                
                if i0_val is not None and i0_val > 0.0:
                    safe_denominator = i0_val
                else:
                    total_area = np.sum(aligned_1d)
                    safe_denominator = total_area if total_area != 0 else 1e-8 
                
                normed_1d = aligned_1d / safe_denominator
                normed_2d = aligned_2d / safe_denominator
                
                corr = np.corrcoef(normed_1d, self.running_mean_norm)[0, 1]
                self.correlation_history.append(corr)

                if self.track_integrity:
                    self.qa_R.append(corr)
                    if len(self.qa_R) > self.integrity_window:
                        self.qa_R.pop(0)

                if corr > self.config['thresholds']['shape_correlation_min']:
                    self._welford_update(aligned_1d, aligned_2d, normed_1d, normed_2d, current_raw_int)
                    self.logger.info(f"[ACCEPTED] {file_path.name} (R={corr:.3f})")
                    self.needs_export = True
                    
                    for key, val in current_metadata.items():
                        if key not in self.metadata_history:
                            self.metadata_history[key] = []
                        self.metadata_history[key].append(val)
                else:
                    self.logger.warning(f"[REJECTED] {file_path.name} (R={corr:.3f})")
                    self.needs_export = True 
            
            self.processed_files.add(file_path)
            self._export_data_and_plot()
        except Exception as e:
            self.logger.critical(f"Critical error on {file_path.name}: {e}")

    def _wait_for_write_completion(self, filepath: Path):
        """Format-agnostic fast-polling to ensure OS write operations are complete."""
        historical_size = -1
        while True:
            try:
                current_size = filepath.stat().st_size
                if current_size == historical_size and current_size > 0:
                    try:
                        with open(filepath, 'rb+') as f:
                            return True
                    except (IOError, OSError):
                        pass 
                historical_size = current_size
            except FileNotFoundError:
                pass
            time.sleep(self.file_poll_interval)
    
    def _export_data_and_plot(self, force=False):
        """Exports statistical objects to HDF5 and renders the visual dashboard."""
        if not self.is_initialized or self.n_valid_scans == 0: 
            return
        
        current_time = time.time()
        if not force and (current_time - self.last_export_time) < self.export_throttle:
            return
            
        self.last_export_time = current_time
        self.needs_export = False 

        if self.n_valid_scans > 1:
            var_raw = self.running_M2_raw / (self.n_valid_scans - 1)
            sem_raw = np.sqrt(var_raw / self.n_valid_scans)
            
            var_norm = self.running_M2_norm / (self.n_valid_scans - 1)
            sem_norm = np.sqrt(var_norm / self.n_valid_scans)
        else:
            sem_raw = np.zeros_like(self.running_mean_raw)
            sem_norm = np.zeros_like(self.running_mean_norm)

        pixels = np.arange(len(self.running_mean_raw))

        # --- Data Archive Export ---
        h5_path = self.watch_dir / self.output_h5_name
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                with h5py.File(h5_path, 'w') as hf:
                    hf.create_dataset('energy_axis_pixels', data=pixels)
                    
                    hf.create_dataset('raw_intensity', data=self.running_mean_raw)
                    hf.create_dataset('raw_uncertainty_sem', data=sem_raw)
                    hf.create_dataset('raw_2d_roi', data=self.running_mean_2d_raw, compression="gzip")
                    
                    hf.create_dataset('normalized_intensity', data=self.running_mean_norm)
                    hf.create_dataset('normalized_uncertainty_sem', data=sem_norm)
                    hf.create_dataset('normalized_2d_roi', data=self.running_mean_2d_norm, compression="gzip")
                    
                    if self.dark_1d is not None:
                        hf.create_dataset('integrated_dark_1d', data=self.dark_1d)
                        
                    hf.attrs['scans_integrated'] = self.n_valid_scans
                    hf.attrs['scans_total_attempted'] = self.n_total_scans
                    
                    if self.metadata_history:
                        meta_group = hf.create_group('scan_metadata')
                        for key, val_list in self.metadata_history.items():
                            try:
                                meta_group.create_dataset(key.replace('/', '_'), data=np.asarray(val_list))
                            except Exception as e:
                                self.logger.debug(f"Skipped metadata {key} due to format error: {e}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.05)
                else:
                    self.logger.warning(f"Could not write HDF5 after {max_retries} attempts (file lock): {e}")

        # --- Dashboard Rendering ---
        axis_mode = self.config['processing'].get('integration_axis', 'vertical')
        corr_threshold = self.config.get('thresholds', {}).get('shape_correlation_min', 0.95)
        
        try:
            generate_plot(
                data_1d = self.running_mean_norm,
                data_2d = self.running_mean_2d_norm,
                integration_axis = axis_mode,
                title = "Pipeline Dashboard",
                n_scans = self.n_valid_scans,
                correlation_history = self.qa_R,
                threshold = corr_threshold,
                save_path = self.watch_dir / self.output_dash_name,
                data_1d_sem = sem_norm,
                total_scans = self.n_total_scans
            )
        except Exception as e:
            print(f"Dashboard error: {e}")

def generate_plot(data_1d, data_2d, integration_axis="vertical", 
                            title="Spectroscopy Scan", n_scans=None, 
                            correlation_history=None, threshold = None,
                            save_path=None, data_1d_sem=None, total_scans = None
                            ):
        """
        Universal plotter for pipeline or jupyter notebook
        """
        fig = plt.figure(figsize=(10, 8))
        
        if n_scans is not None:
            title = f"{title} | Scans: {n_scans}"

        has_qa = correlation_history is not None and len(correlation_history) > 0
        
        if has_qa:
            gs_main = gridspec.GridSpec(2, 1, height_ratios=[2, 0.7], hspace=0.3)
            gs_spectra = gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=gs_main[0], 
                height_ratios=[1, 1], width_ratios=[1, 0.03], 
                hspace=0.0, wspace=0.02
            )
            ax3 = fig.add_subplot(gs_main[1])
        else:
            # If no qa
            gs_spectra = gridspec.GridSpec(2, 2, width_ratios=[1, 0.03], hspace=0.0, wspace=0.02)

        # --- 2. TOP: 1D Spectrum ---
        ax1 = fig.add_subplot(gs_spectra[0, 0])
        ax1.plot(data_1d, color='blue')
        if data_1d_sem is not None:
            x_vals = np.arange(len(data_1d))
            ax1.fill_between(x_vals, 
                            data_1d - 2*data_1d_sem, 
                            data_1d + 2*data_1d_sem, 
                            color='blue', alpha=0.3, label="2σ SEM")
            ax1.legend(loc="upper right")
        ax1.set_title(title, fontsize=14, pad=15)
        ax1.set_ylabel("Normalized Intensity")
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # --- 3. MIDDLE: 2D Heatmap ---
        ax2 = fig.add_subplot(gs_spectra[1, 0], sharex=ax1)
        
        if integration_axis == 'vertical':
            im = ax2.imshow(data_2d, aspect='auto', cmap='magma', origin='lower')
            ax2.set_xlabel("Dispersive Axis")
            ax2.set_ylabel("Spatial Axis")
        else:
            # Transpose trick for horizontal 
            im = ax2.imshow(data_2d.T, aspect='auto', cmap='magma', origin='lower')
            ax2.set_xlabel("Dispersive Axis")
            ax2.set_ylabel("Spatial Axis")

        # --- 4. COLORBAR ---
        cax = fig.add_subplot(gs_spectra[1, 1])
        plt.colorbar(im, cax=cax, label='Normalized Intensity')

        # --- 5. BOTTOM: qa ---
        if has_qa:
            corr_arr = np.array(correlation_history)
            scans = np.arange(1, len(corr_arr) + 1)

            end_scan = total_scans if total_scans else len(corr_arr)
            start_scan = max(1, end_scan - len(corr_arr) + 1)
            scans = np.arange(start_scan, end_scan + 1)
                       
            # Create masks for accepted and rejected data
            accepted = corr_arr > threshold
            rejected = corr_arr <= threshold
            
            # Plot Accepted
            if np.any(accepted):
                ax3.scatter(scans[accepted], corr_arr[accepted], color='blue', marker='.', s=30, label="Accepted")
                
            # Plot Rejected
            if np.any(rejected):
                ax3.scatter(scans[rejected], corr_arr[rejected], color='red', marker='x', s=20, label="Rejected")
            
            ax3.set_title("Sample Health (Pearson Correlation)")
            ax3.set_ylabel("R-Value")
            ax3.set_xlabel("Scan Number")
            ax3.set_ylim(min(corr_arr) - 0.05, max(corr_arr) + 0.05)

        # --- EXPORT LOGIC ---
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig) 
        else:
            return fig
        
class DataFileHandler(FileSystemEventHandler):
    """Watchdog event handler to trigger processing upon file creation."""
    def __init__(self, pipeline: AutomatedDetectorPipeline):
        self.pipeline = pipeline
        self.valid_extensions = {'.npy', '.h5', '.hdf5', '.sif','.tif','.tiff'}

    def on_created(self, event):
        if event.is_directory: return
        file_path = Path(event.src_path)
        if file_path.name.startswith('.'): return
        
        if file_path.suffix in self.valid_extensions:
            self.pipeline.process_file(file_path)


if __name__ == "__main__":
    # Force Matplotlib to run in headless background mode
    matplotlib.use('Agg')


    pipeline = AutomatedDetectorPipeline()
    
    # --- Check for Existing Backlog ---
    print(f"Checking {pipeline.watch_dir} for existing backlog...")
    valid_extensions = {'.npy', '.h5', '.hdf5', '.sif','.tif','.tiff'}
    
    existing_files = sorted([
        f for f in pipeline.watch_dir.iterdir() 
        if f.is_file() and f.suffix in valid_extensions and not f.name.startswith('.')
    ])
    
    if existing_files:
        for file_path in existing_files:
            pipeline.process_file(file_path)


    # --- Live System Monitoring ---
    event_handler = DataFileHandler(pipeline)
    observer = Observer()
    observer.schedule(event_handler, path=str(pipeline.watch_dir), recursive=False)
    observer.start()
    
    print(f"Monitoring Directory: {pipeline.watch_dir} ... (Press Ctrl+C to stop)")
    
    try:
        while True:
            time.sleep(1)
            # Idle flush: Force a final dashboard update if data flow stops
            if pipeline.needs_export and (time.time() - pipeline.last_export_time) > 1.0:
                pipeline.logger.info("Pipeline idle. Flushing final dashboard frame...")
                pipeline._export_data_and_plot(force=True)
                
    except KeyboardInterrupt:
        observer.stop()
        print("\nShutting down pipeline...")
        
    observer.join()