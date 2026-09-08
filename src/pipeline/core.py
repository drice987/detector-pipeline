import time
import logging
from pathlib import Path
import numpy as np
import h5py
from scipy.ndimage import uniform_filter, median_filter, shift
from scipy.optimize import curve_fit
from watchdog.events import FileSystemEventHandler
import yaml
from typing import Any

from .io import read_detector_file, valid_extensions
from .stats import WelfordRollingStats, EMAStats
from .rendering import DashboardRenderer

class AutomatedDetectorPipeline:
    """
    Live data reduction pipeline for 2D detectors.
    Handles background subtraction, alignment, and rolling statistics.
    """    
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        dir_cfg = self.config['directories']
        proc_cfg = self.config['processing']
        export_cfg = self.config['export']
        qa_cfg = self.config.get('qa', {})
        init_cfg = self.config.get('initialization', {})
        filter_cfg = self.config.get('filtering', {})

        self.watch_dir = Path(dir_cfg['watch_dir'])
        self.watch_dir.mkdir(exist_ok=True)
        self.processed_files = set()
        
        # Pipeline Parameters
        self.integration_axis = proc_cfg.get('integration_axis', 'vertical')
        self.alignment_mode = proc_cfg.get('alignment_mode', 'dynamic')
        
        self.use_dark = dir_cfg.get('use_dark', False)
        self.dark_filename = dir_cfg.get('dark_filename', 'dark_background.h5')
        self.file_poll_interval = dir_cfg.get('poll_interval_sec', 0.01)
        self.file_timeout = dir_cfg.get('file_timeout_sec', 10.0)
        
        self.min_init_scans = init_cfg.get('min_init_scans', 3)
        self.glitch_threshold = filter_cfg.get('glitch_threshold_pct', 0.05)
        
        # State and Buffers
        self.is_initialized = False
        self.reference_center = 0.0
        
        self.init_buffer_arrays = []
        self.init_buffer_arrays_2d = []
        self.init_buffer_intensities = []
        self.init_buffer_files = []
        self.init_buffer_i0 = []
        self.init_buffer_metadata = []
        
        # Rolling stats
        self.n_total_scans = 0
        self.correlation_history = []
        method = proc_cfg.get('statistics_method', 'welford').lower()
        if method == 'welford':
            self.stats = WelfordRollingStats()
        elif method == 'ema':
            alpha = proc_cfg.get('ema_alpha', 0.1)
            self.stats = EMAStats(alpha=alpha)
        else:
            raise ValueError(f'Unknown statistics_method: {method}')
        
        # QA and metadata
        self.track_integrity = qa_cfg.get('enable_integrity_tracking', True)
        self.integrity_window = qa_cfg.get('rolling_window_size', 200)
        self.qa_R = []
        self.metadata_config = export_cfg.get('metadata_paths', 'none')
        self.metadata_history = {}
        
        # Export control
        self.export_throttle = export_cfg.get('throttle_sec', 0.5)
        self.output_h5_name = export_cfg.get('output_h5_filename', 'CURRENT_AVERAGE.h5')
        self.file_poll_interval = self.config.get('poll_interval_sec', 0.01)
        self.file_timeout = self.config.get('file_timeout_sec', 10.0)
        self.last_export_time = 0.0
        self.needs_export = False
        
        self.renderer = DashboardRenderer(self.config, self.watch_dir)
        
        # Hardware Calibration
        self.master_dark = None
        self.dark_1d = None
        self._initialize_dark()

    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """Detects and patches NaNs and Infs with the local array median."""
        bad_mask = ~np.isfinite(data)
        if bad_mask.any():
            safe_median = np.nanmedian(data) if not np.all(bad_mask) else 0.0
            data[bad_mask] = safe_median
        return data

    def _initialize_dark(self) -> None:
        """Locates, cleans, and slices the master dark frame."""
        if not self.use_dark: return
        dark_path = self.watch_dir / self.dark_filename
        
        if dark_path.is_file():
            try:
                raw_dark, _, _ = read_detector_file(dark_path, self.config)
            except (ImportError, OSError, KeyError, ValueError) as e:
                self.logger.error(f"Dark frame load failed for ({dark_path.name}): {e}")
                return
                
            raw_dark = self._clean_data(raw_dark)
            
            r_start, r_end = self.config['processing'].get('row_bounds', [0, raw_dark.shape[0]])
            c_start, c_end = self.config['processing'].get('col_bounds', [0, raw_dark.shape[1]])
            
            self.master_dark = self._remove_cosmic_rays(raw_dark[r_start:r_end, c_start:c_end])
            self.processed_files.add(dark_path)
            
            if self.integration_axis == 'vertical':
                self.dark_1d = np.sum(self.master_dark, axis=0)
            else:
                self.dark_1d = np.sum(self.master_dark, axis=1)

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
        max_glitches = z_grid.size * self.glitch_threshold

        if 0 < num_glitches < max_glitches:
            pad = filter_size // 2
            padded_grid = np.pad(z_grid, pad, mode='reflect')
            for y, x in zip(bad_y, bad_x):
                neighborhood = padded_grid[y : y + filter_size, x : x + filter_size]
                cleaned[y, x] = np.median(neighborhood)
        elif num_glitches >= max_glitches:
            cleaned = median_filter(z_grid, size=filter_size)

        return cleaned

    def _find_elastic_params(self, profile: np.ndarray) -> float:
        """Fits a 1D Gaussian to locate the spectral center of mass."""
        x = np.arange(len(profile))
        max_idx = np.argmax(profile)
        amp_guess = profile[max_idx] - np.min(profile)
        
        expected_center = self.config['processing'].get('expected_peak_center', None)
        fit_window = self.config['processing'].get('fit_window', None)
        w_guess = self.config['processing'].get('alignment_width_guess', 5.0)

        if expected_center is not None and fit_window is not None:
            c_target = int(expected_center)
            w_half = int(fit_window)
            
            start_idx = max(0, c_target - w_half)
            end_idx = min(len(profile), c_target + w_half)
        else:
            start_idx = 0
            end_idx = len(profile)

        fit_profile = profile[start_idx:end_idx]
        x_fit = np.arange(start_idx, end_idx) 
        
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
            return float(popt[0])
        except RuntimeError:
            return float(c_guess_global)

    def _attempt_initialization(self) -> None:
        """Processes the cold-start buffer and establishes the reference frame."""
        centers = [self._find_elastic_params(raw_1d) for raw_1d in self.init_buffer_arrays]
        self.reference_center = np.median(centers)
                
        for raw_1d, raw_2d, i0_val, current_metadata, intensity, file_name in zip(
            self.init_buffer_arrays, 
            self.init_buffer_arrays_2d, 
            self.init_buffer_i0, 
            self.init_buffer_metadata,
            self.init_buffer_intensities,
            self.init_buffer_files):
            
            c = self._find_elastic_params(raw_1d)
            shift_val = self.reference_center - c if self.alignment_mode == 'dynamic' else 0.0
            
            if shift_val == 0.0:
                aligned_1d = raw_1d
                aligned_2d = raw_2d
            else:
                aligned_1d = shift(raw_1d, shift_val, mode='nearest', order=1)
                shift_2d_vec = [0, shift_val] if self.integration_axis == 'vertical' else [shift_val, 0]
                aligned_2d = shift(raw_2d, shift_2d_vec, mode='nearest', order=1)
            
            norm_factor = i0_val if (i0_val is not None and i0_val > 0) else np.sum(aligned_1d)
            norm_factor = max(norm_factor, 1e-8)
            
            normed_1d = aligned_1d / norm_factor
            normed_2d = aligned_2d / norm_factor
            
            self.stats.update(aligned_1d, aligned_2d, normed_1d, normed_2d, intensity)
            self.logger.info(f"[INIT] {file_name}")
            
            for key, val in current_metadata.items():
                self.metadata_history.setdefault(key, []).append(val)
            
        self.is_initialized = True
        self.needs_export = True

        self.init_buffer_arrays.clear()
        self.init_buffer_arrays_2d.clear()

    def process_file(self, file_path: Path) -> None:
        """Main processing loop for detector files."""
        if file_path.name == self.output_h5_name or file_path in self.processed_files:
            return
        
        self.n_total_scans += 1
        if not self._wait_for_write_completion(file_path):
            return

        i0_val = None
        current_metadata = {}
        
        try:
            raw_2d, i0_val, current_metadata = read_detector_file(file_path, self.config)
        except (ImportError, OSError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to read {file_path.name}: {e}")
            return
                
        raw_2d = self._clean_data(raw_2d)
        
        r_start, r_end = self.config['processing'].get('row_bounds', [0, raw_2d.shape[0]])
        c_start, c_end = self.config['processing'].get('col_bounds', [0, raw_2d.shape[1]])
        grid = raw_2d[r_start:r_end, c_start:c_end]
        current_raw_int = np.sum(grid)
        
        if self.master_dark is not None:
            grid = grid - self.master_dark
            
        clean_grid = self._remove_cosmic_rays(grid)
        raw_1d = np.sum(clean_grid, axis=0) if self.integration_axis == 'vertical' else np.sum(clean_grid, axis=1)

        if not self.is_initialized:
            self.init_buffer_arrays.append(raw_1d)
            self.init_buffer_arrays_2d.append(clean_grid)
            self.init_buffer_intensities.append(current_raw_int)
            self.init_buffer_files.append(file_path.name)
            self.init_buffer_i0.append(i0_val)
            self.init_buffer_metadata.append(current_metadata)
            
            if len(self.init_buffer_arrays) >= self.min_init_scans: 
                self._attempt_initialization()
        else:
            c = self._find_elastic_params(raw_1d)
            shift_val = self.reference_center - c if self.alignment_mode == 'dynamic' else 0.0
            
            if shift_val == 0.0:
                aligned_1d = raw_1d
                aligned_2d = clean_grid
            else:
                aligned_1d = shift(raw_1d, shift_val, mode='nearest', order=1)
                shift_2d_vec = [0, shift_val] if self.integration_axis == 'vertical' else [shift_val, 0]
                aligned_2d = shift(clean_grid, shift_2d_vec, mode='nearest', order=1)
            
            norm_factor = i0_val if (i0_val is not None and i0_val > 0) else np.sum(aligned_1d)
            norm_factor = max(norm_factor, 1e-8)
            
            normed_1d = aligned_1d / norm_factor
            normed_2d = aligned_2d / norm_factor
            
            corr = np.corrcoef(normed_1d, self.stats.mean_1d_norm)[0, 1]
            self.correlation_history.append(corr)

            if self.track_integrity:
                self.qa_R.append(corr)
                if len(self.qa_R) > self.integrity_window:
                    self.qa_R.pop(0)

            if corr > self.config['thresholds']['shape_correlation_min']:
                self.stats.update(aligned_1d, aligned_2d, normed_1d, normed_2d, current_raw_int)
                self.logger.info(f"Accepted {file_path.name} (R={corr:.3f})")
                self.needs_export = True
                
                for key, val in current_metadata.items():
                    self.metadata_history.setdefault(key, []).append(val)
            else:
                self.logger.warning(f"Rejected {file_path.name} (R={corr:.3f})")
                self.needs_export = True 
        
        self.processed_files.add(file_path)
        self._export_data_and_plot()

    def _wait_for_write_completion(self, filepath: Path) -> bool:
        """Polls file until OS finishes writing, with a timeout to prevent hanging."""
        historical_size = -1
        start_time = time.time()
        
        while True:
            if (time.time() - start_time) > self.file_timeout:
                self.logger.warning(f"Timeout waiting for file to finish writing: {filepath.name}")
                return False
                
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
    
    def _export_data_and_plot(self, force: bool = False) -> None:
        """Exports statistical objects to HDF5 and renders the visual dashboard."""
        if not self.is_initialized or self.stats.n_scans == 0: 
            return
        
        current_time = time.time()
        if not force and (current_time - self.last_export_time) < self.export_throttle:
            return
            
        self.last_export_time = current_time
        self.needs_export = False 

        sem_raw, sem_norm = self.stats.get_sem()
        pixels = np.arange(len(self.stats.mean_1d_raw))

        h5_path = self.watch_dir / self.output_h5_name
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                with h5py.File(h5_path, 'w') as hf:
                    hf.create_dataset('energy_axis_pixels', data=pixels)
                    
                    hf.create_dataset('raw_intensity', data=self.stats.mean_1d_raw)
                    hf.create_dataset('raw_uncertainty_sem', data=sem_raw)
                    hf.create_dataset('raw_2d_roi', data=self.stats.mean_2d_raw, compression="gzip")
                    
                    hf.create_dataset('normalized_intensity', data=self.stats.mean_1d_norm)
                    hf.create_dataset('normalized_uncertainty_sem', data=sem_norm)
                    hf.create_dataset('normalized_2d_roi', data=self.stats.mean_2d_norm, compression="gzip")
                    
                    if self.dark_1d is not None:
                        hf.create_dataset('integrated_dark_1d', data=self.dark_1d)
                        
                    hf.attrs['scans_integrated'] = self.stats.n_scans
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

        # Dashboard Rendering 
        try:
            self.renderer.render(
                data_1d=self.stats.mean_1d_norm,
                data_2d=self.stats.mean_2d_norm,
                n_scans=self.stats.n_scans,
                total_scans=self.n_total_scans,
                correlation_history=self.qa_R,
                data_1d_sem=sem_norm
            )
        except Exception as e:
            self.logger.error(f"Dashboard rendering failed: {e}")


class DataFileHandler(FileSystemEventHandler):
    """Watchdog event handler to trigger processing upon file creation."""
    def __init__(self, pipeline: AutomatedDetectorPipeline) -> None:
        self.pipeline = pipeline
        self.valid_extensions = valid_extensions

    def on_created(self, event: Any) -> None:
        if event.is_directory: return
        file_path = Path(event.src_path)
        if file_path.name.startswith('.'): return
        
        if file_path.suffix in self.valid_extensions:
            self.pipeline.process_file(file_path)