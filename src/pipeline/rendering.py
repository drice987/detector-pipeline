from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DashboardRenderer:
    """Handles the rendering and exporting of pipeline visual dashboards."""
    
    def __init__(self, config: dict, watch_dir: Path):
        self.integration_axis = config['processing'].get('integration_axis', 'vertical')
        self.corr_threshold = config.get('thresholds', {}).get('shape_correlation_min', 0.95)
        
        output_dash_name = config.get('export', {}).get('output_dashboard_filename', 'PIPELINE_DASHBOARD.png')
        self.save_path = watch_dir / output_dash_name
        self.title = "Pipeline Dashboard"
        
        matplotlib.use('Agg')

    def render(self, data_1d: np.ndarray, data_2d: np.ndarray, n_scans: int, 
               total_scans: int, correlation_history: list[float], 
               data_1d_sem: np.ndarray | None = None) -> None:
        """Generates and saves the pipeline dashboard."""
        
        fig = plt.figure(figsize=(10, 8))
        current_title = f"{self.title} | Scans: {n_scans}"

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
            gs_spectra = gridspec.GridSpec(2, 2, width_ratios=[1, 0.03], hspace=0.0, wspace=0.02)

        # 1D Spectrum
        ax1 = fig.add_subplot(gs_spectra[0, 0])
        ax1.plot(data_1d, color='blue')
        if data_1d_sem is not None:
            x_vals = np.arange(len(data_1d))
            ax1.fill_between(x_vals, 
                             data_1d - 2*data_1d_sem, 
                             data_1d + 2*data_1d_sem, 
                             color='blue', alpha=0.3, label="2σ SEM")
            ax1.legend(loc="upper right")
        ax1.set_title(current_title, fontsize=14, pad=15)
        ax1.set_ylabel("Normalized Intensity")
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # 2D
        ax2 = fig.add_subplot(gs_spectra[1, 0], sharex=ax1)
        if self.integration_axis == 'vertical':
            im = ax2.imshow(data_2d, aspect='auto', cmap='magma', origin='lower')
        else:
            im = ax2.imshow(data_2d.T, aspect='auto', cmap='magma', origin='lower')
            
        ax2.set_xlabel("Dispersive Axis")
        ax2.set_ylabel("Spatial Axis")

        cax = fig.add_subplot(gs_spectra[1, 1])
        plt.colorbar(im, cax=cax, label='Normalized Intensity')

        # QA
        if has_qa:
            corr_arr = np.array(correlation_history)
            end_scan = total_scans if total_scans else len(corr_arr)
            start_scan = max(1, end_scan - len(corr_arr) + 1)
            scans = np.arange(start_scan, end_scan + 1)
                     
            accepted = corr_arr > self.corr_threshold
            rejected = corr_arr <= self.corr_threshold
            
            if np.any(accepted):
                ax3.scatter(scans[accepted], corr_arr[accepted], color='blue', marker='.', s=30, label="Accepted")
            if np.any(rejected):
                ax3.scatter(scans[rejected], corr_arr[rejected], color='red', marker='x', s=20, label="Rejected")
            
            ax3.set_title("Sample Health (Pearson Correlation)")
            ax3.set_ylabel("R-Value")
            ax3.set_xlabel("Scan Number")
            ax3.set_ylim(min(corr_arr) - 0.05, max(corr_arr) + 0.05)

        fig.savefig(self.save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)