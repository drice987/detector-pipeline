import h5py
import numpy as np
import time
from pathlib import Path

def generate_synthetic_run(output_dir="../synthetic_data", num_scans=150,spatial = 'vertical'):
    out_path = Path(output_dir)
        
    print(f"Scans in '{output_dir}'...")

    # Detector parameters
    rows, cols = 256, 1024
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # --- Generate Dark Frame ---
    cy, cx = rows / 2.0, cols / 2.0
    background_shape = 10.0 + 10.0 * (((y - cy) / cy)**2 + ((x - cx) / cx)**2)
    dark_filename = out_path / "dark_background.h5"
    background = np.copy(background_shape)
     # Add a few permanent "hot pixels" to the dark frame to test dark subtraction
    background[20, 150] = 3000
    background[100, 800] = 4500
    final_background = np.random.poisson(lam=background, size=(rows, cols))
    final_background = background
    if spatial == 'horizontal':
        final_background = background.T
       
    with h5py.File(dark_filename, 'w') as hf:
        hf.create_dataset('entry/data/counts', data=final_background, compression="gzip")
        
    print(f"Generated Dark: {dark_filename.name}")
    time.sleep(0.5) 

    # --- Base Spectra ---
    base_peak_params = np.array([
        [1.0, 150.0, 1.0],
        [0.7, 400.0, 1.5],
        [0.7, 700.0, 1.5],
        [0.7, 705.0, 1.5], 
        [0.7, 745.0, 1.5]
    ])

    # --- Main Loop ---
    print(f"Generating {num_scans} scans...")
    damage_start = 20

    for i in range(1, num_scans + 1):
        
        # Incident Flux (I0) decays over time
        base_i0 = 15.0 * np.exp(-i / 1000) 
        i0_val = np.random.normal(base_i0, base_i0 * 0.02)
        drift_offset = 0
        # Environmental variables
        motor_x = -5.0 + (10.0 * i / num_scans)  
        motor_y = -1.0 + (2.0 * i / num_scans)
        temperature = 295.0 + np.random.normal(0, 0.5)  
        
        # Base Background (matches dark frame base)
        final_image_clean = np.copy(background) 
        
        current_params = base_peak_params.copy()

        # --- Simulate Beam Damage ---
        if i > damage_start:
            damage = min(1.0, (i - damage_start) / (num_scans - damage_start))
            current_params[0, 0] = max(0.0, 1.0 - (1.0 * damage))
            current_params[0, 2] = 1.0 + (4.0 * damage)
            current_params[1, 1] = 400.0 - (50.0 * damage) 
            current_params[1, 0] = 0.7 + (0.5 * damage)  
            
        # --- Build Spectrum ---
        for p in range(len(current_params)):
            rel_amp = current_params[p, 0]
            center_x = current_params[p, 1] + drift_offset
            sigma_x = 12.0 * current_params[p, 2] 
            
            peak_amplitude = i0_val * 0.1 * rel_amp
            
            peak = peak_amplitude * np.exp(
                -((x - center_x) ** 2) / (2 * sigma_x ** 2)
            )
            final_image_clean += peak
            
        noisy_image = np.random.poisson(lam=np.clip(final_image_clean, 0, None))
        
        # Randomly decide how many cosmic rays hit this specific frame (0 to 5)
        num_glitches = np.random.randint(0, 6)
        for _ in range(num_glitches):
            # Keep away from the absolute edges so our streak doesn't crash the array
            gy = np.random.randint(5, rows - 5)
            gx = np.random.randint(5, cols - 5)
            
            # A cosmic ray can be a single pixel or a streak up to 6 pixels long
            track_length = np.random.randint(1, 7)
            
            # Initial massive energy
            intensity = np.random.randint(i0_val*2, i0_val * 10)
            
            curr_y, curr_x = gy, gx
            for step in range(track_length):
                noisy_image[curr_y, curr_x] += intensity
                
                # The particle walks to an adjacent pixel (creates a streak/cluster)
                curr_y += np.random.randint(-1, 2)
                curr_x += np.random.randint(-1, 2)
                
                # Charge dissipates as it travels through the silicon
                intensity = int(intensity * 0.7)
        if spatial == 'horizontal':
            noisy_image = noisy_image.T
        # --- Write to HDF5 ---
        filename = out_path / f"scan_{i:04d}.h5"
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('entry/data/counts', data=noisy_image, compression="gzip")
            hf.create_dataset('entry/instrument/beam/i0', data=i0_val)
            hf.create_dataset('entry/instrument/motor_x', data=motor_x)
            hf.create_dataset('entry/motor_y', data=motor_y)
            hf.create_dataset('entry/sample/temperature', data=temperature)
            
        print(f"Generated {filename.name}")
        time.sleep(0.0)

    print("Scans Complete!")

if __name__ == "__main__":
    generate_synthetic_run(num_scans=50,spatial= 'horizontal')