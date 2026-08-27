import argparse
from pathlib import Path
import numpy as np
import h5py

def generate_synthetic_run(output_dir="./shared_data", num_scans=150, spatial='vertical'):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating scans in '{output_dir}'")

    # Detector parameters
    rows, cols = 256, 1024
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # Generate baseline dark frame
    cy, cx = rows / 2.0, cols / 2.0
    background_shape = 10.0 + 10.0 * (((y - cy) / cy)**2 + ((x - cx) / cx)**2)
    dark_filename = out_path / "dark_background.h5"
    
    background = np.copy(background_shape)
    # Add hot pixels
    background[20, 150] = 3000
    background[100, 800] = 4500
    
    # Poisson noise
    final_background = np.random.poisson(lam=background, size=(rows, cols))
    
    if spatial == 'horizontal':
        final_background = final_background.T
        
    with h5py.File(dark_filename, 'w') as hf:
        hf.create_dataset('entry/data/counts', data=final_background, compression="gzip")
        
    print(f"Generated Dark: {dark_filename.name}")

    # Base spectral parameters: [rel_amp, center_x, sigma_multiplier]
    base_peak_params = np.array([
        [1.0, 150.0, 1.0],
        [0.7, 400.0, 1.5],
        [0.7, 700.0, 1.5],
        [0.7, 705.0, 1.5], 
        [0.7, 745.0, 1.5]
    ])

    print(f"Generating {num_scans} scans")
    damage_start = 20

    for i in range(1, num_scans + 1):
        # Incident flux (I0) decay and env noise
        base_i0 = 15.0 * np.exp(-i / 1000) 
        i0_val = np.random.normal(base_i0, base_i0 * 0.02)
        
        motor_x = -5.0 + (10.0 * i / num_scans)  
        motor_y = -1.0 + (2.0 * i / num_scans)
        temperature = 295.0 + np.random.normal(0, 0.5)  
        
        final_image_clean = np.copy(background) 
        current_params = base_peak_params.copy()

        # Simulate sample degradation over time
        if i > damage_start:
            damage = min(1.0, (i - damage_start) / (num_scans - damage_start))
            current_params[0, 0] = max(0.0, 1.0 - (1.0 * damage))
            current_params[0, 2] = 1.0 + (4.0 * damage)
            current_params[1, 1] = 400.0 - (50.0 * damage) 
            current_params[1, 0] = 0.7 + (0.5 * damage)  
            
        # Spectrum
        for rel_amp, center_x, sigma_mult in current_params:
            sigma_x = 12.0 * sigma_mult
            peak_amplitude = i0_val * 0.1 * rel_amp
            
            peak = peak_amplitude * np.exp(
                -((x - center_x) ** 2) / (2 * sigma_x ** 2)
            )
            final_image_clean += peak
            
        noisy_image = np.random.poisson(lam=np.clip(final_image_clean, 0, None))
        
        # Add cosmic rays
        num_glitches = np.random.randint(0, 6)
        for _ in range(num_glitches):
            gy = np.random.randint(5, rows - 5)
            gx = np.random.randint(5, cols - 5)
            track_length = np.random.randint(1, 7)
            intensity = np.random.randint(int(i0_val * 2), int(i0_val * 10))
            
            curr_y, curr_x = gy, gx
            for step in range(track_length):
                noisy_image[curr_y, curr_x] += intensity
                
                curr_y += np.random.randint(-1, 2)
                curr_x += np.random.randint(-1, 2)
                intensity = int(intensity * 0.7)
                
        if spatial == 'horizontal':
            noisy_image = noisy_image.T
            
        filename = out_path / f"scan_{i:04d}.h5"
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('entry/data/counts', data=noisy_image, compression="gzip")
            hf.create_dataset('entry/instrument/beam/i0', data=i0_val)
            hf.create_dataset('entry/instrument/motor_x', data=motor_x)
            hf.create_dataset('entry/motor_y', data=motor_y)
            hf.create_dataset('entry/sample/temperature', data=temperature)
            
        # Progress update
        if i % 10 == 0 or i == num_scans:
            print(f"Generated {i}/{num_scans} scans")

    print("Scans Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generator for 2D Detector Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./shared_data',
        help='Output directory for generated HDF5 files'
    )
    parser.add_argument(
        '-n', '--scans',
        type=int,
        default=50,
        help='Number of synthetic scans to generate'
    )
    parser.add_argument(
        '-s', '--spatial',
        type=str,
        choices=['vertical', 'horizontal'],
        default='horizontal',
        help='Spatial orientation of the detector'
    )
    
    args = parser.parse_args()
    
    generate_synthetic_run(
        output_dir=args.output,
        num_scans=args.scans,
        spatial=args.spatial
    )