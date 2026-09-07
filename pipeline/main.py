import time
import argparse
from pathlib import Path
from watchdog.observers import Observer
from pipeline import AutomatedDetectorPipeline, DataFileHandler, valid_extensions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Live Data Reduction Pipeline for 2D Detectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'config',
        nargs='?', 
        default='config.yaml',
        help='Path to the YAML configuration file'
    )

    args = parser.parse_args()

    pipeline = AutomatedDetectorPipeline(config_path=args.config)
    
    existing_files = sorted([
        f for f in pipeline.watch_dir.iterdir() 
        if f.is_file() and f.suffix in valid_extensions and not f.name.startswith('.')
    ], key=lambda x: x.stat().st_mtime)
    
    if existing_files:
        for file_path in existing_files:
            pipeline.process_file(file_path)

    event_handler = DataFileHandler(pipeline)
    observer = Observer()
    observer.schedule(event_handler, path=str(pipeline.watch_dir), recursive=False)
    observer.start()

    print(f"Monitoring Directory: {pipeline.watch_dir} ... (Press Ctrl+C to stop)") 
    try:
        while True:
            time.sleep(1)
            if pipeline.needs_export and (time.time() - pipeline.last_export_time) > 1.0:
                pipeline._export_data_and_plot(force=True)
                
    except KeyboardInterrupt:
        print("Shutting down pipeline.")
    finally:
        observer.stop()
        observer.join()