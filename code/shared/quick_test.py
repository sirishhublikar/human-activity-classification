import sys
sys.path.insert(0, 'code/shared')
from data_loader import load_dat_file
from signal_processing import compute_spectrogram
from pathlib import Path

# grab any one .dat file
fp = next(Path('data').rglob('*.dat'))
print(f"Testing: {fp.name}")

radar = load_dat_file(fp)
print(f"  fc={radar['fc']/1e9:.1f} GHz  NTS={radar['NTS']}  "
      f"nc={radar['nc']}  duration={radar['record_length']:.1f}s")

spec, d_axis, t_axis = compute_spectrogram(radar)
print(f"  spectrogram shape: {spec.shape}")
print("  OK!")