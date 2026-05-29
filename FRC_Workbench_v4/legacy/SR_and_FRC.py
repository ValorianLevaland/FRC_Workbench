import numpy as np
from tifffile import imread
from skimage.transform import resize
import napari

recon_path = r"C:\Users\ovahirua\Documents\PALM_Acquisition\2025_03_04_Attempts\2025_03_04_NO2\2025_03_04_NO2_posA_1\Test_RSP_RSE\Averaged shifted histograms-2.tif"          # your comet recon
frc_path   = r"C:\Users\ovahirua\Documents\PALM_Acquisition\2025_03_04_Attempts\2025_03_04_NO2\2025_03_04_NO2_posA_1\Test_RSP_RSE\2025_03_04_NO2_posA_comet_reconstructed_frc_map.tif"  # tile FRC map (nm)

recon = imread(recon_path)          # H×W
frc   = imread(frc_path).astype(float)   # Ty×Tx (tile grid)

# Upsample to the reconstruction grid (bilinear). Preserve NaNs for no-data tiles.
frc_up = resize(frc, recon.shape, order=1, preserve_range=True, anti_aliasing=False)
frc_up = np.where(np.isfinite(frc_up), frc_up, np.nan)

v = napari.Viewer()
v.add_image(recon, name='reconstruction', blending='opaque')
# Show “better=smaller nm” with an inverse LUT range if you like:
v.add_image(frc_up, name='FRC (nm)', colormap='turbo', blending='additive', opacity=0.6)
napari.run()
