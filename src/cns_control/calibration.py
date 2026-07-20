import nidaqmx
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import json
import time
import scipy.ndimage as ndi
from raman_mda_engine.aiming import (
    SimpleGridSource,
)
from skimage.measure import ransac, CircleModel, label
from scipy.ndimage import center_of_mass, binary_dilation
from matplotlib.widgets import RectangleSelector
from .coordtransformer import CoordTransformer
from scipy.interpolate import griddata

__all__ = [
    "Calibrator",
    "ManualImageSelector"
]

class Calibrator:
    """
    A class for performing laser calibration using a microscope and a DAQ-controlled galvo system.
    
    This class provides methods for:
    - Collecting calibration images at different laser positions.
    - Performing calibration by mapping brightfield (BF) coordinates to galvo voltages.

    Parameters
    ----------
    core : object
        Microscope control object with methods:
        - `setConfig()`, `setShutterOpen()`, `stopSequenceAcquisition()`, `setExposure()`, `snap()`.
    daq : object
        DAQ control object that must support:
        - `galvo.stop()`, `galvo.timing.cfg_samp_clk_timing()`, `galvo.out_stream.output_buf_size`.
    transformer : object
        A transformation utility with a `BF_to_volts()` method for converting BF coordinates to galvo voltages.
    max_volts : float, optional
        The maximum allowed galvo voltage. Default is 1.5.

    """

    def __init__(self, core, daq, transformer, collector, N, exp, max_volts=1.5):
        self.core = core
        self.daq = daq
        self.transformer = transformer
        self.collector = collector
        self.max_volts = max_volts
        self.N = N
        self.exp = exp

    def collect_calibration_images(self, volts, thres, relative_pos=None):
        """
        Collects images at specified voltage positions and saves them as an xarray Dataset.

        Parameters
        ----------
        volts : ndarray of shape (N, 2)
            The (X, Y) galvo voltages for laser positioning.
        relative_pos : ndarray of shape (N, 2), optional
            Relative brightfield (BF) positions. Default is None.

        Returns
        -------
        ds : xarray.Dataset
            Dataset containing:
            - "laser_pos" (idx, volt): The galvo voltages used for positioning.
            - "imgs" (idx, Y, X): The captured images at each position.
            - "BF_bkd" (Y, X): The background brightfield image.
            - "rel_BF_pos" (optional): The relative BF positions, if provided.
        """
        N = self.N
        exp = self.exp
        self.daq._galvo.out_stream.output_buf_size = 1000

        idx = np.abs(volts) > thres
        too_big = idx[:, 0] | idx[:, 1]

        self.daq._galvo.timing.cfg_samp_clk_timing(
            1e4, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
        )

        imgs = []
        specs = []
        self.core.setAutoShutter(False)

        for volts_xy in tqdm(volts[~too_big]):
            spec = self.collector.collect_spectra_pts(np.tile([volts_xy], (N,1)), exp)  # Collect spectral data
            imgs.append(self.core.snap())  # Capture image
            specs.append(spec)
            time.sleep(0.1)

        imgs = np.asarray(imgs)
        specs = np.asarray(specs)
        self.core.setAutoShutter(True)

        ds = xr.Dataset(
            {
                "laser_pos": xr.DataArray(volts[~too_big], dims=("idx", "volt")),
                "imgs": xr.DataArray(imgs, dims=("idx", "Y", "X")),
                "specs": xr.DataArray(specs, dims=("idx", "N", "spec_dim")),
                "BF_bkd": xr.DataArray(self.core.snap(), dims=("Y", "X")),
            }
        )

        if relative_pos is not None:
            ds["rel_BF_pos"] = xr.DataArray(relative_pos[~too_big], dims=("idx", "rel_BF"))

        import uuid
        from datetime import datetime

        ds.attrs["time"] = str(datetime.now())
        name = f"calibration_{uuid.uuid4()}.zarr"
        ds.to_zarr(name)
        print(f"Saved calibration dataset to {name}")

        return ds

    def calibrate(self, N=5, thres=1.5, plot=True):
        """
        Performs laser calibration by mapping brightfield (BF) coordinates to galvo voltages 
        and collecting images at specified positions.

        Parameters
        ----------
        N : int, optional
            Number of grid points in X and Y for calibration. Default is 5.
        plot : bool, optional
            If True, plots the acquired images with marked positions. Default is True.

        Returns
        -------
        ds : xarray.Dataset
            Dataset containing the captured images and corresponding galvo positions.

        """
        self.daq.galvo.stop()
        self.core.setConfig("Channel", "RM")
        # self.core.setShutterOpen("Fluoshutter", True)

        width = self.core.getImageWidth()
        height = self.core.getImageHeight()

        grid = SimpleGridSource(N, N)
        rel_BF = grid.get_current_points()

        volts = self.transformer.BF_to_volts(
            (rel_BF * [width, height])[:, ::-1] / [height, width], max_volts=self.max_volts
        )

        self.core.stopSequenceAcquisition()
        self.core.setExposure(1)

        ds = self.collect_calibration_images(
            volts,
            relative_pos=rel_BF * np.array([width, height])[None, :],
            thres=thres,
        )

        # self.core.setShutterOpen("Fluoshutter", False)

        if plot:
            plt.figure()
            plt.imshow(ds["imgs"].max(axis=0))  # Max projection of images
            pix_BF = rel_BF * np.array([width, height])[None, :]
            plt.scatter(pix_BF[:, 0], pix_BF[:, 1], color="r")  # Plot calibration points

        return ds
    
    def save_new_model(self, ds, selected_points, model_name):
        idx = np.isnan(selected_points)
        selected_points = np.array(selected_points)
        not_nan = ~(idx[:, 0] | idx[:, 1])
        rel_bf = (selected_points / ds["imgs"].shape[-2:])[not_nan]
        rel_rm = ((ds["laser_pos"] + self.max_volts) / (2*self.max_volts))[not_nan]
        v_degs = (3, 3)
        model = CoordTransformer.fit_model(rel_bf, rel_rm, v_degs, alpha=0.001)
        CoordTransformer.save_model(model_name + ".json", model, v_degs)
        transformer = CoordTransformer.from_json(model_name + ".json")
        return transformer
    
    def interpolate2d(self, ds, plot=True):
        intensity = np.median(ds["specs"].values, axis=1)
        coords = ds["rel_BF_pos"].values  # (N, 2) in (Y, X)

        # Define full image grid (X = cols, Y = rows)
        grid_x, grid_y = np.meshgrid(
            # np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100),
            # np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
            np.linspace(0, 1344, 1344),
            np.linspace(0, 1024, 1024)
        )

        grid_z = griddata(coords, intensity, (grid_x, grid_y), method='cubic')

        # Plot
        if plot:
            plt.figure()
            # plt.imshow(grid_z, extent=(coords[:,0].min(), coords[:,0].max(),
            #                            coords[:,1].min(), coords[:,1].max()),
            #            origin='lower', aspect='auto')
            plt.imshow(grid_z, extent=(0, 1344, 0, 1024),
                    origin='lower', aspect='auto')
            plt.imshow(ds["imgs"].max(axis=0), alpha=0.1)
            plt.scatter(coords[:,0], coords[:,1], c=intensity, edgecolor='k')

        return grid_x, grid_y, grid_z


class ManualImageSelector:
    def __init__(self, ds):
        self.images = ds['imgs'].values
        self.coms = self.find_coms(self.images)  # Now properly assigned
        self.num_images = self.images.shape[0]
        self.current_idx = 0
        self.selected_points = [(None, None)] * self.num_images
        self.manual_selections = [False] * self.num_images

        # Create figure with two subplots
        self.fig = plt.figure(figsize=(15, 7))
        self.ax_full = self.fig.add_subplot(121)  # Full image
        self.ax_zoom = self.fig.add_subplot(122)  # Zoomed view

        self.zoom_window_size = 100  # Size of zoom window in pixels
        self.zoom_scale = 4  # Zoom factor

        self.cid = None
        self.show_image()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def find_coms(self, imgs):
        """Finds the center of mass for each image based on thresholding."""
        coms = []
        for img in imgs:
            mask = img > img.mean() + 4 * img.std()
            com = ndi.center_of_mass(mask)
            coms.append(com)
        return np.array(coms)  # Now it returns the result!

    def make_image_mask(self, image, point):
        """Creates a binary mask based on image intensity and distance from selected point."""
        cy, cx = point
        mask = (image > image.mean() + 3 * image.std()) & (
            np.sqrt((np.indices(image.shape)[0] - cy) ** 2 +
                    (np.indices(image.shape)[1] - cx) ** 2) <= 300)
        mask = binary_dilation(mask, structure=np.ones((1, 1)))

        labeled_mask = label(mask)
        sizes = np.bincount(labeled_mask.ravel())
        sizes[0] = 0  # Ignore background

        largest_label = sizes.argmax()
        mask = labeled_mask == largest_label

        return mask * image  # Correctly returning masked image

    def update_zoom_window(self, center=None):
        """Update the zoomed view around the given center or current selection."""
        img = self.images[self.current_idx]
        if center is None:
            if self.manual_selections[self.current_idx]:
                center = self.selected_points[self.current_idx]
            else:
                if self.selected_points[self.current_idx] == (None, None):
                    center = self.coms[self.current_idx]
                else:
                    center = self.selected_points[self.current_idx]

        cy, cx = center
        half_size = self.zoom_window_size // 2

        # Calculate zoom window boundaries with padding
        y_min = max(0, int(cy - half_size))
        y_max = min(img.shape[0], int(cy + half_size))
        x_min = max(0, int(cx - half_size))
        x_max = min(img.shape[1], int(cx + half_size))

        # Extract and display zoomed region
        self.ax_zoom.clear()
        self.ax_zoom.imshow(img[y_min:y_max, x_min:x_max], cmap='gray')

        # If point is within zoom window, show it
        if y_min <= cy <= y_max and x_min <= cx <= x_max:
            self.ax_zoom.scatter(cx - x_min, cy - y_min, c='r', marker='x', s=100)

        self.ax_zoom.set_title("Zoomed View")
        self.fig.canvas.draw()

        # Store zoom window coordinates for click conversion
        self.zoom_coords = (x_min, x_max, y_min, y_max)

    def show_image(self):
        self.ax_full.clear()
        img = self.images[self.current_idx]

        # Use existing selection, manual point, or COM
        if self.manual_selections[self.current_idx]:
            cy, cx = self.selected_points[self.current_idx]
        else:
            if self.selected_points[self.current_idx] == (None, None):
                cy, cx = self.coms[self.current_idx]
                self.selected_points[self.current_idx] = (cy, cx)
            else:
                cy, cx = self.selected_points[self.current_idx]

        # Only create mask if point exists
        if cx is not None and cy is not None:
            mask = self.make_image_mask(img, (cy, cx))

            # Only update COM if not manually selected
            if not self.manual_selections[self.current_idx]:
                cy, cx = center_of_mass(mask)
                self.selected_points[self.current_idx] = (int(cy), int(cx))

        self.ax_full.imshow(img, cmap='gray')
        if cx is not None and cy is not None:
            self.ax_full.scatter(cx, cy, c='r', marker='x', s=100,
                                 label="Manual Selection" if self.manual_selections[self.current_idx]
                                 else "Automatic Center")

        self.ax_full.set_title(f"Image {self.current_idx + 1}/{self.num_images}\n"
                               f"Click to set center, Enter to confirm, Backspace to go back")
        self.ax_full.legend()

        # Update zoom window
        self.update_zoom_window((cy, cx))

        if self.cid:
            self.fig.canvas.mpl_disconnect(self.cid)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Check which axes was clicked
        if event.inaxes == self.ax_full:
            # Click in main image
            cy, cx = int(event.ydata), int(event.xdata)
        elif event.inaxes == self.ax_zoom:
            # Click in zoom window - convert coordinates
            x_min, x_max, y_min, y_max = self.zoom_coords
            cx = int(event.xdata + x_min)
            cy = int(event.ydata + y_min)
        else:
            return

        # Store clicked point and mark as manually selected
        self.selected_points[self.current_idx] = (cy, cx)
        self.manual_selections[self.current_idx] = True
        self.show_image()

    def on_key_press(self, event):
        if event.key == 'enter':
            if self.current_idx < self.num_images - 1:
                self.current_idx += 1
                self.show_image()
            else:
                print("Finished selection.")
                plt.close()

        elif event.key == 'backspace':
            if self.current_idx > 0:
                self.current_idx -= 1
                self.show_image()

        elif event.key == 'r':  # Reset current point
            self.manual_selections[self.current_idx] = False
            self.selected_points[self.current_idx] = (None, None)
            self.show_image()

        elif event.key.lower() == 'n':  # Mark as NaN
            self.manual_selections[self.current_idx] = True
            self.selected_points[self.current_idx] = (np.nan, np.nan)
            if self.current_idx < self.num_images - 1:
                self.current_idx += 1
                self.show_image()
            else:
                print("Finished selection.")
                plt.close()
            if self.current_idx < self.num_images - 1:
                self.current_idx += 1
                self.show_image()
            else:
                print("Finished selection.")
                plt.close()

    def start(self):
        plt.show()
        return self.selected_points
    

def _vandermonde_terms(pts, degree):
    """Design matrix with terms x^i * y^j for i + j <= degree.
 
    IMPORTANT: term ordering MUST match _vandermonde_design in
    raman_mda_engine's utils (graded order: for each total degree d,
    terms x^i * y^(d-i) with i ascending), because the model C is fit
    here and applied there. Degree 1 gives [1, y, x] under both orderings,
    which is why only degree 1 worked before this was aligned.
    """
    pts = np.atleast_2d(pts)
    x, y = pts[:, 0], pts[:, 1]
    terms = []
    for total_deg in range(degree + 1):
        for i in range(total_deg + 1):
            j = total_deg - i
            terms.append((x**i) * (y**j))
    return np.stack(terms, axis=1)
 
 
def fit_vandermonde(pts, targets, degree):
    """Least-squares fit mapping pts (N, 2) -> targets (N, 2)."""
    A = _vandermonde_terms(pts, degree)
    C, *_ = np.linalg.lstsq(A, np.atleast_2d(targets), rcond=None)
    return C
 
 
def apply_vandermonde(pts, C, degree):
    return _vandermonde_terms(pts, degree) @ C
 
 
def save_vandermonde_model(json_path, C, degree, img_center=None, xy_center=None):
    """Save in the format load_vandermonde_model expects.
 
    img_center / xy_center are stored as extra keys for reference
    (dict-based loaders that only read "degree" and "C" ignore them).
    """
    model = {"degree": int(degree), "C": np.asarray(C).tolist()}
    if img_center is not None:
        model["img_center"] = np.asarray(img_center).tolist()
    if xy_center is not None:
        model["xy_center"] = np.asarray(xy_center).tolist()
    with open(json_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Saved Vandermonde model (degree={degree}) to {json_path}")
 
 
# ---------------- StagePointPicker ----------------
 
class StagePointPicker:
    """Frame-by-frame single-point picker (for pixel->stage calibration).
 
    Non-blocking, same usage pattern as ManualImageSelector: construct with
    plt.ion() active, let the user click through, read .points afterwards.
 
    - Left click: mark/overwrite the point for the current frame
    - Enter: next frame (closes the window after the last one)
    - Backspace: previous frame
    - R: reset current frame to NaN
    - N: mark current frame NaN and advance
    - Scroll: zoom; middle-drag: pan (if mpl_interactions is installed)
 
    Attributes
    ----------
    points : (n_frames, 2) array of (x, y) pixel coords, NaN where unmarked.
    """
 
    def __init__(self, imgs, cmap="gray"):
        self.imgs = np.asarray(imgs)
        self.n_frames = len(self.imgs)
        self.points = np.full((self.n_frames, 2), np.nan)
        self.i = 0
 
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.imgs[0], cmap=cmap)
        self.marker, = self.ax.plot(
            [], [], "r+", markersize=15, markeredgewidth=2
        )
        self.title = self.ax.set_title("")
 
        # pan/zoom without breaking left-click marking
        try:
            from mpl_interactions import zoom_factory, panhandler
            self._disconnect_zoom = zoom_factory(self.ax)
            self._panhandler = panhandler(self.fig, button=2)
        except ImportError:
            pass
 
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
 
    def _draw(self):
        img = self.imgs[self.i]
        self.im.set_data(img)
        self.im.set_clim(img.min(), img.max())
        if np.isnan(self.points[self.i]).any():
            self.marker.set_data([], [])
        else:
            self.marker.set_data(
                [self.points[self.i, 0]], [self.points[self.i, 1]]
            )
        n_done = int((~np.isnan(self.points).any(axis=1)).sum())
        self.title.set_text(
            f"Frame {self.i}/{self.n_frames - 1}  ({n_done} marked)\n"
            "Click point | Enter next | Backspace prev | R reset | N NaN"
        )
        self.fig.canvas.draw_idle()
 
    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.points[self.i] = [event.xdata, event.ydata]
        self._draw()
 
    def _advance(self):
        if self.i < self.n_frames - 1:
            self.i += 1
            self._draw()
        else:
            print("Finished picking.")
            plt.close(self.fig)
 
    def _on_key(self, event):
        if event.key == "enter":
            self._advance()
        elif event.key == "backspace":
            if self.i > 0:
                self.i -= 1
                self._draw()
        elif event.key == "r":
            self.points[self.i] = np.nan
            self._draw()
        elif event.key and event.key.lower() == "n":
            self.points[self.i] = np.nan
            self._advance()
