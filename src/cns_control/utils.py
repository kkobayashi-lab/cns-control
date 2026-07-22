import json
import time

import numpy as np
from skimage.draw import disk
from scipy.ndimage import center_of_mass, distance_transform_edt
from tqdm.auto import tqdm

from raman_mda_engine.aiming.autotracking import segment_single_img


# Values that mean "do not use an autofocus object / layer".
_NO_AUTOFOCUS = (None, "None", "none", "")
_GRID_PLACEHOLDER_LAYER = "Grid position placeholders"


def _remove_grid_placeholder(viewer):
    """Remove the grid placeholder layer if a previous Raman grid created it."""
    try:
        layer = viewer.layers[_GRID_PLACEHOLDER_LAYER]
    except (KeyError, TypeError):
        return
    viewer.layers.remove(layer)


def _set_grid_placeholder(viewer, n_positions):
    """Create or update a tiny hidden 6-D layer that exposes the position axis."""
    data = np.zeros((1, n_positions, 1, 1, 1, 1), dtype=np.uint8)
    try:
        layer = viewer.layers[_GRID_PLACEHOLDER_LAYER]
    except (KeyError, TypeError):
        return viewer.add_image(
            data,
            name=_GRID_PLACEHOLDER_LAYER,
            visible=False,
        )

    layer.data = data
    layer.visible = False
    return layer


def _grid_preview_sequence(sequence, preview_channel):
    """Return a one-channel preview sequence without modifying the Raman plan."""
    if sequence.channels:
        channel = sequence.channels[0].replace(config=preview_channel)
    else:
        from useq import Channel

        channel = Channel(config=preview_channel)
    return sequence.replace(channels=(channel,))


def _get_seq_from_napari(main_window):
    """Import the napari-MDA adapter only when a sequence is requested."""
    from raman_mda_engine.utils import get_seq_from_napari

    return get_seq_from_napari(main_window)


def _get_mda_widget_from_napari(main_window):
    """Import the napari-MDA widget adapter only when it is needed."""
    from raman_mda_engine.utils import get_mda_widget_from_napari

    return get_mda_widget_from_napari(main_window)


def _segment_single_img(image, *, scale):
    """Load the optional Cellpose stack only for automated segmentation."""
    from raman_mda_engine.aiming.autotracking import segment_single_img

    return segment_single_img(image, scale=scale)


def _is_no_autofocus(autofocus_object) -> bool:
    """True when the caller has asked for no autofocus object."""
    return autofocus_object in _NO_AUTOFOCUS


# ---------------------------------------------------------------------------
# Vandermonde pixel-offset -> stage-offset model
# ---------------------------------------------------------------------------

def load_vandermonde_model(json_path):
    """
    Load a fitted Vandermonde pixel-offset -> stage-offset model from JSON.

    Expected JSON structure (as produced by `fit_vandermonde` earlier):
        {
            "degree": <int>,
            "C": <list, shape (n_terms, 2)>
        }

    IMPORTANT: this model must have been fit on (pixel offset from image
    center) -> (stage offset) pairs, NOT on absolute pixel/stage positions.
    Fitting on absolute positions (with a bias/intercept term) and then
    applying it to a centered offset will give the wrong answer.

    Returns
    -------
    C : np.ndarray, shape (n_terms, 2)
    degree : int
    """
    with open(json_path) as f:
        model = json.load(f)
    C = np.array(model["C"])
    degree = int(model["degree"])
    return C, degree


def _vandermonde_design(src, degree):
    """
    Build a 2D polynomial design matrix from src=(x, y) up to `degree`,
    including cross terms (e.g. degree=2 -> [1, x, y, x^2, xy, y^2]).
    `src` can be a single (2,) point or an (n, 2) array.
    """
    src = np.atleast_2d(src)
    x, y = src[:, 0], src[:, 1]
    terms = []
    for total_deg in range(degree + 1):
        for i in range(total_deg + 1):
            j = total_deg - i
            terms.append((x**i) * (y**j))
    return np.stack(terms, axis=1)


def apply_vandermonde_model(src, C, degree):
    """
    Apply a fitted Vandermonde model to a single (x, y) pixel offset,
    returning the corresponding (dx, dy) stage displacement.
    """
    D = _vandermonde_design(src, degree)
    return (D @ C).ravel()


# ---------------------------------------------------------------------------
# napari layer / mask helpers
# ---------------------------------------------------------------------------

def add_mask_with_hole(
    viewer,
    image_size,
    circle_radius=200,
    color=(255, 0, 0),
    alpha=50,
    circle_center=None,
    small_circle_radius=10,
    small_circle_color=(0, 255, 0),
    small_circle_alpha=255
):
    """
    Adds a semi-transparent colored mask with a transparent circular hole and a solid
    colored small circle (dot) in the center to a napari viewer.

    Parameters:
    -----------
    viewer : napari.Viewer
        The napari viewer instance.
    image_size : tuple
        The (height, width) of the image.
    circle_radius : int
        Radius of the main transparent hole.
    color : tuple
        RGB color of the main mask.
    alpha : int
        Transparency of the main mask (0-255).
    circle_center : tuple or None
        (y, x) center of both holes. Defaults to image center.
    small_circle_radius : int
        Radius of the small central dot.
    small_circle_color : tuple
        RGB color of the small dot.
    small_circle_alpha : int
        Alpha for the small dot (default fully opaque).
    """

    image_size = tuple(image_size)
    rgba_image = np.zeros((image_size[0], image_size[1], 4), dtype=np.uint8)

    # Fill with the main mask color and alpha
    rgba_image[:, :, :3] = color
    rgba_image[:, :, 3] = alpha

    # Set center
    if circle_center is None:
        circle_center = (image_size[0] / 2, image_size[1] / 2)

    # Transparent main circular hole
    rr_main, cc_main = disk(circle_center, circle_radius, shape=rgba_image.shape[:2])
    rgba_image[rr_main, cc_main, :] = 0

    # Solid small colored circle in the center
    rr_small, cc_small = disk(circle_center, small_circle_radius, shape=rgba_image.shape[:2])
    rgba_image[rr_small, cc_small, :3] = small_circle_color
    rgba_image[rr_small, cc_small, 3] = small_circle_alpha

    # Add to napari
    viewer.add_image(rgba_image, rgb=True)


def create_point_sources(viewer,
                         point_transformer,
                         broadcast_dims=(0, 2, 3),
                         ndim=6,
                         size=35,
                         names=['cells', 'autofocus'],
                         colors=['#aa0000ff', 'springgreen']):
    """
    Creates and adds multiple point source layers to a Napari viewer.

    Parameters:
    -----------
    viewer : napari.Viewer
        The Napari viewer where the point sources will be added.
    point_transformer : callable
        A transformation function applied to the point coordinates.
    broadcast_dims : tuple, optional
        Dimensions along which the points should be broadcasted (default: (0, 2, 3)).
    ndim : int, optional
        Number of dimensions for the point sources (default: 6).
    size : float, optional
        Size of the points in pixels (default: 35).
    names : list of str, optional
        Names of the point source layers (default: ['cells', 'autofocus']).
    colors : list of str, optional
        List of colors for the point sources, specified in hex format.

    Returns:
    --------
    list of PointsLayerSource
        A list of `PointsLayerSource` objects, each corresponding to a created point layer.
    """
    # These imports initialize napari's layer stack. Keep them out of module
    # scope so lightweight helpers (for example ``filter_mean``) remain usable
    # without importing napari, the MDA engine, Cellpose, or PyTorch.
    from napari_broadcastable_points import BroadcastablePoints
    from raman_mda_engine.aiming import PointsLayerSource

    sources = []
    for name, color in zip(names, colors):
        points = BroadcastablePoints(
                    None,
                    #               t, c, z
                    broadcast_dims=broadcast_dims,
                    ndim=ndim,
                    name=name,
                    size=size,
                    face_color=color,
                    border_color="#5500ffff",
                )
        viewer.add_layer(points)
        sources.append(PointsLayerSource(points, name=name, transformer=point_transformer))
    return sources


def filter_mean(spec, f=2):
    mean_spec = np.mean(spec, axis=0)
    std_spec = np.std(spec, axis=0)

    # Create a mask for values within 3 standard deviations
    mask = (spec >= (mean_spec - f * std_spec)) & (spec <= (mean_spec + f * std_spec))

    # Compute the mean while ignoring values outside 3 std
    filtered_mean_spec = np.sum(spec * mask, axis=0) / np.sum(mask, axis=0)
    return filtered_mean_spec


def set_up_new_seq(main_window, point_transformer, engine, seq=None, total_exposure=1000, batch=False, z_plan='all'):
    """
    Configures a new Raman sequence with updated metadata and exposure settings.

    Parameters:
    -----------
    main_window : object
        The main Napari window instance used to retrieve the sequence.
    point_transformer : object
        An object that provides the `multiplier` attribute for adjusting exposure time.
    engine : object
        The engine controlling Raman measurements, which stores default exposure settings.
    total_exposure : int, optional (default=1000)
        The total exposure time in milliseconds (ms) for Raman acquisition.
    batch : bool, optional (default=False)
        If False, the exposure time is divided by `point_transformer.multiplier`.
        If True, `total_exposure` is used directly without modification.
    z_plan : str, optional (default='all')
        Determines which Z positions to acquire Raman spectra from:
        - `'all'`: Acquire Raman data at all Z positions.
        - `'middle'`: Acquire Raman data only at the middle Z position.
        - Any other value raises a `ValueError`.

    Returns:
    --------
    new_seq : object
        A modified sequence object with updated metadata specifying the Raman acquisition plan.
    """

    if not batch:
        engine.default_rm_exposure = total_exposure / point_transformer.multiplier
    else:
        engine.default_rm_exposure = total_exposure

    if engine.default_rm_exposure < 73.8:
        raise ValueError('Minimal exposure time per Raman collection is 0.0738s')

    if seq is None:
        seq = _get_seq_from_napari(main_window)
    new_meta = dict(seq.metadata)
    num_z = seq.z_plan.num_positions()
    # add in raman metadata to do raman
    if z_plan == 'all':
        new_meta["raman"] = {"z": np.arange(num_z).tolist()} # do all Zs
    elif z_plan == 'middle':
        new_meta["raman"] = {"z": [num_z//2]} # only do the middle Z
    else:
        raise ValueError('Please choose a valid z_plan: middle or all')
    new_seq = seq.replace(metadata=new_meta)
    return new_seq


def find_clear_center_point(mask, threshold=20):
    background = mask == 0
    dist_map = distance_transform_edt(background)
    center = np.array(mask.shape) / 2
    valid_points = np.argwhere(dist_map >= threshold)

    if len(valid_points) == 0:
        raise ValueError("No point found that meets the threshold distance.")

    dists_to_center = np.linalg.norm(valid_points - center, axis=1)
    best_idx = np.argmin(dists_to_center)

    return np.array(valid_points[best_idx])


def get_n_most_centered_coms(
    label_mask,
    N=10,
    center=None,
    radius=250,
    autofocus_object='glass',
    bkd_threshold=50
):
    """
    Returns up to N center-of-mass points closest to a given center,
    within a distance threshold.

    When `autofocus_object` is one of the focus targets, a clear-background
    point is inserted at index 0 (used for autofocus). When it is None / "None"
    (or 'cell'), NO autofocus point is prepended -- every returned point is a
    cell center-of-mass.

    Parameters:
    -----------
    label_mask : 2D array
        Labeled mask image.
    N : int
        Number of points to return.
    center : tuple or None
        (y, x) center to compare distances. Defaults to [672, 512].
    radius : float
        Max distance from center to include a point. Pass np.inf for no limit.
    autofocus_object : str or None
        If a focus target ('glass', 'quartz', 'laser', 'software'), adds a clear
        point at index 0. If None / "None", no autofocus point is added.
    bkd_threshold : int
        Passed to find_clear_center_point.

    Returns:
    --------
    np.ndarray of shape (<=N, 2): list of center-of-mass coordinates
    """
    labels = np.unique(label_mask)
    labels = labels[labels != 0]  # ignore background

    if center is None:
        center = np.array([672, 512])
    else:
        center = np.array(center)

    coms = []
    dists = []

    for label in labels:
        com = center_of_mass(np.ones_like(label_mask), labels=label_mask, index=label)
        com = np.array(com)
        dist = np.linalg.norm(com - center)
        if dist <= radius:
            coms.append(com)
            dists.append(dist)

    # Sort by distance to center
    sorted_coms = [com for _, com in sorted(zip(dists, coms), key=lambda x: x[0])]

    # Optionally insert autofocus point (skipped entirely when no autofocus).
    if autofocus_object in ['glass', 'quartz', 'laser', 'software']:
        clear_com = find_clear_center_point(label_mask, threshold=bkd_threshold)
        sorted_coms.insert(0, clear_com)

    return np.array(sorted_coms[:N])


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def automated_point_selections(
    core, viewer, main_window, point_transformer, N,
    center=None, radius=250, autofocus_object='glass', bkd_thres=50, batch=True,
    center_cell=False, vandermonde_model_path=None, cellpose_model='cyto2'
):
    """
    Parameters
    ----------
    center_cell : bool, optional (default=False)
        If False, behaves exactly as before: for each original stage
        position, up to N cells are detected at their raw pixel location
        (within `radius` of `center`), and the position is repeated once
        per detected cell.

        If True, up to N cells are still detected per original stage
        position -- but the search is UNBOUNDED (the `radius` argument is
        ignored in this mode, since we want every cell in the FOV, not just
        ones near center). Instead of repeating the SAME stage position for
        each detected cell, a NEW, independently-corrected stage position is
        generated per cell using a pre-fit Vandermonde pixel-offset ->
        stage-offset model, such that each cell ends up exactly under the
        image center. So N=100 means: try to find up to 100 cells in that
        one original FOV, then split it into up to 100 new stage positions,
        each one centered on a different cell.

        The 'cells' point recorded for every new position is therefore
        always the image center (that's where the cell will sit once the
        stage moves there). If `autofocus_object` is set, the SAME raw
        autofocus point (detected once for that original FOV, at its
        original, un-shifted pixel location) is recorded against every new
        position split from it -- autofocus is a property of the FOV /
        reference surface, not of an individual cell.

    vandermonde_model_path : str or None
        Path to a JSON file with the fitted model (see
        `load_vandermonde_model`). Required when `center_cell=True`. Must be
        fit on (pixel offset from image center) -> (stage offset) pairs.

    Returns
    -------
    sources, autofocus_p, new_seq
        Same 3-tuple contract as before. `autofocus_p` is simply
        `np.arange(len(new_positions))` when `center_cell=True`, since every
        new position is already its own independent FOV (no repeats/groups
        to index into).
    """
    no_autofocus = _is_no_autofocus(autofocus_object)

    if center_cell:
        if vandermonde_model_path is None:
            raise ValueError("vandermonde_model_path must be provided when center_cell=True")
        C, degree = load_vandermonde_model(vandermonde_model_path)

    seq = _get_seq_from_napari(main_window)
    images = []
    masks = []
    points = []          # per ORIGINAL position: raw detected point(s), pixel coords (y, x)
    new_positions = []   # only populated when center_cell=True -- one entry per detected cell

    for i in tqdm(range(len(seq.stage_positions))):
        orig_pos = seq.stage_positions[i]
        core.setXYPosition(orig_pos.x, orig_pos.y)
        core.waitForSystem()
        time.sleep(5)
        core.snapImage()
        core.waitForSystem()
        image = core.getImage()
        core.waitForSystem()
        time.sleep(1)
        images.append(image)
        mask = segment_single_img(image, scale=1, cellpose_model=cellpose_model, circle_center=center, circle_radius = radius)
        masks.append(mask)

        if center_cell:
            # try to find up to N cells (+1 autofocus point if requested);
            # radius is unbounded so we don't miss cells far from center.
            n_needed = N if no_autofocus else N + 1
            found = get_n_most_centered_coms(
                mask, N=n_needed, center=center, radius=np.inf,
                autofocus_object=autofocus_object, bkd_threshold=bkd_thres,
            )
            points.append(found)

            if found.size == 0:
                continue

            cell_slice = slice(0, None) if no_autofocus else slice(1, None)
            cells_yx = found[cell_slice, :]

            img_center_yx = (
                np.array(image.shape[:2]) / 2.0 if center is None else np.array(center)
            )

            # one new, independently-corrected stage position PER detected cell
            for cell_yx in cells_yx:
                offset_yx = cell_yx - img_center_yx
                offset_xy = np.array([offset_yx[1], offset_yx[0]])
                stage_dx, stage_dy = apply_vandermonde_model(offset_xy, C, degree)

                print(f"[debug] cell_yx={cell_yx}, img_center_yx={img_center_yx}, "
                    f"offset_yx={offset_yx}, offset_xy={offset_xy}, "
                    f"stage_dx={stage_dx:.3f}, stage_dy={stage_dy:.3f}")
                new_positions.append(
                    orig_pos.replace(
                        x=float(orig_pos.x - stage_dx),
                        y=float(orig_pos.y - stage_dy),
                    )
                )
        else:
            points.append(
                get_n_most_centered_coms(
                    mask, N=N, center=center, radius=radius,
                    autofocus_object=autofocus_object, bkd_threshold=bkd_thres,
                )
            )

    images = np.array(images)
    masks = np.array(masks)

    # ------------------------- center_cell = True -------------------------
    if center_cell:
        new_seq = seq.replace(stage_positions=new_positions)

        if no_autofocus:
            sources = create_point_sources(
                viewer, point_transformer, size=15,
                names=['cells'], colors=['#aa0000ff'],
            )
        else:
            sources = create_point_sources(viewer, point_transformer, size=15)

        core.run_mda(new_seq)

        # image center is the same for every FOV (same camera/detector)
        img_center_yx = np.array(images.shape[1:3]) / 2.0 if center is None else np.array(center)

        p = 0
        for found in points:
            if found.size == 0:
                continue
            cell_slice = slice(0, None) if no_autofocus else slice(1, None)
            n_cells_here = found[cell_slice, :].shape[0]
            autofocus_yx = None if no_autofocus else found[0]

            for _ in range(n_cells_here):
                if not no_autofocus:
                    sources[1]._points.add([0, p, 0, 0, autofocus_yx[0], autofocus_yx[1]])
                # 2 repeated points at center (DAQ needs >= 2 samples per channel)
                sources[0]._points.add([0, p, 0, 0, img_center_yx[0], img_center_yx[1]])
                if point_transformer.multiplier <= 1:
                    sources[0]._points.add([0, p, 0, 0, img_center_yx[0], img_center_yx[1]])
                p += 1

        return sources, np.arange(len(new_positions)), new_seq

    # ------------------------- center_cell = False (original) --------------
    if no_autofocus:
        cell_slice = slice(0, None)
        repeats = [len(point) for point in points]
    else:
        cell_slice = slice(1, None)
        repeats = [len(point) - 1 for point in points]

    repeated_positions = [pos for pos, n in zip(seq.stage_positions, repeats) for _ in range(n)]
    new_seq = seq.replace(stage_positions=repeated_positions)

    if no_autofocus:
        sources = create_point_sources(
            viewer, point_transformer, size=15,
            names=['cells'], colors=['#aa0000ff'],
        )
    else:
        sources = create_point_sources(viewer, point_transformer, size=15)

    if batch:
        core.run_mda(new_seq)
        expanded_indices = [i for i, n in enumerate(repeats) for _ in range(n)]
        # all cell points across every FOV, in expanded order
        all_cells = np.vstack([arr[cell_slice, :] for arr in points])
        for p in range(len(repeated_positions)):
            if not no_autofocus:
                sources[1]._points.add(
                    [0, p, 0, 0,
                     points[expanded_indices[p]][0, 0],
                     points[expanded_indices[p]][0, 1]]
                )
            pt = all_cells[p]
            sources[0]._points.add([0, p, 0, 0, pt[0], pt[1]])
        return sources, np.cumsum([0] + repeats[:-1]), new_seq
    else:
        core.run_mda(seq)
        for p in range(len(seq.stage_positions)):
            if not no_autofocus:
                sources[1]._points.add([0, p, 0, 0, points[p][0, 0], points[p][0, 1]])
            for pt in points[p][cell_slice, :]:
                sources[0]._points.add([0, p, 0, 0, pt[0], pt[1]])
        return sources, np.arange(len(seq.stage_positions)), seq


def manual_point_selections(core, viewer, main_window, point_transformer, N,
                            autofocus_object='glass', batch=True):
    """
    Set up point-source layers for MANUAL selection (click points by hand).

    Mirrors `automated_point_selections` but does NO imaging, segmentation or
    point placement -- it only creates the (empty) source layers and returns
    the same (sources, autofocus_p, new_seq) contract the MDA expects. The user
    clicks points into the layers afterwards.

    N cells-per-FOV is fixed up front (exactly like automated selection):
    - batch=True : each stage position is repeated N times to build new_seq,
      and the user must click N cell points per FOV to match.
    - batch=False: new_seq is the current sequence unchanged; the user clicks
      however many points they like per FOV.

    Layers:
    - autofocus_object None/"None" -> one layer  ('cells').
    - real focus target           -> two layers ('cells','autofocus').

    Returns
    -------
    sources : list[PointsLayerSource]
    autofocus_p : np.ndarray
        batch=True  -> first flat index of each FOV's repeated block.
        batch=False -> every stage-position index.
    new_seq : MDASequence
        batch=True  -> positions repeated N times.
        batch=False -> the current napari sequence, unchanged.
    """
    no_autofocus = _is_no_autofocus(autofocus_object)

    seq = _get_seq_from_napari(main_window)

    # Create the empty source layers (one or two).
    if no_autofocus:
        sources = create_point_sources(
            viewer, point_transformer, size=15,
            names=['cells'], colors=['#aa0000ff'],
        )
    else:
        sources = create_point_sources(viewer, point_transformer, size=15)

    if batch:
        # Repeat every position N times, exactly like the automated batch path.
        repeats = [N for _ in range(len(seq.stage_positions))]
        repeated_positions = [
            pos for pos in seq.stage_positions for _ in range(N)
        ]
        new_seq = seq.replace(stage_positions=repeated_positions)

        # Establish the broadcast/position dimensions on the empty layers so
        # per-position clicks map correctly (automated does this too).
        core.run_mda(new_seq)

        return sources, np.cumsum([0] + repeats[:-1]), new_seq
    else:
        core.run_mda(seq)
        return sources, np.arange(len(seq.stage_positions)), seq
    
def center_manual_selections(core, viewer, main_window, point_transformer,
                             sources, vandermonde_model_path,
                             autofocus_object='glass', center=None):
    """
    Convert hand-clicked cell points into centered stage positions.

    Workflow: run manual_point_selections (batch=False), click your cell
    points (and one autofocus point per FOV if autofocus is enabled), then
    call this. Each clicked cell becomes its OWN new stage position, shifted
    with the Vandermonde model so the cell lands at the image center --
    exactly like automated center_cell mode, but with hand-picked cells.

    NOTE: only valid after a NON-batch manual selection (batch repeats
    positions, which breaks the p -> stage-position mapping used here).

    Returns
    -------
    sources, autofocus_p, new_seq : same 3-tuple contract as the other
        selection functions. autofocus_p = np.arange(n_new).
    """
    no_autofocus = _is_no_autofocus(autofocus_object)
    C, degree = load_vandermonde_model(vandermonde_model_path)
    seq = _get_seq_from_napari(main_window)
    img_y = int(core.getImageHeight())
    img_x = int(core.getImageWidth())
    img_center_yx = (
        np.array([img_y / 2.0, img_x / 2.0]) if center is None
        else np.array(center)
    )

    cell_data = np.asarray(sources[0]._points.data)
    if cell_data.size == 0:
        raise ValueError("No clicked cell points found -- click points first.")
    af_data = None
    if not no_autofocus:
        if len(sources) < 2 or len(np.asarray(sources[1]._points.data)) == 0:
            raise ValueError(
                "Autofocus is enabled but no autofocus points were clicked."
            )
        af_data = np.asarray(sources[1]._points.data)

    new_positions = []
    af_points = []   # raw autofocus pixel point carried to each new position
    for p in sorted(set(cell_data[:, 1].astype(int))):
        orig_pos = seq.stage_positions[p]
        cells_yx = cell_data[cell_data[:, 1].astype(int) == p][:, -2:]
        if not no_autofocus:
            af_here = af_data[af_data[:, 1].astype(int) == p]
            if len(af_here) == 0:
                raise ValueError(f"No autofocus point clicked at position {p}.")
            af_yx = af_here[0, -2:]
        for cell_yx in cells_yx:
            offset_yx = cell_yx - img_center_yx
            offset_xy = np.array([offset_yx[1], offset_yx[0]])
            stage_dx, stage_dy = apply_vandermonde_model(offset_xy, C, degree)
            print(f"[manual center] p={p}, cell_yx={cell_yx}, "
                  f"stage_dx={stage_dx:.3f}, stage_dy={stage_dy:.3f}")
            new_positions.append(orig_pos.replace(
                x=float(orig_pos.x - stage_dx),
                y=float(orig_pos.y - stage_dy),
            ))
            if not no_autofocus:
                af_points.append(af_yx)

    new_seq = seq.replace(stage_positions=new_positions)
    if no_autofocus:
        new_sources = create_point_sources(
            viewer, point_transformer, size=15,
            names=['cells'], colors=['#aa0000ff'],
        )
    else:
        new_sources = create_point_sources(viewer, point_transformer, size=15)
    core.run_mda(new_seq)
    for p in range(len(new_positions)):
        if not no_autofocus:
            new_sources[1]._points.add(
                [0, p, 0, 0, af_points[p][0], af_points[p][1]]
            )
        new_sources[0]._points.add(
            [0, p, 0, 0, img_center_yx[0], img_center_yx[1]]
        )
        # 2 repeated points at center when the transformer doesn't multiply
        # (DAQ needs >= 2 samples per channel)
        if point_transformer.multiplier <= 1:
            new_sources[0]._points.add(
                [0, p, 0, 0, img_center_yx[0], img_center_yx[1]]
            )
    return new_sources, np.arange(len(new_positions)), new_seq

def grid_point_selections(core, viewer, main_window, point_transformer,
                          fov_x, fov_y,
                          x_range, y_range, x_step, y_step,
                          repeats=2, preview_channel="BF",
                          autofocus_object='None'):
    """
    Build a grid of stage positions centered on the CURRENT stage XY, each
    carrying `repeats` copies of the SAME single fixed point (fov_x, fov_y).
    No autofocus, non-batch.

    repeats : int >= 2
        How many identical points to place at (fov_x, fov_y) per position. Must
        be >= 2 (the DAQ needs at least 2 samples per channel).

    ``preview_channel`` selects the channel used to image the grid before Raman.
    Passing ``None`` skips hardware acquisition and creates a hidden placeholder
    layer that establishes napari's position axis instead.

    Writes the freshly-built grid of positions into the napari MDA widget and
    returns the (sources, autofocus_p, new_seq) contract the MDA expects. The
    returned sequence always keeps its original Raman acquisition channels.
    autofocus_object : str or None
        If a real focus target ('glass', 'quartz', 'laser', 'software',
        'cell'), an 'autofocus' layer is also created with one point per
        position at the same fixed (fov_y, fov_x). If None/"None", only the
        'cells' layer is created (original behavior).
    """
    repeats = int(repeats)
    if repeats < 2:
        raise ValueError("repeats must be an integer >= 2")

    no_autofocus = _is_no_autofocus(autofocus_object)
    seq = _get_seq_from_napari(main_window)

    # Origin = current stage position; grid spans origin +/- range.
    origin_x, origin_y = core.getXYPosition()
    xs = np.arange(origin_x - x_range, origin_x + x_range + x_step / 2.0, x_step)
    ys = np.arange(origin_y - y_range, origin_y + y_range + y_step / 2.0, y_step)
    template = seq.stage_positions[0]
    grid_positions = []
    for gx in xs:
        for gy in ys:
            grid_positions.append(template.replace(x=float(gx), y=float(gy)))
    new_seq = seq.replace(stage_positions=grid_positions)

    # Write the generated positions into the napari MDA widget so the later
    # Raman MDA uses the same grid regardless of the preview mode.
    try:
        mda_settings = _get_mda_widget_from_napari(main_window)
        if hasattr(mda_settings, "setValue"):
            mda_settings.setValue(new_seq)
            new_seq = _get_seq_from_napari(main_window)  # read back the truth
        else:
            print("[grid setup] MDA widget has no setValue -- grid may not show")
    except Exception as e:
        print(f"[grid setup] couldn't write positions to MDA widget: {e}")
    if no_autofocus:
        sources = create_point_sources(
            viewer, point_transformer, size=15,
            names=['cells'], colors=['#aa0000ff'],
        )
    else:
        sources = create_point_sources(viewer, point_transformer, size=15)

    # Establish napari's position dimension without changing the sequence that
    # the later Raman MDA consumes. A real channel gets a temporary one-channel
    # preview sequence; Raman mode gets a tiny hidden placeholder layer.
    if preview_channel is None:
        _set_grid_placeholder(viewer, len(new_seq.stage_positions))
        print("[grid setup] grid ready using Raman placeholders")
    else:
        _remove_grid_placeholder(viewer)
        preview_seq = _grid_preview_sequence(new_seq, preview_channel)
        core.run_mda(preview_seq)
        print(f"[grid setup] grid ready after {preview_channel} preview")

    # Place `repeats` identical points at (fov_x, fov_y) for every position.
    # Row order (.., y, x) matches the other selection functions.
    for p in range(len(new_seq.stage_positions)):
        for _ in range(repeats):
            sources[0]._points.add([0, p, 0, 0, fov_y, fov_x])
    if not no_autofocus:
        n_pos = len(new_seq.stage_positions)
        # one autofocus point per position, at the same fixed FOV pixel
        af_pts = np.array(
            [[0, p, 0, 0, fov_y, fov_x] for p in range(n_pos)],
            dtype=float,
        )
        sources[1]._points.add(af_pts)
    return sources, np.arange(len(new_seq.stage_positions)), new_seq
def unload(core, N=20):
    n = 0.1  # starting sleep time
    for attempt in range(N):
        if attempt == N-1:
            print('reach reloading maxiter')
        try:
            time.sleep(n)
            try:
                core.events.channelGroupChanged.disconnect()
            except Exception:
                None
            try:
                core.events.configGroupChanged.disconnect()
            except Exception:
                None
            try:
                core.events.propertyChanged.disconnect()
            except Exception:
                None
            try:
                core.events.systemConfigurationLoaded.disconnect()
            except Exception:
                None
            try:
                core.events.configSet.disconnect()
            except Exception:
                None

            core.unloadAllDevices()
            core.waitForSystem()
            return  # success!
        except Exception as e:
            n += 1  # increase wait time and retry