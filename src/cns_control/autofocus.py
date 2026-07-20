import numpy as np
from pymmcore_plus import CMMCorePlus
import time
from tqdm.auto import tqdm
# from raman_control.andor import AndorSpectraCollector
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def gaussian(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def rescale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def remove_outlier(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    filtered_data = np.where(data < mean + 4 * std, data, 0)
    
# for a single point
def autofocus_w_bkd(core, daq, collector, volts, search_range=20, search_pts=15, exposure=1000):
    focusZ = core.getPosition()
    core.stopSequenceAcquisition()

    daq.galvo.stop()
    core.setConfig("Channel", "RM")
    # core.setShutterOpen("Fluoshutter", True)
    coarse_Z = np.linspace(-search_range, search_range, search_pts)
    coarse_raman = []
    all_raman = []
    for z in tqdm(coarse_Z):
        core.setPosition(focusZ+z)
        core.waitForSystem()
        spec = collector.collect_spectra_pts(np.array(volts), exposure)
        coarse_raman.append(np.mean(spec[:, :], axis=0))
        all_raman.append(spec)

    coarse_raman = np.asarray(coarse_raman)
    all_raman = np.asarray(all_raman)

    # core.setShutterOpen("Fluoshutter", False)
    return focusZ, coarse_raman, all_raman

# def autofocus_w_bkd(core, daq, collector, volts, start=1004, end=1030, search_range=20, search_pts=15, exposure=1000, plot=True):
#     focusZ = core.getPosition()
#     core.stopSequenceAcquisition()
#     # pts = single_pt / [1024, 1344]
#     # volts = transformer.BF_to_volts((pt.reshape(1, -1)*[1344, 1024])/[1024, 1344], max_volts=max_volt)
#     # print(volts.shape)
#     # print(volts.shape[0]*.5/60)
#     daq.galvo.stop()
#     core.setConfig("Channel", "RM")
#     core.setShutterOpen("Fluoshutter", True)
#     coarse_Z = np.linspace(-search_range, search_range, search_pts)
#     coarse_raman = []
#     all_raman = []
#     for z in tqdm(coarse_Z):
#         core.setPosition(focusZ+z)
#         core.waitForSystem()
#         # time.sleep(0.25)
#         spec = collector.collect_spectra_pts(np.array(volts), exposure)
#         coarse_raman.append(np.mean(spec[:, :], axis=0))
#         all_raman.append(spec)

#     coarse_raman = np.asarray(coarse_raman)
#     all_raman = np.asarray(all_raman)
#     # coarse_intensities = coarse_raman[:, start:end].sum(axis=1)
#     # coarse_idx = np.argmax(coarse_intensities)

#     cell_raman = rescale(coarse_raman[:, start:end].sum(axis=1) / np.median(coarse_raman))
#     popt, _ = curve_fit(gaussian, coarse_Z, cell_raman,
#                        p0 = [1,0,2],
#                        method='trf',
#                        maxfev=1e4)

#     max_laser_offset = popt[1]
#     if plot:
#         plt.figure()
#         plt.plot(coarse_Z, cell_raman, "o-")
#         # plt.plot(coarse_Z, coarse_intensities, "o-")
#         plt.plot(np.linspace(-search_range, search_range, 100), gaussian(np.linspace(-search_range, search_range, 100), *popt))
#         plt.axvline(max_laser_offset, c='k', linestyle='dashed')

#     if np.abs(max_laser_offset) >= search_range:
#         max_laser_offset = 0

#     core.setShutterOpen("Fluoshutter", False)
#     return focusZ, max_laser_offset+focusZ, coarse_raman, all_raman

def try_set_ZPosition(core, z, N=20):
    n = 0.1  # starting sleep time
    for attempt in range(N):
        try:
            time.sleep(n)
            core.setZPosition(z)
            core.waitForSystem()
            return  # success!
        except RuntimeError:
            n += 1  # increase wait time and retry

def autofocus_w_raman(core, daq, collector, transformer, pt, start=500, end=2000, search_range=10, search_pts=10, max_volt=1.5, plot=True):
    focusZ = core.getPosition()
    core.stopSequenceAcquisition()
    # pts = single_pt / [1024, 1344]
    volts = transformer.BF_to_volts((pt.reshape(1, -1)*[1344, 1024])/[1024, 1344], max_volts=max_volt)
    # volts = np.array([[0,0], [0,0]])
    # print(volts.shape)
    # print(volts.shape[0]*.5/60)
    daq.galvo.stop()
    core.setConfig("Channel", "RM")
    core.setShutterOpen("Fluoshutter", True)
    core.waitForSystem()
    coarse_Z = np.linspace(-search_range, search_range, search_pts)
    coarse_raman = []
    for z in tqdm(coarse_Z):
        try_set_ZPosition(core, focusZ+z)
        # n = 0
        # try:            
        #     time.sleep(n)
        #     n += 1
        #     core.setPosition(focusZ+z)
        #     core.waitForSystem()
        # except RuntimeError:
        #     try:
        #         time.sleep(n)
        #         n += 1
        #         core.setPosition(focusZ+z)
        #         core.waitForSystem()
        #     except RuntimeError:
        #         try:
        #             time.sleep(n)
        #             n += 1
        #             core.setPosition(focusZ+z)
        #             core.waitForSystem()
        #         except RuntimeError:
        #             try:
        #                 time.sleep(n)
        #                 n += 1               
        #                 core.setPosition(focusZ+z)
        #                 core.waitForSystem()
        #             except RuntimeError:
        #                 try:
        #                     time.sleep(n)
        #                     n += 1               
        #                     core.setPosition(focusZ+z)
        #                     core.waitForSystem()
        #                 except RuntimeError:
        #                     time.sleep(n)
        #                     n += 1               
        #                     core.setPosition(focusZ+z)
        #                     core.waitForSystem()

        # core.setPosition(focusZ+z)
        # core.waitForSystem()
        # time.sleep(0.25)
        spec = collector.collect_spectra_pts(np.array([volts[0], volts[0]]), 10)
        coarse_raman.append(np.mean(spec[:, :], axis=0))

    coarse_raman = np.asarray(coarse_raman)
    # coarse_intensities = coarse_raman[:, start:end].sum(axis=1)
    # coarse_idx = np.argmax(coarse_intensities)

    cell_raman = rescale(coarse_raman[:, start:end].sum(axis=1) / np.median(coarse_raman))

    # Interpolation (cubic gives a smooth curve)
    interp_func = interp1d(coarse_Z, cell_raman, kind='cubic')
    
    # Finer x-values for interpolation
    x_fine = np.linspace(coarse_Z.min(), coarse_Z.max(), 1000)
    y_fine = interp_func(x_fine)
    
    # Find the maximum
    max_index = np.argmax(y_fine)
    x_peak = x_fine[max_index]
    y_peak = y_fine[max_index]
    max_laser_offset = x_peak

    if plot:
        plt.figure()
        plt.scatter(coarse_Z, cell_raman)
        plt.plot(x_fine, y_fine, '-', label='Interpolated curve')
        plt.plot(x_peak, y_peak, 'r*', markersize=10, label='Peak')
        # plt.plot(coarse_Z, coarse_intensities, "o-")
        # plt.plot(np.linspace(-search_range, search_range, 100), gaussian(np.linspace(-search_range, search_range, 100), *popt))
        # plt.axvline(max_laser_offset, c='k', linestyle='dashed')

    if np.abs(max_laser_offset) >= search_range:
        max_laser_offset = 0

    core.setShutterOpen("Fluoshutter", False)
    core.watiForSystem(0)
    return focusZ, max_laser_offset+focusZ, coarse_raman
