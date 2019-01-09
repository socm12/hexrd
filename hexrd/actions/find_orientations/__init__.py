"""find_orientations command"""
from __future__ import print_function, division, absolute_import

import os

import numpy as np
import timeit

from hexrd import constants as cnst
from hexrd import instrument
from hexrd import matrixutil as mutil
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd.xrd import indexer
from hexrd.xrd import transforms_CAPI as xfcapi
from .utils import get_eta_ome, generate_orientation_fibers, run_cluster
from .utils import analysis_id

def find_orientations(cfg, hkls=None, clean=False, profile=False, nsim=100):
    print('ready to run find_orientations')
    # %%
    # =============================================================================
    # SEARCH SPACE GENERATION
    # =============================================================================

    hedm = cfg.instrument.hedm
    plane_data = cfg.material.plane_data

    ncpus = cfg.multiprocessing

    # for indexing
    active_hkls = cfg.find_orientations.orientation_maps.active_hkls 
    fiber_ndiv = cfg.find_orientations.seed_search.fiber_ndiv
    fiber_seeds = cfg.find_orientations.seed_search.hkl_seeds
    on_map_threshold = cfg.find_orientations.threshold

    # for clustering
    cl_radius = cfg.find_orientations.clustering.radius
    min_compl = cfg.find_orientations.clustering.completeness
    compl_thresh = cfg.find_orientations.clustering.completeness

    eta_ome = get_eta_ome(cfg, clean=clean)

    print("INFO:\tgenerating search quaternion list using %d processes" % ncpus)
    start = timeit.default_timer()
    qfib = generate_orientation_fibers(
        eta_ome, hedm.chi, on_map_threshold,
        fiber_seeds, fiber_ndiv,
        ncpus=ncpus
    )
    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
    print("INFO: will test %d quaternions using %d processes"
          % (qfib.shape[1], ncpus))

    # %%
    # =============================================================================
    # ORIENTATION SCORING
    # =============================================================================

    scoredq_filename = 'scored_orientations_' + analysis_id(cfg) + '.npz'

    print("INFO:\tusing map search with paintGrid on %d processes"
          % ncpus)
    start = timeit.default_timer()

    completeness = indexer.paintGrid(
        qfib,
        eta_ome,
        etaRange=np.radians(cfg.find_orientations.eta.range),
        omeTol=np.radians(cfg.find_orientations.omega.tolerance),
        etaTol=np.radians(cfg.find_orientations.eta.tolerance),
        omePeriod=np.radians(cfg.find_orientations.omega.period),
        threshold=on_map_threshold,
        doMultiProc=ncpus > 1,
        nCPUs=ncpus
        )
    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
    completeness = np.array(completeness)

    # export scored orientations
    np.savez_compressed(scoredq_filename,
                        quaternions=qfib,
                        completeness=completeness)
    print("INFO:\tsaved scored orientations to file: '%s'"
          % (scoredq_filename))
    
    # %%
    # =============================================================================
    # CLUSTERING AND GRAINS OUTPUT
    # =============================================================================

    if not os.path.exists(cfg.analysis_dir):
        os.makedirs(cfg.analysis_dir)
    qbar_filename = 'accepted_orientations_' + analysis_id(cfg) + '.dat'

    print("INFO:\trunning clustering using '%s'"
          % cfg.find_orientations.clustering.algorithm
    )
    start = timeit.default_timer()

    # Simulate N random grains to get neighborhood size
    print("INFO:\trunning %d simulations to determine neighborhood size"
          % nsim
    )
    seed_hkl_ids = [
        plane_data.hklDataList[active_hkls[i]]['hklID'] for i in fiber_seeds
    ]
    
    # need ome_ranges from imageseries
    # CAVEAT: assumes that all imageseries have same omega ranges!!!
    oims = OmegaImageSeries(cfg.image_series.itervalues().next())
    ome_ranges = [
        (np.radians([i['ostart'], i['ostop']])) for i in oims.omegawedges.wedges
    ]
    
    if seed_hkl_ids is not None:
        rand_q = mutil.unitVector(np.random.randn(4, nsim))
        rand_e = np.tile(2.*np.arccos(rand_q[0, :]), (3, 1)) \
          * mutil.unitVector(rand_q[1:, :])
        refl_per_grain = np.zeros(nsim)
        num_seed_refls = np.zeros(nsim)
        grain_param_list = np.vstack([rand_e, 
                                      np.zeros((3, nsim)),
                                      np.tile(cnst.identity_6x1, (nsim, 1)).T]).T
        sim_results = hedm.simulate_rotation_series(
                plane_data, grain_param_list, 
                eta_ranges=np.radians(cfg.find_orientations.eta.range),
                ome_ranges=ome_ranges,
                ome_period=np.radians(cfg.find_orientations.omega.period)
        )
        
        refl_per_grain = np.zeros(nsim)
        seed_refl_per_grain = np.zeros(nsim)
        for sim_result in sim_results.itervalues():
            for i, refl_ids in enumerate(sim_result[0]):
                refl_per_grain[i] += len(refl_ids)
                seed_refl_per_grain[i] += np.sum([sum(refl_ids == hkl_id) for hkl_id in seed_hkl_ids])
    
        min_samples = max(
            int(np.floor(0.5*cfg.find_orientations.clustering.completeness*min(seed_refl_per_grain))),
            2
            )
        mean_rpg = int(np.round(np.average(refl_per_grain)))
    else:
        min_samples = 1
        mean_rpg = 1
    
    print("INFO:\tmean reflections per grain: %d" % mean_rpg)
    print("INFO:\tneighborhood size: %d" % min_samples)
    
    qbar, cl = run_cluster(
        completeness, qfib, plane_data.getQSym(), cfg,
        min_samples=min_samples,
        compl_thresh=compl_thresh,
        radius=cl_radius
    )

    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
    print("INFO:\tfound %d grains; saved to file: '%s'"
          % (qbar.shape[1], qbar_filename))

    np.savetxt(qbar_filename, qbar.T,
               fmt='%.18e', delimiter='\t')

    gw = instrument.GrainDataWriter(os.path.join(cfg.analysis_dir, 'grains.out'))
    grain_params_list = []
    for gid, q in enumerate(qbar.T):
        phi = 2*np.arccos(q[0])
        n = xfcapi.unitRowVector(q[1:])
        grain_params = np.hstack([phi*n, cnst.zeros_3, cnst.identity_6x1])
        gw.dump_grain(gid, 1., 0., grain_params)
        grain_params_list.append(grain_params)
    gw.close()
