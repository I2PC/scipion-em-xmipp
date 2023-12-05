## Release 3.23.11 - Nereus
   - New protocols
      - Volume local adjustment
   - Protocols updated
      - convert_pdb: Allowed conversion natively from CIFs
      - particle_pick_automatic: The model can now be given by a directory
      - volume_local_adjust: Save occupancy volume
      - extract_particles: Added two different cases for downsampling, by dimensions and by sampling rate
   - Protocols fixed
      - movie_resize: Fixed movie resize output size
      - movie_gain: Fix update output step by using a generic one from scipion
      - tilt_analysis: Fixes in the generated tilt images and in updating correctly the output sets
      - ctf_consensus: Fix the dependencies of the step
      - preprocess_micrographs: Fixed output size in preprocess micrographs
      - deep_center_assignment: Fixed deep center calls
      - extract_particles_movies: Get coords correctly
      - particle_pick_consensus: Fix MicsPointer
      - trigger_data: fix updateOutput and close correclty the output set
   - Protocols deprecated (For more details visit [this](https://github.com/I2PC/xmipp/wiki/Deprecating-programs-and-protocols))
      - classification_gpuCorr
      - classification_gpuCorr_full
      - classification_gpuCorr_semi
   - More scipio-em-xmipp
      - Updated Nvidia driver required version


## Release 3.23.07 - Morpheus 
   - New protocols
      - Movie Dose analysis
      - deep_center
      - deep_global_assignment
      - deep_center_predict
      - deep_global_assignment_predict
   - Protocols updated
      - consensus_classes (Efficient p-value calculation, updated intersection merging process, generalized protocol for other set of classes)
      - Movie Gain: changed _stepsCheckSecs and fixed inputMovies calling, np.asscalar discontinued in numpy 1.16
      - convert_pdb: dont allow set size if template volume, to convert a set of pdbs to volumes, generates an mrc file
      - CTF_consensus: add 4 threads by default
      - process: Better instantiation of Scipion subclasses
      - create_mask3d: Addding a validate in 3dmask, add :mrc to input filename
      - consensus_local_ctf: save defocus in proper fields,  compute consensus for local defocus U and V separately, add consensus angle
      - align_volume: Included the label in the volumes
      - crop_resize: Add mask as input. Mask resize is now possible
      - subtract_projection: change pad validation error for warning, parallelized
   - Protocols fixed
      - Tilt analysis: Close correctly the output sets once finished
      - Deep micrograph cleaner: fix two bugs that occured during streaming implementation bug 
      - volume_adjust_sub: fix with :mrc
      - Picking consensus: define correctly the possibleOutputs bug 
      - Center particles: streaming bug when definining the outputs bug
      - volume_subtraction: bug fixed in filename
      - compare_reprojections: fix update subtract projection output
      - deep_micrograph_screen: Bug fix that prevents using small GPUs
      - consensus_classes:Fixed manual output generation
   - Protocols deprecated (For more details visit [this](https://github.com/I2PC/xmipp/wiki/Deprecating-programs-and-protocols))
      - apply_deformation_zernike3d
      - classify_kmeans2d
      - kmeans_clustering
      - particle_boxSize
      - rotational_spectra
      - split_volume_hierarchical_cluster
   - Viewers
      - viewer_resolution_fs: fixing 0.1 threshold not found
      - viewer_projmatch, viewer_metaprotocol_golden_highres: Fixing viewers, change removed ChimeraClientView to ChimeraView
      - monores_viewer: fix histogram
      - viewer_structure_map: Change the label for each volume


## Release 3.23.03 - Kratos
  - New protocol status: beta, new, production and updated. Will appear in the left pannel of Scipion 
  - Protocol subtract_projection: user experience improvements, no final mask by default, apply ciruclar mask in adjustment image to avoid edge artifacts, validate same sampling rate with tolerance in third decimal
  - Protocol convert_pdb: Allowed to save centered PDB used for conversion. 
  - Protocol align_volume_and_particles: add alingment validation
  - Protocol FlexAlign: updating protocol to reflect changes in the executable, fixed test, removing unused protocol (Movie average)
  - Protocol align_volume_and_particles:Align volume and particles adapted to tomography and works in the absence of tomo plugin.
  - Protocol volume_consensus: validate same sampling rate with tolerance in third decimal
  - Protocols deprecated (for more details visit the [wiki](https://github.com/I2PC/xmipp/wiki/Deprecating-programs)): protocol_deep _align, reconstruct_heterogeneous, protocol_metaprotocol_create_output, protocol_metaprotocol_discrete_heterogeneity_scheduler

  
## Hot fix 3.22.11.2
- Align volume and particles works in the absence of tomo plugin.

## Hot fix 3.22.11.1
- Align volume and particles adapted to tomography. Defines possible outputs. Optimized. Test more exhaustive for matrices

## Release 3.22.11 - Iris
  - Protocol_cl2d_align: The input can now be a set of averages or a set of 2D classes 
  - Protocol_local_ctf: Default value are now changed for maxDefocusChange
  - Protocol_apply_zernike3d: Now accepts either a Volume or SetOfVolumes and applies the coefficients in a loop in the deform step
  - Protocol_postProcessing_deepPostProcessing: Managed GPU memory to avoid errors
  - Protocol_resolution_deepres: Mandatory mask
  - Protocol center particles and Gl2d (all options): Fix streaming
  - Protocol_create_3d_mask: Allows volume Null=True
  - Protocol_reconstruct_fourier: Set pixel size
  - GL2D static: Bug fixing
  - Protocol_trigger_data: Bug fixing
  - Protocol_crop_resize: Set sampling rate of mrc files when cropping resizing volumes or particles
  - subtract_projection: New protocol for boosting particles. Add protocol to wizard XmippParticleMaskRadiusWizard as now the protocol uses it

  - **New tests:** deep_hand, pick_noise, screen_deep_learning, resolution_B_factor
  - Fixed TestHighres test

## Release 3.22.07 - Helios 
- rotate_volume: New protocol
- subtract_projection: New implementation based on adjustment by regression instead of POCS and improved performance
- local_ctf: Add new sameDefocus option + formatting
- compare_reprojections & protocol_align_volume: Fast Fourier by default
- crop_resize: Allows input pointers
- resolution_deepres: Resize output to original size
- denoise_particles: Added setOfAverages as input option
- process: Change output from stk (spider) to mrcs (mrc)
- trigger_data: Bug fixed
- screen_deeplearning:  Added descriptive help
- center_particles: Added summary info
- align_volume_and_particles: Summary error fixed
- cl2d: Summary errors solved 
- New tests: test_protocol_reconstruct_fourier, test_protocols_local_defocus, test_protocols_local_defocus, TestXmippAlignVolumeAndParticles,  TestXmippRotateVolume
- Improved tests: test_protocols_deepVolPostprocessing, test_protocols_xmipp_3d, Test ProjSubtracion
- Excluded tests: test_protocols_zernike3d, test_protocols_metaprotocol_heterogeneity

## Release 3.22.04 - Gaia
- protocol_core_analysis: New protocol
- protocol_compare_angles: Bug fix in compare angles under some conditions
- protocol_center_particles: protocol simplified (removed setofCoordinates as output)
- protocol_CTF_consensus: concurrency error fixed
- protocol_convert_pdb: remove size if deactivated
- protocol_resolution_deepres: binary masked not stored in Extra folder and avoiding memory problems on GPUs
- protocol_add_noise: fixes
- protocol_compare_reprojections: improve computation of residuals + tests + fix + formatting
- protocol_screen_deepConsensus: multiple fixes in batch processing, trainging and streaming mode
- protocol_shift_particles: apply transform is now optional


## Release 3.22.01 - Eris
- Visit changeLog.md of xmipp

## Release 3.21.06 - Caerus

- CUDA-11 support
- New protocol: Deep align
- ChimeraX support
- Improvements of streaming process
- Several performance optimizations
- Build time optimization
- Multiple bug fixes
- Improved documentation


## Release 3.20.07 - Boreas

- New Protocol: MicrographCleaner is a new algorithm that removes coordinates picked from carbon edges, aggregations, ice crystals and other contaminations
- New functionality: The protocol compare reprojections can now compute the residuals after alignment
- New protocol: Split frames divide input movies into odd and even movies so that they can be processed independently
- New protocol: Continuous heterogeneity analysis using spherical harmonics (not ready to be used)
- Bug fixing when some micrograph has no coordinates in the consensus-picking.
- New functionalities: Different architectures and training modes
- Normal Mode Analysis protocols have been moved to the plugin ContinuousFlex
- Fixing MPI version of the Fourier Reconstruction
- New protocol: local CTF integration and consensus protocol for local ctf (also the viewers)
- Local CTF analysis tools: Not yet ready for general public
- New functionallity: Introducing the posibility of automatic estimation of the gain orientation.
- Bugs fixings regarding stability on streaming processing
- Support of heterogeneous movie sets
- New protocol: Clustering of subtomogram coordinates into connected components that can be processed independently
- New Protocol: Removing duplicated coordinates
- New protocol: Subtomograms can be projected in several ways to 2D images so that 2D clustering tools can be used
- New protocol: Regions of Interest can be defined in tomograms (e.g., membranes)
- Bug fixing in mask3d protocol
- Bug fix: in helical search symmetry protocol
- Enhanced precision of the FlexAlign program
- Now, deepLearningToolkit is under its own conda environment
- Multiple protocols accelerated using GPU
- New functionality: Xmipp CTF estimation can now take a previous defocus and do not change it
- New functionallity: CTF-consensus is able to take the primary main values or an average of the two.
- New functionallity: CTF-consensus is able to append metadata from the secondary input
- New functionality: Xmipp Highres can now work with non-phase flipped images
- New functionality: Xmipp Preprocess particles can now phase flip the images
- New protocol: Tool to evaluate the quality of a map-model fitting
- Allowing multi-GPU processing using FlexAlign
- Improvement in monores and localdeblur
- Randomize phases also available for images
- Change the plugin to the new Scipion structure
- Migrating the code to python3

## Release 3.19.04 -

- Highres can now take a global alignment performed by any other method
- New protocol: 3D bionotes
- New protocol: Align volume and particles
- New protocol: Center particles
- New protocols: GL2D, GL2D streaming and GL2D static
- New protocol: 2D kmeans clustering
- New protocol: compare angles
- New protocol: consensus 3D classes
- New protocol: CTF consensus
- New protocol: deep denoising
- New protocols: Eliminate empty particles and eliminate empty classes
- New protocol: Extract unit cell
- New protocol: Generate reprojections
- New protocol: metaprotocol heterogenety output, metaprotocol heterogeneity subset and metaprotocol heterogeneity
- New protocol: Movie Max Shift
- New protocol: particle boxsize
- New protocol: pick noise
- New protocol: significant heterogeneity
- New protocol: swarm consensus intial volumes
- New protocol: directional ResDir
- New protocol: local monoTomo
- New protocol: deep consensus picking
- New protocol: screen deep learning
- New protocol: split volume hierarchical
- New protocol: trigger data
