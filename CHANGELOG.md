## Release 26.0.2
- More scipion-em-xmipp
   - Remove scipion dependencies from requirements   

## Release 26.0.1
- More scipion-em-xmipp
   - Fix on xmipp installation flag
     
## Release 26.0.0
- Protocols updated
   - deep_micrograph_screen:  updated enviroment to tensorFlow2 and models updated to  
   - movie_max_shif: Change the criteria to evaluate the shift
   - crop_resize: Crop/Resize half maps too, if they are available
   - resolution3D: The FSC can be computed now using the two halves
   - align_volume_and_particles: Allow considering mirrors in align volume and particles

- More scipion-em-xmipp
   - Disable xmippDep by default (the dependencies will not be installed by default in the Scipion environment), but installing just the libstdcxx dependence
   - Script to generate the protocols map (https://i2pc.github.io/docs/protocolsMap.html#)
   - Remove Relion dependency from TestXmippDenoiseParticles
   - parse json to report summary installation errors
   - View all classes added for XmippProtConsensusClasses


## Release 3.25.06.0 - Rhea
- New protocols
   - compute_likelihood: This protocol computes the likelihood of a set of particles with assigned angles when compared to a set of maps or atomic modelS

- Protocols updated
   - Max shift: tolerate movies without micrographs associated
   - substract_projection: update command to boost and subtract to generate only one output stack
   - Movie alignment consensus protocol: enables irregular input sets. Changes in functionality; comparisons between ons between shifts trajectories, and a additional change related to the minimum consensus or threshold for shifts correlation.
   - volume_adjust_sub: better label for volume 1 vs 2
   - convert_pdb: aDD binThreads
   - align_volume: add possible
   - particle_pick_consensus: Fixed ID mismatch when combining coordinate sets; now supports asymmetrical sets without indexing errors.
   - movie_split_frames: now allows testing sum_frames, includes a summary, improved layout, and updated test.
   - deep_Consensus: GPU handling
   - Deep_hand: handling env to avoid system CUDA settings,  erasing cuda things from env['LD_LIBRARY_PATH']
   - Extract particles protocol: force output boxsize
   - reconstruct_fourier: correctEnvelope False by default
   - reconstruct_highres: test updated, correctEnvelope False by default
   - Structure map: Adding a dendrogram and removing some intermediate files


- Protocols fixed
   - simulate ctf: fix pre-CTF noise np.numpy, init generator for pre and post ctf noise
   - validate FSCQ: test fixed
   - resolution_monogenic_signal: fixing header in MonoRes
   - protocol_process: CropResize halves fix
   - Extract particles test fixed
   - FSO viewer: the fso was failing when the fso did not cross some thresholds
   - reconstruct_significant: fix gpu usage
   - ransac: fix gpu usage
     
- Protocols deprecated (For more details visit [this](https://github.com/I2PC/xmipp/wiki/Deprecating-programs-and-protocols](https://i2pc.github.io/docs/Utils/Deprecated-programs/index.html)))
   - XmippProtScreenDeepLearning

- More scipion-em-xmipp
   - Most of the Xmipp dependencies installed on Scipion env
   - Updating version for pypi requirements
   - Bug cloning tag
   - Use toml for installation
   - Bug fixed related to the WriteSetOfParticles and the Missing ExtraLabels (The Oier and Fede Adventures)
   - Handling nvidia drivers error message
   - improving deps installation
   - Improved the help message of more than 60 protocols
        
## Release 3.24.12 - Poseidon
   - New protocols
      - apply_tilt_to_ctf: Apply a local deviation to the CTF based on the micrograph's tilt
    angle
      - mics_defocus_balancer: It takes as input a certain number of CTFs and makes a balanced sampling of N images based on the defocus U values
      - deep_center_predict: Center a set of particles in 2D using a neural network
      - cl2d_clustering: 2D clustering protocol to group similar images (2D Averages or 2D Classes)

   - Protocols updated
      - flexalign: Added support for reading EER. Optimized and refactor the streaming, the continue action stabilized. FlexAlign can run multiple times in a single GPU or in multiple GPUs from within Scipion3.
      - convert_pdb: now can read cif
      - create_gallery: Allow Scipion protocol create gallery to access missing projection options
      - ctf_consensus: optimized and refactor the streaming, the continue action stabilized. Remove parallelization
      - movie_dose_analysis: optimized and refactor the streaming, the continue action stabilized
      - movie_max_shift: optimized and refactor the streaming, the continue action stabilized, Remove parallelization
      - tilt_analysis: optimized and refactor the streaming, the continue action stabilized
      - subtract_projection: Adaptations for new logic in Xmipp program
      - reconstruct_highres: Add GPU use for local angle assignment 

   - Protocols fixed
      -  trigger_data: Fix the way we split the output
      -  subtract_projection: fnMask sub
      -  classify_pca: Bug fix in the validate function
      -  reconstruct_fourier: Fourier reconstruction now allows performing the CTF correction inside the same protocol.
      -  volume_local_sharpening: local deblur performs now a single iteration by defaul
      -  movie_max_shift: fixed with several parallel threads in streaming
      -  reconstruct_highres: add unassigned vars fnVol1 fnVol2 to set loop
      -  ctf_defocus_group: fix when testing multiple downsampling factors
      -  ctf_micrographs: fix when testing multiple downsampling factors

   - Protocols deprecated (For more details visit [this](https://github.com/I2PC/xmipp/wiki/Deprecating-programs-and-protocols))
      - angular_resolution_alignment
      - movie_opticalflow.py
        
   - More scipion-em-xmipp
      - scikit-learn updated to 1.2
      - protocols.conf list completed. All protocols are listed and clustered in Scipion

## Release 3.24.06 - Oceanus
   - New protocols
      - movie_alignment_consensus: protocol for the comparison of two global alignments
      - PCA classification: 2D Classification method that can works in streaming and in static

   - Protocols updated
      - deep_center: Deep center calculates an approximate center for the particles.
      - validate_fscq: Added cif support
      - reconstruct_fourier: Enable reconstructing with halves in the reconstruct fourier
      - simulate_ctf: Simulate CTF can add noise before and after CTF.
      - extract_particles: Local average added
      - compare_reprojections: Downsampling option,autodown sampling, allows input 2D in several formats, allows input 3D in several formats, ranking option available, extraction option available
      - cl2d: ensuring a save classes creating and adding possible output
   - Protocols fixed
      - resolution_deepres:now works with queue system
      - ProcessVolumes: Fix header for resize voxels
      - resolution_bfactor:Fix local resolution to local bfactor  
      - resolution_bfactor: create output for res bfac
      - extract_particles: Bug fix in the downsampling factor of extract particles
      - core_analysis: define correctly the particles pointer to have indirect pointer
      - tilt_analysis: estimate automatically the window size new option
      - movie_doseanalysis: update correctly the outputSet
      - extract_particles: Check if resizing is enabled
      - particle_pick_remove_duplicates: Remove duplicates coordinates 
   - Protocols deprecated (For more details visit [this](https://github.com/I2PC/xmipp/wiki/Deprecating-programs-and-protocols))
      - deep_denoissing
      - metaprotocol_create_subset
      - metaprotocol_golden_highres
      - solid_angle
      - split_volume
   - More scipion-em-xmipp
      - Solve Sonar Cloud reported bugs
      - Flexible import of pyQT on chimera script
      - Removing tkAgg matplotlib requirement in viewer_structure_map
      - DLTK installation bug local variable and Fix use gpu
      - updated scikit-learn version


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
   - More scipion-em-xmipp
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
