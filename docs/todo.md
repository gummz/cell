- Remember - sometimes it's better to conduct experiments manually than to write code for a single experiment

- Re-draw the same images from the raw dataset so that no automatic annotations are present in the dataset

- Annotate 200 images with the optimal parameters, including validation set (with the samples)
    - Finish validation set
    - Upload to DTU

- Train a model on the images
    - One in which the input is vanilla input
    - One in which the input is the preprocessed (adaptive hist -> bilateral) input
        - Take mean of equalized and raw image


- Compare with previous results
    - Compare when model input is vanilla (bilateral) input
    - Compare when model input is preprocessed (adaptive hist -> bilateral) input
    - Just run eval_track, 1 batch, for each combination (there are four combinations in total)

- If model is significantly better, annotate 400 more images

- Combine Kasra's results with mine and analyze

- Reg R-RCNN

- Dimension reduction methods

- Pixel intensity histograms of all files
    - Compare those for which the equalization works well with those for which it doesn't work as well
    - Does the method's effectiveness depend on the number of pixels above a certain threshold in the target image?
        - Seems so: if there are many bright cells in the image, it works remarkably well
    - Maybe the files have different noise levels?
    - Need to thoroughly understand how the method works

- In the empty space in the subplot in eval_track, add the unprocessed MIP of the beta cell channel.

- Change annotate_from_json so that no intermediate files or folders are created

- Use MIPs, so xy, xz, yz? (x, y, z are the three axes). For training and prediction.

- LabelMe already outputs segmentations that are colored by identity. So just use that to assign identities.

- Fix Java out of memory: install conda on HPC
    - https://allencellmodeling.github.io/aicsimageio/aicsimageio.readers.html?highlight=bioformat#aicsimageio.readers.bioformats_reader.BioformatsReader

- Make an absolute color scale for 3D points (right now, the relatively brightest is a dark dot) (is it?)
    - Investigate what should be considered a high intensity, so that I can determine when a cell activates
    - Use histogram over all timepoints of all
        - sequential processing, one at a time, save to disk
    - Or Silja's method of gradients
    - Investigate cell intensity, why is it so low?

- 3D segmentations
    - use for tracking (size, shape, etc.)
    - could even use a custom algorithm for detecting cell based on object shape

- Add turn-on point for each cell based on steepest ascent of intensity. In `X, Y, P, I, O` where `O` is a dictionary with cell number as key, and frame in which cell turned on as value.

- Create `make` commands for json, train, test, etc.

- Logging, multithread/multiprocessing, learn tensorboard better

- Packages for 3D visualization

- Kasra segmentations

- Make contribution to aicsimageio: warn user when a datafile from the same path has been created multiple times

- Submit bug report to AICSImageIO GitHub regarding memory leakage

- - -

*Done*

- slice_to_mip_movie script to show 5 timepoints before and 5 timepoints after

- Make annotate_from_json.py ready

- Find optimal adaptive histogram equalization parameters

- Model calibration
    - Train model without manual labels and investigate

- Tracking grid search - metric is: average change in number of cells frame-by-frame. Fewer changes = better tracking result.

- Rich visualizations for the tracked centroids. Temporal paths on one image, with a marker where the cell turned on.

- Redo the manual label impact tables with the new runs

- Create script in `/visualization/` which will serve to visualize all kinds of stuff for the report

- Experiment: what threshold to use for matching target with the best prediction, and how uncertain the model can be before rejection.
    - Combined grid search
    - Contour plot/grid of results since 2D

- manual search both where n_img_select is 1100, and where n_img_select == manual_select (compare in thesis to see which is better)

- Look at results from broad gridsearch and narrow it down

- Train model on new annotations, manual_select on new annotations, predict and track with new model, output eval_track with new model

- Use external (offline) video service to make videos from pngs

- Tracking cells
    - Feed centers into TrackPy
    - Visualize TrackPy results in 3D (final output)
    - Visualize TrackPy results in 2D (for validation)
        - eval_track.py
        - Maximal intensity projection
        - cross/dot markers
        - colors which are easy to distinguish
            - intensity of color = intensity of cell?
        - axes with ticks for easy coordinates
        - overlay with tubes (channel other than 0)
        - from `centroids_np`, just use `t`, `x`, `y`, `p` (particle) 
        - turn up signal of cells
        - plot coordinates onto MIP of raw timepoint

- Threshold masks AFTER thresholding scores. So we already have masks with object score over 0.5. Do this to get more precise segmentations. Do this first in eval_model and see how it affect average mask score.

- Experiments
    - Active slices
    - Hyperparameter grid search with optuna (incl pretrained)
        - Investigate how much training data is needed
        - Investigate how many images with manual labels are needed
            - Validate the model on:
                - no manual labels
                - incrementally increasing number of manual labels
                - i.e., load the validation DataLoader with manual_select=0, and =1
        - Investigate which filter (or no filter) is best on original image
            - Add Canny edge detection
    - Preprocessing: filters, thresholds
        - Frangi, median, mean, Gaussian: choose two
        - Adaptive, simple thresholding

- Show plots of val and train losses in manual_search

- `slice_to_mip.py`: input is slice of timepoint of file, output is MIP of timepoint of file. Some images will be identical (because they are from the same timepoint) but that's okay.

- Threshold boxes and masks with regard to score

- Visualize centers from predict_model.py

- Modified Non-Maximum Suppression to eliminate false positives

- Evaluate test set

- Arrange it such that everything is logged properly; Tensorboard/PyTorch Lightning?

- Create a pipeline which:

    - Predicts input
        - Input is a list of filepaths for a sequence of 3D images; each image is a separate file
        - Output is a sequence of segmented 3D images; each image is a separate file
    - Extracts centroids of cells from the output
        - Center of each bounding box
        - For the first bounding box: search within the total width, height of the current cell. If none, then cell ends in the current z-coordinate.
        - The final centroid will be the average of all centroids which were deemed to be from this cell. I.e., all centroids of the z-coordinates.

- Set up a pipeline which can take in how many manually annotated images to include


- Change draw_cells to use access one timepoint at a time

- For faster debugging, have a small dataset in local environment

- Draw the slices in 3D (or view them locally - save a small sample of a tiff file and download it)

- Make validation use a separate embryo

- Create a test set in which manual annotations of the weaker annotations have been added
    - Choose suitable files for testing and add them to `RAW_FILES_GENERALIZE`
        - Debug `LI_2018-11-20_emb6_pos1.lsm` in `draw_cells`
        - Calculate average pixel intensity
        - Cross-reference with `/test/` folder in `/dtu-compute/`
        - (when choosing training set, cross-reference with `/train/`)
            - When doing different versions of datasets, use default names and use `mv` in the terminal afterwards!

    - Use `make_dataset_test.py` to create `/imgs_test/`

    - Use `annotate_test.py` to create png annotations in `/masks_test/`

    - Download `masks_test` to local computer

    - Manually annotate with Labelme

    - Use `generate_json.py` to output manual annotations into subfolders in `/masks_test/`

    - Use `annotate_from_json.py` to go into subfolders, threshold `label.png`, and save to `/masks_test_full/`

- ~~Train a model and keep a holdout embryo for testing, predict with thresholded values~~

- ~~Breyta pickle gagnaskr??m yfir ?? venjulegar skr??r, e??a tiff jafnvel~~
- ~~Finna fleiri skr??r~~
- ~~Athuga skr??r sem hafa ekki "many beta cells" athugasemdina~~

- Extract a small sample of a raw data tiff file for rapid testing on local machine (for predictions)

- ~~extract green tea from the raw flies~~

- Use optuna for grid search in training

- Try applying median blur to /imgs/
- - because of noisy background

- Create script that extracts 9 samples from each raw data file and puts them into a figure

- - - 

*Unimportant to-do list items*

- Debug data augmentation (bounding boxes get messed up)

- Algorithm that selects either the threshold 20 or the threshold 80 image depending on if the segmented area is too large. I know how large a cell can be. If the area is too large, select the higher threshold. Because, for threshold of 80, cells aren't segmented as a big blob as much.






