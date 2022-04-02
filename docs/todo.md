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

- Run the preprocess grid search again (for Gaussian, Median blurring)

- Arrange it such that everything is logged properly; Tensorboard/PyTorch Lightning?

- Create a diary with recent runs (some may need to be rerun)

- ~~extract green tea from the raw flies~~

- Extract a small sample of a raw data tiff file for rapid testing on local machine (for predictions)

- Set up a pipeline which can take in how many manually annotated images to include

- Create a pipeline which:
    - Predicts input
        - Input is a list of filepaths for a sequence of 3D images; each image is a separate file
        - Output is a sequence of segmented 3D images; each image is a separate file
    - Extracts centroids of cells from the output
        - Center of each bounding box
        - For the first bounding box: search within the total width, height of the current cell. If none, then cell ends in the current z-coordinate.
        - The final centroid will be the average of all centroids which were deemed to be from this cell. I.e., all centroids of the z-coordinates.

- For faster debugging, have a small dataset in local environment

- Investigate how much training data is needed


- Draw the slices in 3D (or view them locally - save a small sample of a tiff file and download it)

- ~~Breyta pickle gagnaskrám yfir í venjulegar skrár, eða tiff jafnvel~~
- ~~Finna fleiri skrár~~
- ~~Athuga skrár sem hafa ekki "many beta cells" athugasemdina~~

- ~~Train a model and keep a holdout embryo for testing, predict with thresholded values~~

- - - 

*Unimportant to-do list items*

- Debug data augmentation (bounding boxes get messed up)

- Try applying median blur to /imgs/
- - because of noisy background

- Use optuna for grid search in training

- Athuga histograms af myndum í training set

- Algorithm that selects either the threshold 20 or the threshold 80 image depending on if the segmented area is too large. I know how large a cell can be. If the area is too large, select the higher threshold. Because, for threshold of 80, cells aren't segmented as a big blob as much.

- Prófa 1024x1024 upplausn með hærra batch_size -- spyrja HPC hvers vegna minnisnotkunin er svona lág í tölvupóstsskýrslunni ef það kemur upp minnisvilla í forritinu

- Nota draw_cells.py til að búa til sýnishorn úr öllum hráum skrám í einu - mögulega líka .czi.





