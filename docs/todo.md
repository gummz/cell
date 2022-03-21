



- Arrange it such that everything is logged properly; Tensorboard/PyTorch Lightning?

- Create a diary with recent runs (some may need to be rerun)

- Create a test set in which manual annotations of the weaker annotations have been added

- - Set up a pipeline which can take in how many manually annotated images to include

- Create a pipeline which:
- - Predicts input
- - - Input is a list of filepaths for a sequence of 3D images; each image is a separate file
- - - Output is a sequence of segmented 3D images; each image is a separate file
- - Extracts centroids of cells from the output
- - - Center of each bounding box
- - - For the first bounding box: search within the total width, height of the current cell. If none, then cell ends in the current z-coordinate.
- - - The final centroid will be the average of all centroids which were deemed to be from this cell. I.e., all centroids of the z-coordinates.

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





