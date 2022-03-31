### Testing:

- Original files are in `/imgs_test/`
- Automatic annotations, pasted over the original images, are in `/masks_test/`
- Manual annotations are applied to automatic annotations
- `generate_json.py` puts the complete annotations into folders
- `annotate_from_json` goes into each folder, thresholds the image `label.png`, and saves it in `/masks_test_full/`.