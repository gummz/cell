size=512, batch_size=8, lr=0.00001, wd=0.001, losses='loss_mask', 'loss_rpn_box_reg'
Segmentations not good, bounding boxes good.

size=512, batch_size=8, lr=0.00001, wd=0.001, losses='loss_mask', 'loss_rpn_box_reg' 'loss_classifier', 'loss_objectness' (all losses)
Segmentations good

leave out last file for generalizing to unseen embryos = loss = 0.5.

leave out last file, use low threshold
BAD!

leave out last file, use high threshold - good segmentations

size 128, batch size 16, bad segmentations
