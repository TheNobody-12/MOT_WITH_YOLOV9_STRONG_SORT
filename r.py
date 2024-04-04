...

for path, im, im0s, vid_cap, s in dataset:
    # s = ''
    t1 = time_sync()
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    t2 = time_sync()
    sdt[0] += t2 - t1

    # Inference
    with dt[1]:
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        pred = pred[0][1]
    t3 = time_sync()
    sdt[1] += t3 - t2

    # Apply NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    sdt[2] += time_sync() - t3

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        seen += 1
        if webcam:  # bs >= 1
            p, im0, _ = path[i], im0s[i].copy(), dataset.count
            p = Path(p)  # to Path
            s += f'{i}: '
            txt_file_name = p.stem + f'_{i}'  # Unique text file name
            save_path = str(save_dir / p.stem) + f'_{i}'  # Unique video file name
        else:
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # video file
            if source.endswith(VID_FORMATS):
                txt_file_name = p.stem
                save_path = str(save_dir / p.stem)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

        curr_frames[i] = im0

        txt_path = str(save_dir / 'tracks' / txt_file_name)  # Unique text file path
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        ...
