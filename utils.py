import numpy as np
from ultralytics import YOLO
import yaml
from yaml.loader import SafeLoader
import os
import cv2


class Predict:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.name = self.model.names

    def classify(self, image_path, image_size=512, save=True, confidence=0.5):
        results = self.model.predict(source=image_path, imgsz=image_size, save=save, conf=confidence)
        result = results[0]
        names = result.names
        index = int(result.probs.data.max(axis=0).indices)
        probability = float(result.probs.data.max(axis=0).values)
        label = names[index]
        return label, probability

    def detect(self, image_path, image_size=512, save=True, confidence=0.5):
        results = self.model.predict(source=image_path, imgsz=image_size, save=save, conf=confidence)
        result = results[0]
        boxes = list(result.boxes.data)
        names = result.names
        result_list = []
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            w = int(box[2])
            h = int(box[3])
            conf = float(box[4])
            label = int(box[5])
            label_text = names[label]

            if conf < confidence:
                continue

            result_list.append([x1, y1, w, h, conf, label_text])

        return result_list

    def segment(self, image_path, image_size=512, save=True, confidence=0.5):
        results = self.model(source=image_path, imgsz=image_size, save=save, conf=confidence)
        result = results[0]
        segment_list = []

        for result_xy in result.masks.xy:
            point_ls = []

            for point in result_xy:
                n_point = (int(point[0]), int(point[1]))
                point_ls.append(n_point)

            point_ls = np.array([point_ls], np.int32)

            segment_list.append(point_ls)

        box_list = []
        boxes = list(result.boxes.data)
        names = result.names
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            w = int(box[2])
            h = int(box[3])
            conf = float(box[4])
            label = int(box[5])
            label_text = names[label]

            if conf < confidence:
                continue

            box_list.append([x1, y1, w, h, conf, label_text])

        return segment_list, box_list


class Trainer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def train(self, datasets, epochs=2, image_size=512, batch_size=8, device='cpu', config_file=None):
        if config_file:
            with open(config_file) as f:
                data = yaml.load(f, Loader=SafeLoader)

            train_result = self.model.train(task=data['task'], mode=data['mode'],
                                            model=data['model'], data=data['data'],
                                            epochs=data['epochs'], patience=data['patience'],
                                            batch=data['batch'], imgsz=data['imgsz'],
                                            save=data['save'], save_period=data['save_period'],
                                            cache=data['cache'], device=data['device'],
                                            workers=data['workers'], project=data['project'],
                                            name=data['name'], exist_ok=data['exist_ok'],
                                            pretrained=data['pretrained'], optimizer=data['optimizer'],
                                            verbose=data['verbose'], seed=data['seed'],
                                            deterministic=data['deterministic'], single_cls=data['single_cls'],
                                            rect=data['rect'], cos_lr=data['cos_lr'],
                                            close_mosaic=data['close_mosaic'], resume=data['resume'],
                                            amp=data['amp'], fraction=data['fraction'],
                                            profile=data['profile'], overlap_mask=data['overlap_mask'],
                                            mask_ratio=data['mask_ratio'], dropout=data['dropout'],
                                            val=data['val'], split=data['split'],
                                            save_json=data['save_json'], save_hybrid=data['save_hybrid'],
                                            conf=data['conf'], iou=data['iou'],
                                            max_det=data['max_det'], half=data['half'],
                                            dnn=data['dnn'], plots=data['plots'],
                                            source=data['source'], show=data['show'],
                                            save_txt=data['save_txt'], save_conf=data['save_conf'],
                                            save_crop=data['save_crop'], show_labels=data['show_labels'],
                                            show_conf=data['show_conf'], vid_stride=data['vid_stride'],
                                            line_width=data['line_width'], visualize=data['visualize'],
                                            augment=data['augment'], agnostic_nms=data['agnostic_nms'],
                                            classes=data['classes'], retina_masks=data['retina_masks'],
                                            boxes=data['boxes'], format=data['format'],
                                            keras=data['keras'], optimize=data['optimize'],
                                            int8=data['int8'], dynamic=data['dynamic'],
                                            simplify=data['simplify'], opset=data['opset'],
                                            workspace=data['workspace'], nms=data['nms'],
                                            lr0=data['lr0'], lrf=data['lrf'],
                                            momentum=data['momentum'], weight_decay=data['weight_decay'],
                                            warmup_epochs=data['warmup_epochs'],
                                            warmup_momentum=data['warmup_momentum'],
                                            warmup_bias_lr=data['warmup_bias_lr'], box=data['box'],
                                            cls=data['cls'], dfl=data['dfl'],
                                            pose=data['pose'], kobj=data['kobj'],
                                            label_smoothing=data['label_smoothing'], nbs=data['nbs'],
                                            hsv_h=data['hsv_h'], hsv_s=data['hsv_s'],
                                            degrees=data['degrees'], translate=data['translate'],
                                            scale=data['scale'], shear=data['shear'],
                                            perspective=data['perspective'], flipud=data['flipud'],
                                            fliplr=data['fliplr'], mosaic=data['mosaic'],
                                            mixup=data['mixup'], copy_paste=data['copy_paste'],
                                            cfg=data['cfg'], tracker=data['tracker']
                                            )

        else:
            train_results = self.model.train(
                batch=batch_size,
                device=device,
                data=datasets,
                epochs=epochs,
                imgsz=image_size,
            )


def view_image(image_path, mode):
    predict_path = os.path.join('runs', mode)
    predict_folders = os.listdir(predict_path)

    image_name = image_path.replace('\\', r'/').split('/')[-1]
    if len(predict_folders) == 1:
        image_path = os.path.join(predict_path, 'predict', image_name)

    else:
        predict_list = [a.replace('predict', '') for a in predict_folders]
        predict_list = [a.replace('train', '0') for a in predict_list]
        predict_list = predict_list[1:]
        predict_list = [int(a) for a in predict_list]
        predict_list.sort()
        latest_predict = predict_list[-1]
        image_path = os.path.join(predict_path, f'predict{latest_predict}', image_name)

    image = cv2.imread(image_path)
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
