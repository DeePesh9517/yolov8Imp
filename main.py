import argparse
import sys

from utils import Predict, Trainer, view_image

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help="Path to model", default='yolov8n-cls.pt', type=str)
parser.add_argument('--mode', help="model - classify, segment, detect", default='classify', type=str)
parser.add_argument('--confidence', help='confidence', default=0.5, type=float)
parser.add_argument('--image_size', help='size of image to input', default=512, type=int)
parser.add_argument('--save', help='True or False', default=True, type=bool)
parser.add_argument('--train', help='True or False', default=False, type=bool)
parser.add_argument('--config_file', help='path to the cofig file', type=str)
parser.add_argument('--epochs', help='number of epochs to train on', type=int, default=10)
parser.add_argument('--batch_size', help='size of the batch', type=int, default=8)
parser.add_argument('--device', help='cpu or cuda', type=str, default='cpu')
parser.add_argument('--dataset', help='path to dataset', type=str)
parser.add_argument('--image_path', help='path to the image', type=str)
parser.add_argument('--view_image', help='True or False', type=bool, default=False)

args = vars(parser.parse_args())

if args['mode'] in ['classify', 'segment', 'detect'] and not args['train']:
    if not args['image_path'] or args['image_path'] == '':
        print('use image_path argument for image path')
        sys.exit()
    if args['mode'] == 'classify':
        predict = Predict(args['model_path'])
        results = predict.classify(args['image_path'], args['image_size'], args['save'], args['confidence'])

        if args['save'] and args['view_image']:
            view_image(args['image_path'], args['mode'])

    if args['mode'] == 'segment':
        predict = Predict(args['model_path'])
        results = predict.segment(args['image_path'], args['image_size'], args['save'], args['confidence'])

        if args['save'] and args['view_image']:
            view_image(args['image_path'], args['mode'])

    if args['mode'] == 'detect':
        predict = Predict(args['model_path'])
        results = predict.detect(args['image_path'], args['image_size'], args['save'], args['confidence'])

        if args['save'] and args['view_image']:
            view_image(args['image_path'], 'segment')

if args['train']:
    try:
        trainer = Trainer(args['model_path'])
        trainer.train(datasets=args['dataset'], config_file=args['config_file'],
                      batch_size=args['batch_size'], device=args['device'],
                      image_size=args['image_size'], epochs=args['epochs'])
    except AttributeError as e:
        print(e)
    except TypeError as e:
        print(e)
