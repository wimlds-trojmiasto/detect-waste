import yaml
import sys
import os
import pprint

from my_coco import MYCOCO
from pascal import PASCAL

pp = pprint.PrettyPrinter(indent=0)

configf = 'config.yaml'


def run_convert_from_X2Y(config_file):
    """
    Convert X annnotation to Y annotation format.
    Keys in config file:
        :param load_format: Class with input annotations
        :param save_format: Class to save new annotations
        :param input_path: path with input annotations X
        :param save_path: path to save input annotations Y
        :param img_extension: extension of image file, default is '.png'
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    print(f"Start convert from {config['load_format']} to {config['save_format']}..")

    annotation_X = eval(config['load_format'])

    # create empty json to save new annotations
    if config['save_format'] == 'COCO':
        with open(config['save_path'], mode='a'):
            pass
    else:  # or create dir if it does not exist
        if not os.path.exists(config['save_path']):
            os.mkdir(config['save_path'])

    annotation_Y = eval(config['save_format'])

    loaded_dict = annotation_X._load_labels(config['input_path'],
                                            config['num_workers'])
    save_annotation = annotation_Y()

    # possibility to add dict with classes in config yaml
    if 'categories' not in config:
        config['categories'] = loaded_dict['categories']
    save_annotation.set_classes(config['categories'])
    pp.pprint(save_annotation.classes)
    for item in loaded_dict['items']:
        save_name = item['file_name'] + config['img_extension']
        if os.path.isdir(config['save_path']): # This dir must exist!
            save_name = os.path.join(config['save_path'], save_name)
        try:
            save_annotation.add_item(save_name,
                                     item['boxes'],
                                     item['size'])
        except Exception as e:
            print(e, sys.exc_info(), sep='\n')
            sys.exit()

    try:
        save_annotation.save_labels(config['save_path'])
    except Exception as e:
        print(e, sys.exc_info(), sep='\n')
        sys.exit()

run_convert_from_X2Y(configf)
