# function converts dataset_to_convert to match categories in dataset_template
# it is designed to work with taco and epinote categories

import json

epi_path = '/dih4/dih4_2/wimlds/data/annotations_epi.json'
taco_path = '/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/efficientdet/annotations/annotations-detectwaste-updated.json'

with open(epi_path, 'r') as f:
    dataset_epi = json.loads(f.read())
with open(taco_path, 'r') as f:
    dataset_taco = json.loads(f.read())
save_path = '/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/efficientdet/annotations/annotations-epi-to-taco.json'


def convert_dataset(dataset_template, dataset_to_convert, save_path):
    # function converts dataset_to_convert to match categories in dataset_template
    template_categories = []
    categories = []
    taco_to_epi = {}
    taco_id_to_epi_id = {}

    for i, (template_category, category) in enumerate(zip(dataset_template['categories'], dataset_to_convert['categories'])):
        if template_category['name'] == 'non_recyclable':
            template_category['name'] = 'non-recyclable'
        elif category['name'] == 'non_recyclable':
            category['name'] = 'non-recyclable'
        elif template_category['name'] == 'metals_and_plastic':
            template_category['name'] = 'metals_and_plastics'
        elif category['name'] == 'metals_and_plastic':
            category['name'] = 'metals_and_plastics'

        template_categories.append(template_category['name'])
        categories.append(category['name'])
        taco_to_epi[category['name']] = template_category['name']
        category['name'] = template_category['name']
        category['category'] = template_category['name']
        category['supercategory'] = ''

    for taco_id, category in enumerate(categories):
        epi_id = template_categories.index(category)
        taco_id_to_epi_id[taco_id+1] = epi_id+1

    print('Old categories:', categories)
    print('New categories:', template_categories)
    for annt in dataset_to_convert['annotations']:
        annt['category_id'] = taco_id_to_epi_id[annt['category_id']]

    with open(save_path, 'w') as f:
        dataset = json.dump(dataset_to_convert, f)
    print('Finished')


# convert form epi to taco
convert_dataset(dataset_taco, dataset_epi, save_path)
