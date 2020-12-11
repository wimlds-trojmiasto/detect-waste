import json

# in efficientdet implementation categories should start from 1 not 0
with open('/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/annotations/annotations_detectwaste.json', 'r') as f:
    dataset = json.loads(f.read())

for category in dataset['categories']:
    category['id'] = category['id'] + 1

for annt in dataset['annotations']:
    annt['category_id'] = annt['category_id'] + 1

with open('/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/efficientdet/annotations/annotations-detectwaste-updated.json', 'w') as f:
    dataset = json.dump(dataset,f)
    