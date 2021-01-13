import urllib.request
import json
import requests
import os


def create_json(path, period):
    data = requests.get(f'https://openlittermap.com/global-data?date={period}')
    if data.ok:
        geojson = json.loads(data.text)['geojson']
    else:
        raise ValueError('Got invalid response from the server.')
    files = {os.path.splitext(file)[0] for file in os.listdir(path) if os.path.isfile(file)}
    features = {feature['properties']['photo_id']: feature['properties'] for feature in geojson['features'] if str(feature['properties']['photo_id']) in files}

    with open(os.path.join(path, '..', 'jsondata.json'), 'w') as file:
        json.dump(features, file)


def download_from_time_period(period:str):
    """
    :param period: 'one-year', 'one-week', 'one-month', 'today'
    """
    data = requests.get(f'https://openlittermap.com/global-data?date={period}')
    
    if data.ok:
        geojson = json.loads(data.text)['geojson']
    else:
        raise ValueError('Got invalid response from the server.')
    
    jsondata = dict()#{feature['properties']['photo_id']: feature['properties'] for feature in geojson['features']}


    i = 0
    for feature in geojson['features']:
        _, file_extension = os.path.splitext(feature['properties']['filename'])
        try:
            file_path = '/dih4/dih4_2/wimlds/zklawikowska/openlittermap/images/%s%s' % (feature['properties']['photo_id'], file_extension)
            if not os.path.isfile(file_path):
                urllib.request.urlretrieve(feature['properties']['filename'], file_path)
            #feature['properties']['filename'] = feature['properties']['filename'].replace("\\","\\\\")
            #jsondata[feature['properties']['photo_id']] = feature['properties']
            
            i += 1
            print('saved photos %s of %s' % (i, len(geojson['features'])))
        except:
            pass
        if i % 10 == 0:
            pass
            #with open('/dih4/dih4_2/wimlds/zklawikowska/openlittermap/jsondata.json', 'w') as file:
            #    json.dump(jsondata, file)
    with open('/dih4/dih4_2/wimlds/zklawikowska/openlittermap/jsondata.json', 'w') as file:
        json.dump(jsondata, file)
   # urllib.request.urlretrieve("http://www.gunnerkrigg.com//comics/00000001.jpg", "00000001.jpg")

