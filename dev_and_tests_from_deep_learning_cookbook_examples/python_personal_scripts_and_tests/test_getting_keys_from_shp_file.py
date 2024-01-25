from osgeo import ogr

# Code below copy/pasted from :
# https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#get-shapefile-fields-get-the-user-defined-fields

daShapefile = '/home/incognito/Desktop/developpement/deep_learning/dev_and_tests_from_deep_learning_cookbook_examples/data/world_map_data/ne_10m_admin_0_countries.shp'

dataSource = ogr.Open(daShapefile)
daLayer = dataSource.GetLayer(0)
layerDefinition = daLayer.GetLayerDefn()


for i in range(layerDefinition.GetFieldCount()):
    print(layerDefinition.GetFieldDefn(i).GetName())
