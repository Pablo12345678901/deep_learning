import os
from osgeo import ogr
import pathlib
import sys

# Check that one and only one argument was provided
# Reminder :
#    sys.argv[0] = name of the script
#    sys.argv[1] = first argument ... and so on
if len(sys.argv) == 1:
    print("\nERROR : please provide the path of the shapefile as this script argument.\n")
    sys.exit(1)
elif len(sys.argv) > 2:
    NUMBER_OF_PROVIDED_ARGUMENTS = len(sys.argv) - 1 # -1 because arg[0] = script name
    print("\nERROR : too many arguments (" + str(NUMBER_OF_PROVIDED_ARGUMENTS) + ") were provided : " + str(sys.argv[1:]) + "\n. Only provide the path of the shapefile as this script argument\n")
    sys.exit(2)

# Shapefile is provided as first argument to script
daShapefile = sys.argv[1]
# Two shapefile to compare :
# 1. '/home/incognito/Desktop/developpement/deep_learning/dev_and_tests_from_deep_learning_cookbook_examples/conda_venv/lib/python3.11/site-packages/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp'
#
# 2. '/home/incognito/Desktop/developpement/deep_learning/dev_and_tests_from_deep_learning_cookbook_examples/data/world_map_data/ne_10m_admin_0_countries.shp'


# Check if the file exists - else exit with error
file_exists = os.path.exists(daShapefile)
if not file_exists:
    print("\nERROR : the file \"" + daShapefile + "\" does not exist.\n")
    sys.exit(3)
    
# Check that the file ends with a '.shp' extension
file_extension = str(pathlib.Path(daShapefile).suffix)
if not file_extension == ".shp":
    print("\nERROR : the file \"" + daShapefile + "\" is not a shapefile (does not have a '.shp' extension).\n")
    sys.exit(4)

# Code below copy/pasted from :
# https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#get-shapefile-fields-get-the-user-defined-fields
dataSource = ogr.Open(daShapefile)
daLayer = dataSource.GetLayer(0)
layerDefinition = daLayer.GetLayerDefn()

# Print fields
"""
for i in range(layerDefinition.GetFieldCount()):
    print(layerDefinition.GetFieldDefn(i).GetName())
"""

# Print fields and type
print("Name  -  Type  Width  Precision")
for i in range(layerDefinition.GetFieldCount()):
    fieldName =  layerDefinition.GetFieldDefn(i).GetName()
    fieldTypeCode = layerDefinition.GetFieldDefn(i).GetType()
    fieldType = layerDefinition.GetFieldDefn(i).GetFieldTypeName(fieldTypeCode)
    fieldWidth = layerDefinition.GetFieldDefn(i).GetWidth()
    GetPrecision = layerDefinition.GetFieldDefn(i).GetPrecision()

    print(fieldName + " - " + fieldType+ " " + str(fieldWidth) + " " + str(GetPrecision))
