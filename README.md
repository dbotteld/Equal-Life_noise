# EL_ML
Equal Life Machine learning model


# to use this code, install anaconda
python library dependence can be found in eq_life_env.txt that can be loaded using conda create --name myenv --file ml_model_env.txt
start conda prompt
activate the environment

# UseModel
The code for running the trained ML model (layout of D3.5) on a local dataset. The main code can be found in ml_indicators_for_area.py and calculates pre-defined indicators at all locations in an area. and overview of the code layout can be found in D3.5.

The input parameters are assembled in a .json file
an example is found in: UseModel\ghent_antwerp_200dwellings.json

The input data is a file with coordinates (in WKT) and a rowID. Example file:
receivers\ghent_antwerp_random_200points_in_dwellings.csv

Starting the code is specified in the beginning of the code:
python UseModel\ml_indicators_for_area.py UseModel\yourdefinition.json  

srid_of_receiver = 4326 #use 4326 if in OSM lat lon
use local reference system is relevant
provide a catresian reference system for the calculations (required for distances).


this version of the code requires explicit edge betweenness file 
all geometrical data are read from OSM online during execution of the code

Output:

numpy and shape files

most important: diurnal indicators per dwelling: file_dwellingshape_save = '../../Predictions/BarcelonaDwelling4b.shp'


Dependencies:
the trained model: file_model = 'CNN4values_new140epochs'
the input feature scaler: file_input_scaler = 'CustomGroupInputScaler.pkl'
The output indicator scaler: file_output_scaler = 'CustomOutputScalerPBF2.pkl'

