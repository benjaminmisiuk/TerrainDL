# TerrainDL
This repository provides the code used for the analysis detailed in the manuscript **Evaluating the deep learning of terrain information from elevation data**, by Benjamin Misiuk, Alexandre Schimel, and Vincent Lecours.

### Code
This repository includes both R and Python code. The scripts are described below, and are generally run in this order to reproduce the analysis.

* **R/terrain_attributes.R** <br>
This script is used to generate terrain parameters from the resampled 5 m Ponui Island DTM. The DTM is made available by Toitū Te Whenua Land Information New Zealand. <br> <br>
*Toitū Te Whenua Land Information New Zealand, 2021. Auckland North LiDAR 1m DEM (2016-2018). https://registry.opendata.aws/nz-elevation/?utm* <br>  

* **R/patches.R** <br>
This script is used to sample the DTM and terrain parameters spatially to produce a training dataset used for analysis. This includes point samples of the terrain parameters and 2D raster samples of the DTM (i.e., "patches").

* **Python/ponui_terrain_attribute_learning.py** <br>
This script is used to train models to predict terrain parameters from DTM patches. The training dataset is generated using R/patches.R (above). The resulting predictions are output in tabular and spatial (raster) format.

* **R/results.R** <br>
This script is used to analyze the results. It loads in predictions, calculates performance statistics, runs a CART (rpart) model, and produces the figures presented in the manuscript.
