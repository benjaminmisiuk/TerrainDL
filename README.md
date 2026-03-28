# TerrainDL
This repository provides the code used for the analysis detailed in the manuscript **Evaluating the deep learning of terrain information from elevation data**, by Benjamin Misiuk, Alexandre Schimel, and Vincent Lecours.

### Code
This repository includes both R and Python code. The scripts are described below, and are generally run in this order to reproduce the analysis.

* **R/terrain_attributes.R** <br>
This script is used to generate terrain parameters from the resampled 5 m Ponui Island DTM. The DTM is made available by Toitū Te Whenua Land Information New Zealand. <br> <br>
*Toitū Te Whenua Land Information New Zealand, 2021. Auckland North LiDAR 1m DEM (2016-2018). https://registry.opendata.aws/nz-elevation/* <br>  

* **R/patches.R** <br>
This script is used to sample the DTM and terrain parameters spatially to produce a training dataset used for analysis. This includes point samples of the terrain parameters and 2D raster samples of the DTM (i.e., "patches").

* **Python/ponui_terrain_attribute_learning.py** <br>
This script is used to train models to predict terrain parameters from DTM patches. The training dataset is generated using R/patches.R (above). The resulting predictions are output in tabular and spatial (raster) format.

* **R/results.R** <br>
This script is used to analyze the results. It loads in predictions, calculates performance statistics, runs a CART (rpart) model, and produces the figures presented in the manuscript.

* **R/Rdata/results.Rdata** <br>
These are the results from our analysis. They may be opened with R and exported to a tabular format, or can be viewed and analyzed within R.

### Usage
Follow these steps to run the code and reproduce our results.

1. Download the DTM data available at: https://registry.opendata.aws/nz-elevation/. Resample the 1 m grid to 5 m using mean aggregation (this can be completed in ArcGIS, QGIS, or with the terra package in R). Expand the extent of the 5 m DTM by 100 cells on all sides. We do not provide the original DTM here because of file size, but the 5 m DTM is contained within data/ponui_dtm_setnull_5m.tif <br>

2. Open the R/terrain_attributes.R script. Indicate the file path to the 5 m raster, and the output filepaths at which to save the terrain parameters. Run the script. <br>

#SHOW EXAMPLES

3. Open the R/patches.R script. Indicate the file path to the 5 m DTM and the terrain parameters from the previous step. Provide filepaths for the outputs. Run the script in order to generate spatially random samples over the extent of the rasters, and to extract elevation patches around each sampled point. These are the inputs to the CNN models. <br>

#ADD EXAMPLE FROM THE SCRIPT THAT SHOWS THE SINGLE PATCHIFYING

4. Open Python/ponui_terrain_attribute_learning.py. Configure the input and output directories and hyperparameters for a given set of model runs. Suggest starting with a single set of hyper-parameters to ensure the code works. This is configured near the beginning of the script like below.

```
# these are hyperparameters that are loopable
sizes = [81] # size of the input
depth = [1]  # depth of the CNN
filters = [16]  # number of filters
response = ['qslope', 'qeastness', 'meanc', 'tpi', 'adjSD'] # target variables
```

5. The outputs from Python/ponui_terrain_attribute_learning.py will write to the directories specified in the beginning setup section of the code. We choose to read and analyze these in R, but this can also be accomplished with Python or another language. Use the R/results.R script to read and plot the results from the directories specified in step 4 (example below). Note that all results have been read and formatted already and are provided within R/Rdata/results.Rdata. Note that the script loads all results by default like so:

```
library(readr)
library(viridis)
library(RColorBrewer)
library(cmna)
library(ggplot2)

#R2 function
R2 <- function(y, y_h, na.rm = FALSE){
  if(na.rm){
    na <- is.na(y_h)|is.na(y)
    y_h <- y_h[!na]
    y <- y[!na]
  }
  
  SSres = sum((y - y_h)^2)
  SStot = sum((y - mean(y))^2)
  1 - (SSres/SStot)
}

save = TRUE

#load and format results if they exist, otherwise read from csv and save
if(save & length(list.files('Rdata/')) > 0){
  load('Rdata/results.Rdata')
} else {
  l <- list()
  for(i in filt){
    for(j in var){
      for(k in w){
        for(m in n){
          for(o in d){
            for(p in act){
              for(r in scale){
                for(s in norm){
                  df <- read_csv(
                    paste0('D:/GIS/Ponui/results/scale', r, '_', s, '/', paste(j, '81', k, m, o, i, p, sep = '_'), '.csv'), 
                    col_names = FALSE,
                    show_col_types = FALSE
                  )
                  
                  names(df) <- c('obs', 'pred')
                  
                  l[[length(l) + 1]] <- data.frame(
                    var = j,
                    w = k,
                    n = m,
                    d = o,
                    filt = i,
                    act = p,
                    scale = r,
                    norm = s,
                    R2 = R2(df$obs, df$pred), 
                    RMSE = RMSE(df$obs, df$pred)
                  )
                  rm(df)
                }
              }
            }
          }
        }
      }
    }
  }
  l <- do.call(rbind, l)
  if(save){
    save(l, file = 'D:/GitHub/terrain_learning/R/Rdata/results.Rdata')
  }
}

#this function allows for quickly subsetting the results by the desired hyperparameters
subsetter <- function(
    df,
    filt = c(16, 32, 64, 128),
    var = c('qslope', 'qeastness', 'meanc', 'tpi', 'adjSD'),
    w = c(3, 9, 27, 81),
    n = c(100, 1000, 10000),
    d = c(1, 2, 3),
    act = 'PReLU',
    scale = c(3, 9),
    norm = c('normzero', 'normmean', 'global', 'sdmean')
){
  df <- df[df$filt %in% filt & df$var %in% var & df$w %in% w & df$n %in% n & df$d %in% d & df$act %in% act & df$scale %in% scale & df$norm %in% norm, ]
}

l$scale_diff <- abs(l$w - l$scale) #calculate the absolute scale difference
l$R2[l$R2 < -2] <- NA #consider any R2 values less than -2 to be non-converged
```

After the results have been loaded, any parts of the analysis can be performed by using the `subsetter()` function to select a slice of the results to view. An example below shows the generation of Figure 8 in the manuscript.

```
df <- subsetter(df = l, filt = 32, d = 2, norm = c('normzero', 'global', 'sdmean'), scale = 3, w = 3)
df$var <- factor(df$var, levels = c('adjSD', 'qeastness', 'meanc', 'qslope', 'tpi'), labels = c("AdjSD", "Eastness", "Mean curve", "Slope", "TPI"))
df$norm <- factor(df$norm, levels = c('global', 'normzero', 'sdmean'), labels = c("Global", "Local", "Mean centre"))
ggplot(df, aes(x = n, y = R2, col = factor(norm), shape = factor(norm))) +
  facet_wrap(~var, ncol = 1, labeller = labeller(var = label_value), strip.position = 'right') +
  geom_abline(intercept=0, slope = 0, col = 'black') +
  geom_line() +
  geom_point(size = 2.5) +
  geom_point(data = df[(df$norm == 'Mean centre' | df$norm == 'Global') & df$var == 'Slope' & df$w == 3 & df$n == 100, ], size = 2.5, col = 'red', show.legend = FALSE) +
  scale_x_continuous(breaks = c(100, 1000, 10000)) +
  labs(
    x = 'Sample size (n)',
    y = expression(R^2),
    color = 'Normalization',
    shape = 'Normalization'
  ) +
  ylim(-0.5, 1) +
  scale_color_grey(start = 0.2, end = 0.75) +
  theme_minimal() + 
  theme(
    text = element_text(size = 8),
    axis.title = element_text(size = 10),
    axis.line.y = element_line(color = 'black', size = 0.5),
    panel.grid.minor.x = element_blank()
  )
```

<img width="450" height="675" alt="part3_norm_3" src="https://github.com/user-attachments/assets/86e24084-bbaf-4b26-a66e-413d11a8036d" />
