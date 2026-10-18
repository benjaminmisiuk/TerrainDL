library(terra)

#choose the scale of terrain attribute to load in
w=9

#load DTM raster
dem <- rast('D:/GIS/Ponui/REVISION/layers/ponui_dtm_setnull_agg5.tif')
dir <- 'D:/GIS/Ponui/REVISION/layers/'

#visualize
plot(stack)

#take a random sample of points
p <- spatSample(stack, 11000, na.rm = TRUE, as.points = TRUE)
plot(dem)
points(p, col = 'red')

#extract raster patches at points
dim <- 81 #set size of patches
out_dir <- 'D:/GIS/Ponui/patches/w_9' #choose an output directory

#these lines perform one iteration of "patchification" for sanity check
p_i <- p[sample(1:length(p), 1)]
r_i <- trim(rasterize(p_i, dem))
e <- extend(r_i, (dim-1)/2, fill = 0)
e
plot(e)
points(p_i)

m <- crop(dem, e)
plot(m)
points(p)

#here we run the full patchification for all points
for(w in c(3, 9)){
  #load terrain attributes calculated with "terrain_attributes.R" script
  stack <- c(
    rast(paste0(dir, 'qslope_', w, '.tif')),
    rast(paste0(dir, 'qeastness_', w, '.tif')),
    rast(paste0(dir, 'meanc_', w, '.tif')),
    rast(paste0(dir, 'tpi_', w, '.tif')),
    rast(paste0(dir, 'adjsd_', w, '.tif'))
  )
  
  #take a random sample of points
  p <- spatSample(stack, 11000, na.rm = TRUE, as.points = TRUE)
  
  #extract raster patches at points
  dim <- 81
  out_dir <- paste0('D:/GIS/Ponui/REVISION/patches/w_', w)
  dir.create(file.path(out_dir), showWarnings = FALSE)
  
  for(i in 1:length(p)){
    cat(i, '\r', end = '   ')
    
    #create and resize DEM patches
    p_i <- p[i]
    r_i <- trim(rasterize(p_i, stack[[1]]))
    e <- extend(r_i, (dim-1)/2, fill = 0)
    m <- crop(dem, e)
    #expand m to match e
    if(nrow(m) < nrow(e) | ncol(m) < ncol(e)){
      m <- extend(m, e, fill = NA)
    }
    
    #save the patch
    writeRaster(m, paste0(out_dir, '/sample_', i, '.tif'))
  }
  
  #save the points
  points_dir <- paste0('D:/GIS/Ponui/REVISION/points/')
  dir.create(file.path(points_dir), showWarnings = FALSE)
  write.csv(p, paste0(points_dir, 'ponui_11000_', dim, '_', w, '.csv'), row.names = FALSE)
}