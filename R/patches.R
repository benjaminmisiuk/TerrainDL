library(terra)

#choose the scale of terrain attribute to load in
w=9

#load DTM raster
dem <- rast('D:/GIS/Ponui/layers/ponui_dtm_setnull_5m.tif')

#load terrain attributes calculated with "terrain_attributes.R" script
stack <- c(
  rast(paste0('D:/GIS/Ponui/layers/qslope_', w, '.tif')),
  rast(paste0('D:/GIS/Ponui/layers/qeastness_', w, '.tif')),
  rast(paste0('D:/GIS/Ponui/layers/meanc_', w, '.tif')),
  rast(paste0('D:/GIS/Ponui/layers/tpi_', w, '.tif')),
  rast(paste0('D:/GIS/Ponui/layers/adjsd_', w, '.tif'))
)

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
p$sample <- 1:length(p)
for(i in 1:length(p)){
  cat(i, '\r', end = '   ')
  
  p_i <- p[i]
  r_i <- trim(rasterize(p_i, stack[[1]]))
  e <- extend(r_i, (dim-1)/2, fill = 0)
  m <- crop(dem, e)
  #expand m to match e
  if(nrow(m) < nrow(e) | ncol(m) < ncol(e)){
    m <- extend(m, e, fill = NA)
  }
  
  #save the DTM patch
  writeRaster(m, paste0(out_dir, '/sample_', i, '.tif'))
}

#save all sample points
write.csv(p, paste0('D:/GIS/Ponui/points/ponui_11000_', dim, '_', w, '.csv'), row.names = FALSE)
