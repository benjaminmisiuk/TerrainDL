library(MultiscaleDTM)
library(terra)

#load the raster and expand
r <- rast('D:/GIS/Ponui/REVISION/layers/ponui_dtm_setnull_agg5.tif')
plot(r)

r <- extend(r, 100)

dir <- 'D:/GIS/Ponui/REVISION/layers'

writeRaster(r, paste0(dir, 'ponui_dtm_setnull_agg5_ext.tif'))

#SCALE 3

#qfit
q3 <- Qfit(r)
writeRaster(q3, paste0(dir, names(q3), '_3.tif'))

#adjsd
sd3 <- AdjSD(r)
writeRaster(sd3, paste0(dir, names(sd3), '_3.tif'))

#TPI
tpi3 <- TPI(r)
writeRaster(tpi3, paste0(dir, names(tpi3), '_3.tif'))

#SCALE 9

#qfit
q9 <- Qfit(r, w=9)
writeRaster(q9, paste0(dir, names(q9), '_9.tif'))

#adjsd
sd9 <- AdjSD(r, w=9)
writeRaster(sd9, paste0(dir, names(sd9), '_9.tif'))

#TPI
tpi9 <- TPI(r, w=9)
writeRaster(tpi9, paste0(dir, names(tpi9), '_9.tif'))
