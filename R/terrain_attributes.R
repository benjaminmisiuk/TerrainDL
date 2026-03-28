library(MultiscaleDTM)
library(terra)

#load the raster
r <- rast('D:/GIS/Ponui/layers/ponui_dtm_setnull_5m_ext.tif')
plot(r)

#SCALE 3

#qfit
q3 <- Qfit(r)
writeRaster(q3, paste0('D:/GIS/Ponui/layers/', names(q3), '_3.tif'))

#adjsd
sd3 <- AdjSD(r)
writeRaster(sd3, paste0('D:/GIS/Ponui/layers/', names(sd3), '_3.tif'))

#TPI
tpi3 <- TPI(r)
writeRaster(tpi3, paste0('D:/GIS/Ponui/layers/', names(tpi3), '_3.tif'))

#SCALE 9

#qfit
q9 <- Qfit(r, w=9)
writeRaster(q9, paste0('D:/GIS/Ponui/layers/', names(q9), '_9.tif'))

#adjsd
sd9 <- AdjSD(r, w=9)
writeRaster(sd9, paste0('D:/GIS/Ponui/layers/', names(sd9), '_9.tif'))

#TPI
tpi9 <- TPI(r, w=9)
writeRaster(tpi9, paste0('D:/GIS/Ponui/layers/', names(tpi9), '_9.tif'))
