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

#if not set
#setwd('C:/Users/benja/Documents/GitHub/terrain_learning/R')

#define which model runs to load
filt <- c(16, 32, 64, 128)
var <- c('qslope', 'qeastness', 'meanc', 'tpi', 'adjSD')
w <- c(3, 9, 27, 81)
n <- c(100, 1000, 10000)
d <- c(1, 2, 3)
act = 'PReLU'
scale = c(3, 9)
norm = c('normzero', 'normmean', 'global', 'sdmean')

#save the loaded results?
#this is recommended because it can take a long time
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


#### RPART ANALYSIS ####

library(rpart)
library(rpart.plot)
library(ggplot2)
library(pdp)
library(cowplot)

hist(l$R2, xlim = c(-10,1), breaks = 1000)

#subset if necessary
df <- subsetter(df = l, norm = c('normzero', 'global', 'sdmean'))

#set factor levels for categorical hyperparameters
df$var <- factor(
  df$var, 
  levels = c('adjSD', 'meanc', 'qeastness', 'qslope', 'tpi'),
  labels = c('AdjSD', 'Curve', 'East', 'Slope', 'TPI')
)
df$norm <- factor(
  df$norm,
  levels = c('global', 'normzero', 'sdmean'),
  labels = c('GlobNorm', 'LocNorm', 'MeanCentre')
)

names(df)[names(df) == 'scale_diff'] <- '|p-w|'

#run rpart and visualize
mod <- rpart(R2 ~ n + d + filt + `|p-w|` + norm + w + var, data = df)
printcp(mod)
rsq.rpart(mod)

raw_importance <- mod$variable.importance
root_ss <- mod$frame$dev[1]
root_imp <- raw_importance / root_ss
root_imp

#create a table for variable importance
imp <- data.frame(
  variable = names(mod$variable.importance),
  importance = as.numeric(mod$variable.importance),
  percent = as.numeric(mod$variable.importance) / sum(as.numeric(mod$variable.importance)) * 100,
  percent_root = root_imp*100
)
imp
#write.csv(imp, '../results/rpart_all_importance.csv', row.names = FALSE)

#this function formats the node labels for rpart results
format_R2 <- function(x, labs, digits, varlen) {
  paste0(sprintf("%.2f", x$frame$yval), "\n", round(x$frame$n/max(x$frame$n)*100), '%')
}

node_vars <- as.character(mod$frame$var)
split_col <- ifelse(node_vars == 'var', 'Black', 'Grey60')

mod_plot <- mod
mod_plot$splits
mod_plot$splits[mod_plot$splits[, "index"] == 550, "index"] <- 1000
mod_plot$splits[mod_plot$splits[, "index"] == 5500, "index"] <- as.numeric(10000)

#here we can tweak the rpart plot
rpart.plot(mod_plot, 
           type = 4, 
           Margin = 0,
           space = 0.75,
           compress = TRUE,
           extra = 100, 
           clip.facs = TRUE,
           facsep = ';',
           node.fun = format_R2,
           digits = -1,
           border.col = 0,
           lwd = 4,
           branch.lwd = (mod_plot$frame$n / max(mod_plot$frame$n)*100)^(1/2),
           split.col = split_col,
           box.palette = "OrBu",
           pal.thresh = 0.5
           #split.lwd = 10
           )

#save the final figure
png(paste0('../figs/rpart_all.png'), width = 7.25, height = 4.5, units = 'in', res = 300, type = 'windows', bg = 'transparent')
par(mar = c(0,0,0,0))
rpart.plot(mod_plot, 
           type = 4, 
           Margin = 0,
           space = 0.75,
           compress = TRUE,
           extra = 100, 
           clip.facs = TRUE,
           facsep = ';',
           node.fun = format_R2,
           digits = -1,
           border.col = 0,
           lwd = 4,
           branch.lwd = (mod_plot$frame$n / max(mod_plot$frame$n)*100)^(1/2),
           split.col = split_col,
           box.palette = "OrBu",
           pal.thresh = 0.5,
           split.lwd = 10)
dev.off()

#partial dependence plots for the most important variables

#first calculate partial dependence mean and quantiles
pd_data <- partial(mod, pred.var = "n", train = df, ice = TRUE)
mean <- aggregate(pd_data$yhat, by = list(n = pd_data$n), FUN = median)
q25 <- aggregate(pd_data$yhat, by = list(n = pd_data$n), FUN = function(x) quantile(x, 0.25))
q75 <- aggregate(pd_data$yhat, by = list(n = pd_data$n), FUN = function(x) quantile(x, 0.75))
q <- merge(q25, q75, by = 'n')
names(q) <- c('n', 'q25', 'q75')
pd_data <- merge(mean, q, by = 'n')
names(pd_data) <- c('n', 'yhat', 'q25', 'q75')
rows <- nrow(pd_data)
segs <- data.frame(
  x=pd_data$n[1:(rows)],
  xend=c(pd_data$n[2:rows], pd_data$n[rows]+max(pd_data$n/10)),
  y=pd_data$yhat[1:(rows)],
  y_pos=pd_data$q75[1:(rows)],
  y_neg=pd_data$q25[1:(rows)]
)

#plot the partial dependence information for each variable (a-d)
color_error <- '#ffb09c'

a <- ggplot(segs, aes(x = x, y = y)) +
  geom_hline(yintercept = 0, col = 'black') +
  geom_segment(aes(x=x, xend=xend, y=y_pos, yend=y_pos), col = color_error, size=0.6, lty=7) +
  geom_segment(aes(x=x, xend=xend, y=y_neg, yend=y_neg), col = color_error, size=0.6, lty=7) +
  geom_segment(aes(x=x, xend=xend, y=y, yend=y), size = 1) +
  coord_cartesian(xlim = c(0, 11000), ylim = c(-0.15, 1), expand = FALSE) +
  labs(title = "Sample size (n)",
       x = NULL,
       y = expression(R^2)) +
  theme_minimal()+
  theme(
    text = element_text(size = 8),
    axis.title = element_text(size = 10),
    axis.line.x = element_blank(),
    axis.line.y = element_line(color = 'black', size = 0.5),
    legend.position = 'none'
  ); a

pd_data <- partial(mod, pred.var = "var", train = df, ice = TRUE)
mean <- aggregate(pd_data$yhat, by = list(var = pd_data$var), FUN = median)
q25 <- aggregate(pd_data$yhat, by = list(var = pd_data$var), FUN = function(x) quantile(x, 0.25))
q75 <- aggregate(pd_data$yhat, by = list(var = pd_data$var), FUN = function(x) quantile(x, 0.75))
q <- merge(q25, q75, by = 'var')
names(q) <- c('var', 'q25', 'q75')
pd_data <- merge(mean, q, by = 'var')
names(pd_data) <- c('var', 'yhat', 'q25', 'q75')
pd_data[order(pd_data$yhat), ]
pd_data$var <- factor(pd_data$var, levels = c('AdjSD', 'Curve', 'TPI', 'Slope', 'East'))
b <- ggplot(pd_data, aes(x = var, y = yhat)) +
  geom_linerange(aes(ymin = q25, ymax = q75), col = color_error, lwd=0.6, lty=7) +
  geom_point(size = 2, col = 'grey30') +
  #geom_col(width = 0.7, fill = 'white', col = 'grey30', lwd=1) +
  coord_cartesian(xlim = c(0.5, 5.5), ylim = c(0, 1), expand = FALSE) +
  labs(title = "Terrain parameter",
       x = NULL,
       y = expression(R^2)) +
  theme_minimal()+
  theme(
    text = element_text(size = 8),
    axis.title = element_text(size = 10),
    axis.line.x = element_line(color = 'black', size = 0.5),
    axis.line.y = element_line(color = 'black', size = 0.5),
    legend.position = 'none'
  ); b

pd_data <- partial(mod, pred.var = "|p-w|", train = df, ice = TRUE)
mean <- aggregate(pd_data$yhat, by = list(`|p-w|` = pd_data$`|p-w|`), FUN = median)
q25 <- aggregate(pd_data$yhat, by = list(`|p-w|` = pd_data$`|p-w|`), FUN = function(x) quantile(x, 0.25))
q75 <- aggregate(pd_data$yhat, by = list(`|p-w|` = pd_data$`|p-w|`), FUN = function(x) quantile(x, 0.75))
q <- merge(q25, q75, by = '|p-w|')
names(q) <- c('|p-w|', 'q25', 'q75')
pd_data <- merge(mean, q, by = '|p-w|')
names(pd_data) <- c('|p-w|', 'yhat', 'q25', 'q75')
rows <- nrow(pd_data)
segs <- data.frame(
  x=pd_data$`|p-w|`[1:(rows)],
  xend=c(pd_data$`|p-w|`[2:rows], pd_data$`|p-w|`[rows]+max(pd_data$`|p-w|`/10)),
  y=pd_data$yhat[1:(rows)],
  y_pos=pd_data$q75[1:(rows)],
  y_neg=pd_data$q25[1:(rows)]
)
c <- ggplot(segs, aes(x = n, y = yhat)) +
  geom_hline(yintercept = 0, col = 'black') +
  geom_segment(aes(x=x, xend=xend, y=y, yend=y), size = 1) +
  geom_segment(aes(x=x, xend=xend, y=y_pos, yend=y_pos), col = color_error, size=0.6, lty=7) +
  geom_segment(aes(x=x, xend=xend, y=y_neg, yend=y_neg), col = color_error, size=0.6, lty=7) +
  coord_cartesian(xlim = c(0, 80), ylim = c(-0.15, 1), expand = FALSE) +
  labs(title = "Scale difference (|p-w|)",
       x = NULL,
       y = expression(R^2)) +
  theme_minimal()+
  theme(
    text = element_text(size = 8),
    axis.title = element_text(size = 10),
    axis.line.x = element_blank(),
    axis.line.y = element_line(color = 'black', size = 0.5),
    legend.position = 'none'
  ); c

pd_data <- partial(mod, pred.var = "norm", train = df, ice = TRUE)
mean <- aggregate(pd_data$yhat, by = list(norm = pd_data$norm), FUN = median)
q25 <- aggregate(pd_data$yhat, by = list(norm = pd_data$norm), FUN = function(x) quantile(x, 0.25))
q75 <- aggregate(pd_data$yhat, by = list(norm = pd_data$norm), FUN = function(x) quantile(x, 0.75))
q <- merge(q25, q75, by = 'norm')
names(q) <- c('norm', 'q25', 'q75')
pd_data <- merge(mean, q, by = 'norm')
names(pd_data) <- c('norm', 'yhat', 'q25', 'q75')
d <- ggplot(pd_data, aes(x = reorder(norm, yhat), y = yhat)) +
  #geom_col(width = 0.7, fill = 'white', col = 'grey30', lwd=1) +
  geom_linerange(aes(ymin = q25, ymax = q75), col = color_error, lwd=0.6, lty=7) +
  geom_point(size = 2, col = 'grey30') +
  coord_cartesian(xlim = c(0.5, 3.5), ylim = c(0, 1), expand = FALSE) +
  labs(title = "Normalization",
       x = NULL,
       y = expression(R^2)) +
  theme_minimal()+
  theme(
    text = element_text(size = 8),
    axis.title = element_text(size = 10),
    axis.line.x = element_line(color = 'black', size = 0.5),
    axis.line.y = element_line(color = 'black', size = 0.5),
    legend.position = 'none'
  ); d

#plot all partial dependence together and save
plot_grid(a, b, c, d, ncol = 1)
ggsave('../figs/pdp_all_quantiles_red.png', width = 3.25, height = 5, dpi = 300)

l$R2[l$var == 'tpi' & l$norm == 'sdmean' & l$n == 1000 & l$d == 2 & l$filt == 32 & l$scale == 9 & l$w == 9]
l$R2[l$var == 'tpi' & l$norm == 'sdmean' & l$n == 1000 & l$d == 2 & l$filt == 32 & l$scale == 9 & l$w == 81]
l$R2[l$var == 'qslope' & l$norm == 'sdmean' & l$n == 100 & l$d == 2 & l$filt == 32 & l$scale == 3 & l$w == 3]
l$R2[l$var == 'qslope' & l$norm == 'global' & l$n == 100 & l$d == 2 & l$filt == 32 & l$scale == 3 & l$w == 3]


#### GRAPHICAL RESULTS ####

#look at spatial scale of data and sample size vs R2, for each variable
#ie Figure 6
df <- subsetter(df = l, filt = 32, d = 2, norm = c('normzero', 'global', 'sdmean'), scale = 9)
df$var <- factor(df$var, levels = c('adjSD', 'qeastness', 'meanc', 'qslope', 'tpi'), labels = c("AdjSD", "Eastness", "Mean curve", "Slope", "TPI"))
df$norm <- factor(df$norm, levels = c('global', 'normzero', 'sdmean'), labels = c("Global", "Local", "Mean centre"))
ggplot(df, aes(x = w-scale, y = R2, col = factor(norm), shape = factor(norm))) +
  facet_grid(var ~ n, labeller = labeller(n = label_both, w = label_both)) +
  geom_abline(intercept=0, slope = 0, col = 'black') +
  geom_vline(xintercept = 0) +
  geom_line() +
  geom_point(size = 2.5) +
  geom_point(data = df[(df$w == 9 | df$w == 81) & df$var == 'TPI' & df$norm == 'Mean centre' & df$n == 1000, ], size = 2.5, col = 'red', show.legend = FALSE) +
  scale_x_continuous(breaks = c(-6, 0, 18, 72)) +
  labs(
    x = 'Scale difference (p-w)',
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
    panel.grid.minor.x = element_blank()
  )
ggsave('../figs/part3_scale_diff_9.png', width = 6, height = 6, dpi = 300)

#generally, which variables are easy vs hard to predict, and at which sample size and with which norm?
#ie, Figure 8
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
ggsave('../figs/part3_norm_3.png', width = 4, height = 6, dpi = 300)