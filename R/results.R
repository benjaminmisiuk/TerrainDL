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

#define which model runs to load
filt <- c(16, 64, 128)
var <- c('qslope', 'qeastness', 'meanc', 'tpi', 'adjSD')
w <- c(3, 9, 27, 81)
n <- c(100, 1000, 10000)
d <- c(1, 2, 3)
act = 'PReLU'
scale = c(3, 9)
norm = c('normzero', 'global', 'sdmean') #normzero = local normalization, global = global normalization, sdmean = mean centering

#save the loaded results?
#this is recommended because it can take a long time to index all results
save = TRUE

#this creates a quarantine for results that did not converge
#so they can be isolated and explored
quarantine <- FALSE
quarantine_limit <- -0.5
quarantine_path <- 'D:/GIS/Ponui/quarantine'

#load and format results if they exist, otherwise read from csv and save
if(save & length(list.files('Rdata/', pattern = '.Rdata')) > 0){
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
                  df <- data.frame(obs=NA, pred1=NA, pred2=NA, pred3=NA)
                  file <- paste0('D:/GIS/Ponui/REVISION/results/scale', r, '_', s, '/', paste(j, '81', k, m, o, i, p, sep = '_'), '.csv')
                  try(
                    df <- read_csv(
                      file, 
                      col_names = FALSE,
                      show_col_types = FALSE
                    ),
                    silent = TRUE
                  )
                  
                  names(df) <- c('obs', 'pred1', 'pred2', 'pred3')
                  
                  r2_1 <- R2(df$obs, df$pred1)
                  r2_2 <- R2(df$obs, df$pred2)
                  r2_3 <- R2(df$obs, df$pred3)
                  
                  if(quarantine){
                    if(r2_1 < quarantine_limit | r2_2 < quarantine_limit | r2_3 < quarantine_limit | is.na(r2_1) | is.na(r2_2) | is.na(r2_3)){
                      message("Moving file to quarantine: ", file)
                      dir.create(paste0(quarantine_path, paste0('scale', r, '_', s, '/')), warnings = FALSE)
                      file.rename(file, paste0(quarantine_path, paste0('scale', r, '_', s, '/', paste(j, '81', k, m, o, i, p, sep = '_'), '.csv')))  
                    }
                  }
                  
                  l[[length(l) + 1]] <- data.frame(
                    var = j,
                    w = k,
                    n = m,
                    d = o,
                    filt = i,
                    act = p,
                    scale = r,
                    norm = s,
                    R2_1 = r2_1,
                    R2_2 = r2_2,
                    R2_3 = r2_3,
                    R2 = mean(c(r2_1, r2_2, r2_3), na.rm = TRUE),
                    SD = sd(c(r2_1, r2_2, r2_3), na.rm = TRUE)
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
    save(l, file = 'Rdata/results.Rdata')
  }
}

#this function allows for quickly subsetting the results by the desired hyperparameters
subsetter <- function(
    df,
    filt = c(16, 64, 128),
    var = c('qslope', 'qeastness', 'meanc', 'tpi', 'adjSD'),
    p = c(3, 9, 27, 81),
    n = c(100, 1000, 10000),
    d = c(1, 2, 3),
    act = 'PReLU',
    w = c(3, 9),
    norm = c('normzero', 'normmean', 'global', 'sdmean'),
    converge = c(0, 1)
){
  df <- df[df$filt %in% filt & df$var %in% var & df$p %in% p & df$n %in% n & df$d %in% d & df$act %in% act & df$w %in% w & df$norm %in% norm & df$converge %in% converge, ]
}

#fix names to match manuscript
names(l)[names(l) == 'w'] <- 'p'
names(l)[names(l) == 'scale'] <- 'w'
l$scale_diff <- abs(l$p - l$w) #calculate the absolute scale difference

#isolate runs that did not converge
fail <- l[l$R2_1 < -0.5 | l$R2_2 < -0.5 | l$R2_3 < -0.5 | is.na(l$R2_1) | is.na(l$R2_2) | is.na(l$R2_3), ]

#observe histogram of negative R2 values to see where non-conergence occurs
hist(l$R2_1[l$R2_1 < 0 & l$R2_1 > -2], breaks = 100)
hist(l$R2_2[l$R2_2 < 0 & l$R2_2 > -2], breaks = 100)
hist(l$R2_3[l$R2_3 < 0 & l$R2_3 > -2], breaks = 100)

#label non-converged runs
l$converge <- ifelse(l$R2_1 < -0.5 | l$R2_2 < -0.5 | l$R2_3 < -0.5, 0, 1)

#set non-converged runs to NA
l$R2_1[l$R2_1 < -0.5] <- NA
l$R2_2[l$R2_2 < -0.5] <- NA
l$R2_3[l$R2_3 < -0.5] <- NA

#calculate mean R2 and SD across the three runs
l$R2 <- apply(l[ ,c('R2_1', 'R2_2', 'R2_3')], 1, mean, na.rm = TRUE)
l$SD <- apply(l[ ,c('R2_1', 'R2_2', 'R2_3')], 1, sd, na.rm = TRUE)

#makre sure all NAs are coded properly
l[l == 'NaN'] <- NA


#### RPART ANALYSIS ####

library(rpart)
library(rpart.plot)
library(ggplot2)
library(pdp)
library(cowplot)

#subsetting
#normzero = local normalization
#global = global normalization
#sdmean = mean centering
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

#run rpart model for mean R2 score
mod <- rpart(R2 ~ n + d + filt + `|p-w|` + norm + var + p, data = df)

#individual rparts for each model run
mod1 <- rpart(R2_1 ~ n + d + filt + `|p-w|` + norm + var + p, data = df)
mod2 <- rpart(R2_2 ~ n + d + filt + `|p-w|` + norm + var + p, data = df)
mod3 <- rpart(R2_3 ~ n + d + filt + `|p-w|` + norm + var + p, data = df)

#varianlce importance stats for mean R2
raw_importance <- mod$variable.importance
root_ss <- mod$frame$dev[1]
root_imp <- raw_importance / root_ss
root_imp

imp <- data.frame(
  variable = names(mod$variable.importance),
  importance = as.numeric(mod$variable.importance),
  percent = as.numeric(mod$variable.importance) / sum(as.numeric(mod$variable.importance)) * 100,
  percent_root = root_imp*100
)
imp

#variable importance stats for each run
raw_importance1 <- mod1$variable.importance
raw_importance2 <- mod2$variable.importance
raw_importance3 <- mod3$variable.importance

root_ss1 <- mod1$frame$dev[1]
root_ss2 <- mod2$frame$dev[1]
root_ss3 <- mod3$frame$dev[1]

root_imp1 <- raw_importance1 / root_ss1
root_imp2 <- raw_importance2 / root_ss2
root_imp3 <- raw_importance3 / root_ss3

imp1 <- data.frame(
  variable = names(mod1$variable.importance),
  importance = as.numeric(mod1$variable.importance),
  percent = as.numeric(mod1$variable.importance) / sum(as.numeric(mod1$variable.importance)) * 100,
  percent_root = root_imp1*100
)

imp2 <- data.frame(
  variable = names(mod2$variable.importance),
  importance = as.numeric(mod2$variable.importance),
  percent = as.numeric(mod2$variable.importance) / sum(as.numeric(mod2$variable.importance)) * 100,
  percent_root = root_imp2*100
)

imp3 <- data.frame(
  variable = names(mod3$variable.importance),
  importance = as.numeric(mod3$variable.importance),
  percent = as.numeric(mod3$variable.importance) / sum(as.numeric(mod3$variable.importance)) * 100,
  percent_root = root_imp3*100
)

#calculate mean and SD for each variable across the three runs
imp$sd <- NA
imp$mean <- NA

for(i in imp$variable){
  imp1_i <- imp1$percent_root[imp1$variable == i]
  imp2_i <- imp2$percent_root[imp2$variable == i]
  imp3_i <- imp3$percent_root[imp3$variable == i]
  
  sd_i <- sd(c(imp1_i, imp2_i, imp3_i), na.rm = TRUE)
  mean_i <- mean(c(imp1_i, imp2_i, imp3_i), na.rm = TRUE)
  
  imp$sd[imp$variable == i] <- sd_i
  imp$mean[imp$variable == i] <- mean_i
}

imp

#this function formats the node labels for rpart results
format_R2 <- function(x, labs, digits, varlen) {
  paste0(sprintf("%.2f", x$frame$yval), "\n", round(x$frame$n/max(x$frame$n)*100), '%')
}

node_vars <- as.character(mod$frame$var)
split_col <- ifelse(node_vars == 'var', 'Black', 'Grey60')

#use the mean R2 for plotting
mod_plot <- mod
mod_plot$splits
mod_plot$splits[mod_plot$splits[, "index"] == 550, "index"] <- 1000
mod_plot$splits[mod_plot$splits[, "index"] == 5500, "index"] <- as.numeric(10000)

#customize the rpart plot
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

#partial dependence for mean R2 rpart model
pd_data <- partial(mod, pred.var = "n", train = df, ice = TRUE)
mean <- aggregate(pd_data$yhat, by = list(n = pd_data$n), FUN = mean)
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

#sample size PDP
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

#variable PDP
pd_data <- partial(mod, pred.var = "var", train = df, ice = TRUE)
mean <- aggregate(pd_data$yhat, by = list(var = pd_data$var), FUN = mean)
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

#scale difference PDP
pd_data <- partial(mod, pred.var = "|p-w|", train = df, ice = TRUE)
mean <- aggregate(pd_data$yhat, by = list(`|p-w|` = pd_data$`|p-w|`), FUN = mean)
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

#normalization PDP
pd_data <- partial(mod, pred.var = "norm", train = df, ice = TRUE)
mean <- aggregate(pd_data$yhat, by = list(norm = pd_data$norm), FUN = mean)
q25 <- aggregate(pd_data$yhat, by = list(norm = pd_data$norm), FUN = function(x) quantile(x, 0.25))
q75 <- aggregate(pd_data$yhat, by = list(norm = pd_data$norm), FUN = function(x) quantile(x, 0.75))
q <- merge(q25, q75, by = 'norm')
names(q) <- c('norm', 'q25', 'q75')
pd_data <- merge(mean, q, by = 'norm')
names(pd_data) <- c('norm', 'yhat', 'q25', 'q75')
d <- ggplot(pd_data, aes(x = reorder(norm, yhat), y = yhat)) +
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


#### GRAPHICAL RESULTS ####

#look at spatial scale of data and sample size vs R2, for each variable
#ie Figure 6
df <- subsetter(df = l, filt = 64, d = 2, norm = c('normzero', 'global', 'sdmean'), w = 9)
df$var <- factor(df$var, levels = c('adjSD', 'qeastness', 'meanc', 'qslope', 'tpi'), labels = c("AdjSD", "Eastness", "Mean curve", "Slope", "TPI"))
df$norm <- factor(df$norm, levels = c('global', 'normzero', 'sdmean'), labels = c("Global", "Local", "Mean centre"))
ggplot(df, aes(x = p-w, y = R2, col = factor(norm), shape = factor(norm))) +
  facet_grid(var ~ n, labeller = labeller(n = label_both, w = label_both)) +
  geom_abline(intercept=0, slope = 0, col = 'black') +
  geom_vline(xintercept = 0) +
  geom_line() +
  geom_point(size = 2.5) +
  geom_segment(aes(x = p-w, xend = p-w, y = R2 - SD, yend = R2 + SD), size = 0.5) +
  geom_point(data = df[(df$p == 9 | df$p == 81) & df$var == 'TPI' & df$norm == 'Mean centre' & df$n == 1000, ], size = 2.5, col = 'red', show.legend = FALSE) +
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
    plot.title = element_text(hjust = 0),
    panel.grid.minor.x = element_blank()
  )
ggsave('../figs/part3_scale_diff_9.png', width = 6, height = 6, dpi = 300)

#R2 values for individual runs corresponding to figs 6, 7
df[df$var == 'TPI' & df$norm == 'Mean centre' & df$n == 1000 & (df$p == 9 | df$p == 81), ]

#generally, which variables are easy vs hard to predict, and at which sample size and with which norm?
#ie, Figure 8
df <- subsetter(df = l, filt = 64, d = 2, norm = c('normzero', 'global', 'sdmean'), w = 3, p = 3)
df$var <- factor(df$var, levels = c('adjSD', 'qeastness', 'meanc', 'qslope', 'tpi'), labels = c("AdjSD", "Eastness", "Mean curve", "Slope", "TPI"))
df$norm <- factor(df$norm, levels = c('global', 'normzero', 'sdmean'), labels = c("Global", "Local", "Mean centre"))

ggplot(df, aes(x = n, y = R2, col = factor(norm), shape = factor(norm))) +
  facet_wrap(~var, ncol = 1, labeller = labeller(var = label_value), strip.position = 'right') +
  geom_abline(intercept=0, slope = 0, col = 'black') +
  geom_line() +
  geom_point(size = 2.5) +
  geom_point(data = df[(df$norm == 'Mean centre' | df$norm == 'Global') & df$var == 'Slope' & df$n == 100, ], size = 2.5, col = 'red', show.legend = FALSE) +
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
    plot.title = element_text(hjust = 0),
    panel.grid.minor.x = element_blank()
  )
ggsave('../figs/part3_norm_3.png', width = 4, height = 6, dpi = 300)

#R2 values for individual runs corresponding to figs 8, 9
df[df$var == 'Slope' & (df$norm == 'Mean centre' | df$norm == 'Global') & df$n == 100, ]


