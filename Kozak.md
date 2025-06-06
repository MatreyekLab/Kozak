ACE2_Kozak
================
Kenneth Matreyek
1/8/2021

``` r
## Clear existing variables, load required packages, set the seed and base theme.
rm(list = ls())
library(reshape2)
library(ggrepel)
```

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 4.2.3

``` r
library(ggbeeswarm)
library(patchwork)
library(tidyverse)
```

    ## Warning: package 'tidyr' was built under R version 4.2.3

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.2     ✔ readr     2.1.4
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.0
    ## ✔ lubridate 1.9.2     ✔ tibble    3.2.1
    ## ✔ purrr     1.0.2     ✔ tidyr     1.3.1

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(data.table)
```

    ## 
    ## Attaching package: 'data.table'
    ## 
    ## The following objects are masked from 'package:lubridate':
    ## 
    ##     hour, isoweek, mday, minute, month, quarter, second, wday, week,
    ##     yday, year
    ## 
    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     between, first, last
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     transpose
    ## 
    ## The following objects are masked from 'package:reshape2':
    ## 
    ##     dcast, melt

``` r
library(randomForest)
```

    ## randomForest 4.7-1.1
    ## Type rfNews() to see new features/changes/bug fixes.
    ## 
    ## Attaching package: 'randomForest'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine
    ## 
    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(MASS)
```

    ## 
    ## Attaching package: 'MASS'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     select
    ## 
    ## The following object is masked from 'package:patchwork':
    ## 
    ##     area

``` r
#install.packages("plotly")
library(plotly)
```

    ## 
    ## Attaching package: 'plotly'
    ## 
    ## The following object is masked from 'package:MASS':
    ## 
    ##     select
    ## 
    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     last_plot
    ## 
    ## The following object is masked from 'package:stats':
    ## 
    ##     filter
    ## 
    ## The following object is masked from 'package:graphics':
    ## 
    ##     layout

``` r
#install.packages("htmlwidgets")
library(htmlwidgets)
set.seed(1234567)
theme_set(theme_bw())

## Setting some universal thresholds for some of the subsequent analyses
quantile_cutoff <- 0.99
minquant_fraction <- 0.2

## Setings for knitting the document into a Markdown file
knitr::opts_chunk$set(
  warning = FALSE, # DO NOT show warnings
  message = FALSE, # DO NOT show messages
  error = FALSE, # interrupt generation in case of errors,
  echo = TRUE  # show R code
)
```

``` r
## Let's first make a dataframe of all of the possible Kozak sequences
complete_frame <- data.frame("index" = seq(1,4^6), "sequence" = "")
nucleotide_list <- c("A","C","G","T")
x = 1
for(p1 in nucleotide_list){
  for(p2 in nucleotide_list){
    for(p3 in nucleotide_list){
      for(p4 in nucleotide_list){
        for(p5 in nucleotide_list){
          for(p6 in nucleotide_list){
            complete_frame$sequence[x] <- paste(p1,p2,p3,p4,p5,p6,"ATG",sep="")
            x = x + 1
          }
        }
      }
    }
  }
}
```

``` r
## Importing the data from the Illumina Nextseq kits
myfiles1 = list.files(path="Data/NextSeq001", pattern="*.tsv", full.names=TRUE)
myfiles2 = list.files(path="Data/NextSeq002", pattern="*.tsv", full.names=TRUE)
myfiles3 = list.files(path="Data/NextSeq003", pattern="*.tsv", full.names=TRUE)
myfiles4 = list.files(path="Data/NextSeq004", pattern="*.tsv", full.names=TRUE)
myfiles6 = list.files(path="Data/NextSeq006", pattern="*.tsv", full.names=TRUE)
myfiles = c(myfiles1, myfiles2,myfiles3,myfiles4,myfiles6)

ns1_template <- read.csv(file = "Data/NextSeq1_template.csv", header = T, stringsAsFactors = F)
ns2_template <- read.csv(file = "Data/NextSeq2_template.csv", header = T, stringsAsFactors = F)
ns3_template <- read.csv(file = "Data/NextSeq3_template.csv", header = T, stringsAsFactors = F)
ns4_template <- read.csv(file = "Data/NextSeq4_template.csv", header = T, stringsAsFactors = F)
template <- rbind(ns1_template, ns2_template, ns3_template, ns4_template)
```

``` r
## Making a function for analyzing the four-way sorting data
#list_of_indexes <- c(72,73,74,75,76,77,78,79) ## Delete whenever. Only for troubleshooting.

fourWaySortAnalysis <- function(list_of_indexes){
  ## Bin 1
  rep1 <- read.delim(myfiles[list_of_indexes[1]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count))
  rep1_density_table <- ggplot_build(rep1_density)$data[[1]]
  rep1_minquant <- rep1_density_table %>% filter(x < quantile(rep1_density_table$x, minquant_fraction))
  rep1_density_minima <- rep1_minquant[rep1_minquant$y == min(rep1_minquant$y),"x"]
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count)) + geom_vline(xintercept = rep1_density_minima)
  rep1_filtered <- rep1 %>% filter(log10_count > rep1_density_minima)
  
  rep2 <- read.delim(myfiles[list_of_indexes[2]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count))
  rep2_density_table <- ggplot_build(rep2_density)$data[[1]]
  rep2_minquant <- rep2_density_table %>% filter(x < quantile(rep2_density_table$x, minquant_fraction))
  rep2_density_minima <- rep2_minquant[rep2_minquant$y == min(rep2_minquant$y),"x"]
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count)) + geom_vline(xintercept = rep2_density_minima)
  rep2_filtered <- rep2 %>% filter(log10_count > rep2_density_minima)
  
  bin1 <- merge(rep1_filtered, rep2_filtered, by = "X", all = T);bin1$last3 <- substr(bin1$X,7,9)
  bin1_non_atg <- bin1 %>% filter(last3 != "ATG") %>% arrange(desc(count.x), desc(count.y))
  bin1[is.na(bin1)]<-0
  #bin1$bin1 <- rowSums(bin1[,c("count.x","count.y")], na.rm = T)
  #bin1$bin1_freq <- bin1$bin1 / sum(bin1$bin1)
  bin1$freq.x <- bin1$count.x / sum(bin1$count.x)
  bin1$freq.y <- bin1$count.y / sum(bin1$count.y)
  bin1$bin1_freq <- rowMeans(bin1[c("freq.x", "freq.y")])
  
  ## Bin 2
  rep1 <- read.delim(myfiles[list_of_indexes[3]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count))
  rep1_density_table <- ggplot_build(rep1_density)$data[[1]]
  rep1_minquant <- rep1_density_table %>% filter(x < quantile(rep1_density_table$x, minquant_fraction))
  rep1_density_minima <- rep1_minquant[rep1_minquant$y == min(rep1_minquant$y),"x"]
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count)) + geom_vline(xintercept = rep1_density_minima)
  rep1_filtered <- rep1 %>% filter(log10_count > rep1_density_minima)
  
  rep2 <- read.delim(myfiles[list_of_indexes[4]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count))
  rep2_density_table <- ggplot_build(rep2_density)$data[[1]]
  rep2_minquant <- rep2_density_table %>% filter(x < quantile(rep2_density_table$x, minquant_fraction))
  rep2_density_minima <- rep2_minquant[rep2_minquant$y == min(rep2_minquant$y),"x"]
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count)) + geom_vline(xintercept = rep2_density_minima)
  rep2_filtered <- rep2 %>% filter(log10_count > rep2_density_minima)
  
  bin2 <- merge(rep1_filtered, rep2_filtered, by = "X", all = T);bin2$last3 <- substr(bin2$X,7,9)
  bin2_non_atg <- bin2 %>% filter(last3 != "ATG") %>% arrange(desc(count.x), desc(count.y))
  bin2[is.na(bin2)]<-0
  #bin2$bin2 <- rowSums(bin2[,c("count.x","count.y")], na.rm = T)
  #bin2$bin2_freq <- bin2$bin2 / sum(bin2$bin2)
  bin2$freq.x <- bin2$count.x / sum(bin2$count.x)
  bin2$freq.y <- bin2$count.y / sum(bin2$count.y)
  bin2$bin2_freq <- rowMeans(bin2[c("freq.x", "freq.y")])
  
  ## Bin 3
  rep1 <- read.delim(myfiles[list_of_indexes[5]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count))
  rep1_density_table <- ggplot_build(rep1_density)$data[[1]]
  rep1_minquant <- rep1_density_table %>% filter(x < quantile(rep1_density_table$x, minquant_fraction))
  rep1_density_minima <- rep1_minquant[rep1_minquant$y == min(rep1_minquant$y),"x"]
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count)) + geom_vline(xintercept = rep1_density_minima)
  rep1_filtered <- rep1 %>% filter(log10_count > rep1_density_minima)
  
  rep2 <- read.delim(myfiles[list_of_indexes[6]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count))
  rep2_density_table <- ggplot_build(rep2_density)$data[[1]]
  rep2_minquant <- rep2_density_table %>% filter(x < quantile(rep2_density_table$x, minquant_fraction))
  rep2_density_minima <- rep2_minquant[rep2_minquant$y == min(rep2_minquant$y),"x"]
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count)) + geom_vline(xintercept = rep2_density_minima)
  rep2_filtered <- rep2 %>% filter(log10_count > rep2_density_minima)
  
  bin3 <- merge(rep1_filtered, rep2_filtered, by = "X", all = T);bin3$last3 <- substr(bin3$X,7,9)
  bin3_non_atg <- bin3 %>% filter(last3 != "ATG") %>% arrange(desc(count.x), desc(count.y))
  bin3[is.na(bin3)]<-0
  #bin3$bin3 <- rowSums(bin3[,c("count.x","count.y")], na.rm = T)
  #bin3$bin3_freq <- bin3$bin3 / sum(bin3$bin3)
  bin3$freq.x <- bin3$count.x / sum(bin3$count.x)
  bin3$freq.y <- bin3$count.y / sum(bin3$count.y)
  bin3$bin3_freq <- rowMeans(bin3[c("freq.x", "freq.y")])
  
  ## Bin 4
  rep1 <- read.delim(myfiles[list_of_indexes[7]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count))
  rep1_density_table <- ggplot_build(rep1_density)$data[[1]]
  rep1_minquant <- rep1_density_table %>% filter(x < quantile(rep1_density_table$x, minquant_fraction))
  rep1_density_minima <- rep1_minquant[rep1_minquant$y == min(rep1_minquant$y),"x"]
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count)) + geom_vline(xintercept = rep1_density_minima)
  rep1_filtered <- rep1 %>% filter(log10_count > rep1_density_minima)
  
  rep2 <- read.delim(myfiles[list_of_indexes[8]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count))
  rep2_density_table <- ggplot_build(rep2_density)$data[[1]]
  rep2_minquant <- rep2_density_table %>% filter(x < quantile(rep2_density_table$x, minquant_fraction))
  rep2_density_minima <- rep2_minquant[rep2_minquant$y == min(rep2_minquant$y),"x"]
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count)) + geom_vline(xintercept = rep2_density_minima)
  rep2_filtered <- rep2 %>% filter(log10_count > rep2_density_minima)
  
  bin4 <- merge(rep1_filtered, rep2_filtered, by = "X", all = T);bin4$last3 <- substr(bin4$X,7,9)
  bin4_non_atg <- bin4 %>% filter(last3 != "ATG") %>% arrange(desc(count.x), desc(count.y))
  bin4[is.na(bin4)]<-0
  #bin4$bin4 <- rowSums(bin4[,c("count.x","count.y")], na.rm = T)
  #bin4$bin4_freq <- bin4$bin4 / sum(bin4$bin4)
  bin4$freq.x <- bin4$count.x / sum(bin4$count.x)
  bin4$freq.y <- bin4$count.y / sum(bin4$count.y)
  bin4$bin4_freq <- rowMeans(bin4[c("freq.x", "freq.y")])
  
  ## Combine the data
  combined_frame <- merge(bin1[,c("X","count.x","count.y","bin1_freq")], bin2[,c("X","count.x","count.y","bin2_freq")], by = "X", all = T)
  combined_frame2 <- merge(combined_frame, bin3[,c("X","count.x","count.y","bin3_freq")], by = "X", all = T)
  combined_frame3 <- merge(combined_frame2, bin4[,c("X","count.x","count.y","bin4_freq")], by = "X", all = T)
  colnames(combined_frame3) <- c("sequence","bin1a","bin1b","bin1_freq","bin2a","bin2b","bin2_freq","bin3a","bin3b","bin3_freq","bin4a","bin4b","bin4_freq"); combined_frame3[is.na(combined_frame3)] <- 0
  
  ## Do some additional analysis before returning the data frame
  combined_frame3$total_count <- rowSums(combined_frame3[,c("bin1a","bin1b","bin2a","bin2b","bin3a","bin3b","bin4a","bin4b")])
  combined_frame3$w_ave <- (combined_frame3$bin1_freq * 0 + combined_frame3$bin2_freq * 1/3 + combined_frame3$bin3_freq * 2/3 + combined_frame3$bin4_freq) / (combined_frame3$bin1_freq + combined_frame3$bin2_freq + combined_frame3$bin3_freq + combined_frame3$bin4_freq)
  return(combined_frame3)
}
```

``` r
### Next make a function for analyzing the enrichment of the virus in hygromycin antibiotic (thus selecting for infected cells)
#list_of_indexes <- c(192,193,155,155) ## Delete whenever. Only for troubleshooting.

makeExperimentFrame2 <- function(list_of_indexes){
  ## Deal with the unselected data first
  rep1 <- read.delim(myfiles[list_of_indexes[1]], sep = "\t")
  rep1 <- rbind(c("XXXXXXXXX",template$template[list_of_indexes[1]]),rep1)
  rep1$count <- as.numeric(rep1$count)
  rep1$log10_count <- log10(rep1$count)
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count))
  rep1_density_table <- ggplot_build(rep1_density)$data[[1]]
  rep1_minquant <- rep1_density_table %>% filter(x < quantile(rep1_density_table$x, minquant_fraction, na.rm = T))
  rep1_density_minima <- rep1_minquant[rep1_minquant$y == min(rep1_minquant$y),"x"]
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count)) + geom_vline(xintercept = rep1_density_minima)
  rep1_filtered <- rep1 %>% filter(log10_count > rep1_density_minima)
  
  rep2 <- read.delim(myfiles[list_of_indexes[2]], sep = "\t")
  rep2 <- rbind(c("XXXXXXXXX",template$template[list_of_indexes[2]]),rep2)
  rep2$count <- as.numeric(rep2$count)
  rep2$log10_count <- log10(rep2$count)
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count))
  rep2_density_table <- ggplot_build(rep2_density)$data[[1]]
  rep2_minquant <- rep2_density_table %>% filter(x < quantile(rep2_density_table$x, minquant_fraction, na.rm = T))
  rep2_density_minima <- rep2_minquant[rep2_minquant$y == min(rep2_minquant$y),"x"]
  rep2_density <- ggplot() + geom_density(data = rep2, aes(x = log10_count)) + geom_vline(xintercept = rep2_density_minima)
  rep2_filtered <- rep2 %>% filter(log10_count > rep2_density_minima)
  
  unsel <- merge(rep1_filtered, rep2_filtered, by = "X", all = T)
  unsel$last3 <- substr(unsel$X,7,9)
  unsel_non_atg <- unsel %>% filter(last3 != "ATG" & last3 != "XXX") %>% arrange(desc(count.x), desc(count.y))
  unsel[is.na(unsel)]<-0
  unsel[unsel$count.x <= quantile(unsel_non_atg$count.x,quantile_cutoff, na.rm = T),"count.x"]<-0
  unsel[unsel$count.y <= quantile(unsel_non_atg$count.y,quantile_cutoff, na.rm = T),"count.y"]<-0
  unsel$freq.x <- unsel$count.x / sum(unsel$count.x)
  unsel$freq.y <- unsel$count.y / sum(unsel$count.y)
  #unsel$u <- unsel$count.x + unsel$count.y
  #unsel <- unsel %>% filter(u > quantile(unsel_non_atg$count.x,quantile_cutoff, na.rm = T) + quantile(unsel_non_atg$count.y,quantile_cutoff, na.rm = T))
  #unsel$u_freq <- (unsel$freq.x + unsel$freq.y)/2
  unsel$u_freq <- 10^((log10(unsel$freq.x) + log10(unsel$freq.y))/2)
  
  ## Deal with the HygroR data next
  hygro <- merge(read.delim(myfiles[list_of_indexes[3]], sep = "\t"), read.delim(myfiles[list_of_indexes[4]], sep = "\t"), by = "X", all = T)
  hygro <- rbind(c("XXXXXXXXX",template$template[list_of_indexes[3]],template$template[list_of_indexes[4]]),hygro)
  hygro$last3 <- substr(hygro$X,7,9)
  hygro$count.x <- as.numeric(hygro$count.x)
  hygro$count.y <- as.numeric(hygro$count.y)
  hygro_non_atg <- hygro %>% filter(last3 != "ATG" & last3 != "XXX") %>% arrange(desc(count.x), desc(count.y))
  hygro[is.na(hygro)]<-0
  hygro[hygro$count.x <= quantile(hygro_non_atg$count.x,quantile_cutoff, na.rm = T),"count.x"]<-0
  hygro[hygro$count.y <= quantile(hygro_non_atg$count.y,quantile_cutoff, na.rm = T),"count.y"]<-0
  hygro$freq.x <- hygro$count.x / sum(hygro$count.x)
  hygro$freq.y <- hygro$count.y / sum(hygro$count.y)
  #hygro$u <- hygro$count.x + hygro$count.y
  #hygro <- hygro %>% filter(u > quantile(hygro_non_atg$count.x,quantile_cutoff, na.rm = T) + quantile(hygro_non_atg$count.y,quantile_cutoff, na.rm = T))
  #hygro$h_freq <- (hygro$freq.x + hygro$freq.y)/2
  hygro$h_freq <- 10^((log10(hygro$freq.x) + log10(hygro$freq.y))/2)
  
  ## Combine the data
  combined_frame <- merge(unsel[,c("X","count.x","count.y","u_freq")], hygro[,c("X","count.x","count.y","h_freq")], by = "X", all = T)
  colnames(combined_frame) <- c("sequence","u1","u2","u_freq","h1","h2","h_freq"); combined_frame[is.na(combined_frame)] <- 0
  ## Do some additional analysis before returning the data frame
  combined_frame$h_enrichment <- combined_frame$h_freq / combined_frame$u_freq
  combined_frame$total_reads <- combined_frame$u1+combined_frame$u2+combined_frame$h1+combined_frame$h2
  return(combined_frame)
}

individual_sample_counting <- function(list_of_indexes){
  rep1 <- read.delim(myfiles[list_of_indexes[1]], sep = "\t") %>% mutate(log10_count = log10(count))
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count))
  rep1_density_table <- ggplot_build(rep1_density)$data[[1]]
  rep1_minquant <- rep1_density_table %>% filter(x < quantile(rep1_density_table$x, minquant_fraction))
  rep1_density_minima <- rep1_minquant[rep1_minquant$y == min(rep1_minquant$y),"x"]
  rep1_density <- ggplot() + geom_density(data = rep1, aes(x = log10_count)) + geom_vline(xintercept = rep1_density_minima)
  rep1_filtered <- rep1 %>% filter(log10_count > rep1_density_minima)
  return(rep1_filtered)
}
```

## Comparing Dox titration with the Kozak mutation method

``` r
dox_titr_n <- 5000

dox0 <- read.csv(file = "Data/Flow_cytometry/F267/export_A1_Singlets.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 0)
dox10 <- read.csv(file = "Data/Flow_cytometry/F267/export_A2_Singlets.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 10)
dox25 <- read.csv(file = "Data/Flow_cytometry/F267/export_A4_Singlets.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 25)
dox100 <- read.csv(file = "Data/Flow_cytometry/F267/export_A5_Singlets.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 100)
dox250 <- read.csv(file = "Data/Flow_cytometry/F267/export_A6_Singlets.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 250)
dox1000 <- read.csv(file = "Data/Flow_cytometry/F267/export_A8_Singlets.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 1000)
dox2000ss <- read.csv(file = "Data/Flow_cytometry/F267/export_A10_Singlets.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 2000)

all_dox <- rbind(dox0,dox10,dox100,dox1000)

koz1 <- read.csv(file = "Data/Flow_cytometry/F168/G790A_export_E1_Recombined.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 5)
koz2 <- read.csv(file = "Data/Flow_cytometry/F168/1B_export_E10_Recombined.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 4)
koz3 <- read.csv(file = "Data/Flow_cytometry/F168/HL_export_E5_Recombined.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 3)
koz4 <- read.csv(file = "Data/Flow_cytometry/F168/1M_export_E4_Recombined.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 2)
koz5 <- read.csv(file = "Data/Flow_cytometry/F168/G868A_export_E2_Recombined.csv", header = T, stringsAsFactors = F)[1:dox_titr_n,] %>% mutate(label = 1)

all_koz <- rbind(koz1,koz2,koz3,koz5)

dox_histograms <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(y = "Cell number", x = "NIR MFI") + 
  scale_x_log10(limits = c(1e1,1e5), breaks = c(1e1,1e3,1e5)) + scale_y_continuous(limits = c(0,1500), breaks = c(0,500,1000)) +
  geom_histogram(data = all_dox, aes(x = RL1.A), bins = 20, fill = "grey80", color = "black") +
  facet_grid(rows = vars(label)) +
  NULL; dox_histograms
```

![](Kozak_files/figure-gfm/Comparing%20the%20dox%20titration%20with%20Kozak%20sequence%20controls-1.png)<!-- -->

``` r
ggsave(file = "Plots/dox_histograms.pdf", dox_histograms, height = 2.5, width = 4)

koz_histograms <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(y = "Cell number", x = "NIR MFI") + 
  scale_x_log10(limits = c(1e1,1e5), breaks = c(1e1,1e3,1e5)) + scale_y_continuous(limits = c(0,1500), breaks = c(0,500,1000)) +
  geom_histogram(data = all_koz, aes(x = RL1.A), bins = 20, fill = "grey80", color = "black") +
  facet_grid(rows = vars(label)) +
  NULL; koz_histograms
```

![](Kozak_files/figure-gfm/Comparing%20the%20dox%20titration%20with%20Kozak%20sequence%20controls-2.png)<!-- -->

``` r
ggsave(file = "Plots/koz_histograms.pdf", koz_histograms, height = 2.5, width = 4)


## Summary to show geomean
all_dox_summary <- data.frame("label" = unique(all_dox$label), "geomean" = 0, "geo_cv" = 0, "mean" = 0, "cv" = 0)
for(x in 1:nrow(all_dox_summary)){
  temp_label <- all_dox_summary$label[x]
  all_dox_summary$geomean[x] <- 10^(mean(log10((all_dox %>% filter(label == temp_label & YL2.A > 0))$YL2.A)))
  all_dox_summary$geo_cv[x] <- 10^(sd(log10((all_dox %>% filter(label == temp_label & YL2.A > 0))$YL2.A))/mean(log10((all_dox %>% filter(label == temp_label & YL2.A > 0))$YL2.A)))
  all_dox_summary$mean[x] <- mean((all_dox %>% filter(label == temp_label & YL2.A > 0))$YL2.A)
  all_dox_summary$cv[x] <- sd((all_dox %>% filter(label == temp_label & YL2.A > 0))$YL2.A)/mean((all_dox %>% filter(label == temp_label & YL2.A > 0))$YL2.A)
}

all_koz_summary <- data.frame("label" = unique(all_koz$label), "geomean" = 0, "geo_cv" = 0, "mean" = 0, "cv" = 0)
for(x in 1:nrow(all_koz_summary)){
  temp_label <- all_koz_summary$label[x]
  all_koz_summary$geomean[x] <- 10^(mean(log10((all_koz %>% filter(label == temp_label & RL1.A > 0))$RL1.A)))
  all_koz_summary$geo_cv[x] <- 10^(sd(log10((all_koz %>% filter(label == temp_label & RL1.A > 0))$RL1.A)) / mean(log10((all_koz %>% filter(label == temp_label & RL1.A > 0))$RL1.A)))
  all_koz_summary$mean[x] <- mean((all_koz %>% filter(label == temp_label & RL1.A > 0))$RL1.A)
  all_koz_summary$cv[x] <- sd((all_koz %>% filter(label == temp_label & RL1.A > 0))$RL1.A)/mean((all_koz %>% filter(label == temp_label & RL1.A > 0))$RL1.A)
}
```

## Looking at the MFI distribution of the library as compared to some of the controls

``` r
library_for_scatterplot <- read.csv(file = "Data/Flow_cytometry/F239/csv/export_F5_Singlets.csv", header = T, stringsAsFactors = F)
control_for_scatterplot <- read.csv(file = "Data/Flow_cytometry/F239/csv/export_A1_Singlets.csv", header = T, stringsAsFactors = F)
control_for_scatterplot2 <- read.csv(file = "Data/Flow_cytometry/F239/csv/export_A1_G542A_Singlets.csv", header = T, stringsAsFactors = F)

Red_blue_scatterplot <-  ggplot() + theme(panel.grid = element_blank(), legend.position  = "none") + 
  scale_x_log10(limits = c(1,3e5)) + scale_y_log10(limits = c(1,3e5)) + 
  labs(x  = "Blue fluorescence", y = "Red fluorescence") +
  geom_point(data = library_for_scatterplot[1:3000,], aes(x = VL1.A, y = YL2.A), alpha = 0.01, size = 0.25, color = "red") +
  geom_point(data = control_for_scatterplot[1:3000,], aes(x = VL1.A, y = YL2.A), alpha = 0.01, size = 0.25, color = "black") +
  geom_point(data = control_for_scatterplot2[1:3000,], aes(x = VL1.A, y = YL2.A), alpha = 0.01, size = 0.25, color = "blue")

Red_blue_scatterplot
```

![](Kozak_files/figure-gfm/MFI%20of%20library%20with%20controls-1.png)<!-- -->

``` r
ggsave(file = "plots/Red_blue_scatterplot.pdf", Red_blue_scatterplot, height = 1.8, width = 2.1)
```

``` r
Red_nir_densityplot <- ggplot() + theme(panel.grid = element_blank(), legend.position  = "none") + 
  labs(x  = "Fluorescence", y = "Density") +
  scale_x_log10() + scale_y_continuous(breaks = c(0,1)) +
  geom_density(data = control_for_scatterplot, aes(x = YL2.A, y = ..scaled..), color = "red", alpha = 0.4, linetype = 2) +
  geom_density(data = control_for_scatterplot, aes(x = RL1.A, y = ..scaled..), color = "black", alpha = 0.4, linetype = 2) +
  geom_density(data = library_for_scatterplot, aes(x = YL2.A, y = ..scaled..), color = "red", fill = "red", alpha = 0.4) +
  geom_density(data = library_for_scatterplot, aes(x = RL1.A, y = ..scaled..), color = "black", fill = "black", alpha = 0.4)
Red_nir_densityplot
```

![](Kozak_files/figure-gfm/Comparing%20red%20and%20nir%20for%20the%20recombined%20library-1.png)<!-- -->

``` r
ggsave(file = "plots/Red_nir_densityplot.pdf", Red_nir_densityplot, height = 0.8, width = 1.8)
```

``` r
library_mfi <- read.csv(file = "Data/Flow_cytometry/F237/csv/export_B4_Singlets.csv", header = T, stringsAsFactors = F)
gccacc_mfi <- read.csv(file = "Data/Flow_cytometry/F237/csv/export_B1_Singlets.csv", header = T, stringsAsFactors = F)
ttatgg_mfi <- read.csv(file = "Data/Flow_cytometry/F237/csv/export_B3_Singlets.csv", header = T, stringsAsFactors = F)

combined_red_blue_mfi <- rbind((library_mfi[,c("YL2.A","VL1.A")] %>% mutate(sample = "Library"))[1:25000,],
                           (gccacc_mfi[,c("YL2.A","VL1.A")] %>% mutate(sample = "GCCACC"))[1:25000,],
                           (ttatgg_mfi[,c("YL2.A","VL1.A")] %>% mutate(sample = "TTATGG"))[1:25000,])

Flow_library_red_densityplot <- ggplot() + theme(panel.grid = element_blank(), legend.position  = "top") + 
  scale_x_log10(limits = c(3e1,3e5)) +  
  labs(x  = "MFI (Mean fluorecence intensity)", y = "Cell density") +
  geom_density(data = combined_red_blue_mfi, aes(x = YL2.A, fill = sample, color = sample), alpha = 0.4)
Flow_library_red_densityplot
```

![](Kozak_files/figure-gfm/F237%20MFI%20of%20library%20with%20controls-1.png)<!-- -->

``` r
ggsave(file = "plots/F237_Flow_library_red_densityplot.pdf", Flow_library_red_densityplot, height = 1.75, width = 3)


combined_cell_mfi <- rbind((library_mfi[,c("FSC.A","RL1.A")] %>% mutate(sample = "Library"))[1:25000,],
                           (gccacc_mfi[,c("FSC.A","RL1.A")] %>% mutate(sample = "GCCACC"))[1:25000,],
                           (ttatgg_mfi[,c("FSC.A","RL1.A")] %>% mutate(sample = "TTATGG"))[1:25000,])

quartiles_values <- c(quantile(library_mfi$RL1.A,0.25),quantile(library_mfi$RL1.A,0.5),quantile(library_mfi$RL1.A,0.75))

Flow_library_densityplot <- ggplot() + theme(panel.grid = element_blank(), legend.position  = "top") + 
  scale_x_log10(limits = c(3e1,3e5)) + 
  labs(x  = "MFI (Mean fluorecence intensity)", y = "Cell density") +
  geom_density(data = combined_cell_mfi, aes(x = RL1.A, fill = sample, color = sample), alpha = 0.4) +
  geom_vline(xintercept = quartiles_values, linetype = 2, alpha = 0.5)
Flow_library_densityplot
```

![](Kozak_files/figure-gfm/F237%20MFI%20of%20library%20with%20controls-2.png)<!-- -->

``` r
ggsave(file = "plots/F237_Flow_library_densityplot.pdf", Flow_library_densityplot, height = 1.75, width = 3)
```

``` r
library_mfi <- read.csv(file = "Data/Flow_cytometry/F239/csv/export_F5_Bneg_Rpos.csv", header = T, stringsAsFactors = F)
gccacc_mfi <- read.csv(file = "Data/Flow_cytometry/F239/csv/export_F6_Bneg_Rpos.csv", header = T, stringsAsFactors = F)
ttatgg_mfi <- read.csv(file = "Data/Flow_cytometry/F239/csv/export_F8_Bneg_Rpos.csv", header = T, stringsAsFactors = F)

combined_red_blue_mfi <- rbind((library_mfi[,c("YL2.A","VL1.A")] %>% mutate(sample = "Library"))[1:25000,],
                           (gccacc_mfi[,c("YL2.A","VL1.A")] %>% mutate(sample = "GCCACC"))[1:25000,],
                           (ttatgg_mfi[,c("YL2.A","VL1.A")] %>% mutate(sample = "TTATGG"))[1:25000,])

Flow_library_red_densityplot <- ggplot() + theme(panel.grid = element_blank(), legend.position  = "top") + 
  scale_x_log10(limits = c(3e1,3e5)) +  
  labs(x  = "MFI (Mean fluorecence intensity)", y = "Cell density") +
  geom_density(data = combined_red_blue_mfi, aes(x = YL2.A, fill = sample, color = sample), alpha = 0.4)
Flow_library_red_densityplot
```

![](Kozak_files/figure-gfm/F239%20MFI%20of%20library%20with%20controls-1.png)<!-- -->

``` r
ggsave(file = "plots/F239_Flow_library_red_densityplot.pdf", Flow_library_red_densityplot, height = 1.75, width = 3)


combined_cell_mfi <- rbind((library_mfi[,c("FSC.A","RL1.A")] %>% mutate(sample = "Library"))[1:25000,],
                           (gccacc_mfi[,c("FSC.A","RL1.A")] %>% mutate(sample = "GCCACC"))[1:25000,],
                           (ttatgg_mfi[,c("FSC.A","RL1.A")] %>% mutate(sample = "TTATGG"))[1:25000,])

quartiles_values <- c(quantile(library_mfi$RL1.A,0.25),quantile(library_mfi$RL1.A,0.5),quantile(library_mfi$RL1.A,0.75))

Flow_library_densityplot <- ggplot() + theme(panel.grid = element_blank(), legend.position  = "top") + 
  scale_x_log10(limits = c(3e1,3e5)) + 
  labs(x  = "MFI (Mean fluorecence intensity)", y = "Cell density") +
  geom_density(data = combined_cell_mfi, aes(x = RL1.A, fill = sample, color = sample), alpha = 0.4) +
  geom_vline(xintercept = quartiles_values, linetype = 2, alpha = 0.5)
Flow_library_densityplot
```

![](Kozak_files/figure-gfm/F239%20MFI%20of%20library%20with%20controls-2.png)<!-- -->

``` r
ggsave(file = "plots/F239_Flow_library_densityplot.pdf", Flow_library_densityplot, height = 1.75, width = 3)

nrow(subset(gccacc_mfi, RL1.A < quartiles_values[1])) / nrow(gccacc_mfi) * 100
```

    ## [1] 1.698442

``` r
nrow(subset(gccacc_mfi, RL1.A > quartiles_values[1] & RL1.A < quartiles_values[2])) / nrow(gccacc_mfi) * 100
```

    ## [1] 1.539309

``` r
nrow(subset(gccacc_mfi, RL1.A > quartiles_values[2] & RL1.A < quartiles_values[3])) / nrow(gccacc_mfi) * 100
```

    ## [1] 14.45359

``` r
nrow(subset(gccacc_mfi, RL1.A > quartiles_values[3])) / nrow(gccacc_mfi) * 100
```

    ## [1] 82.30254

``` r
nrow(subset(ttatgg_mfi, RL1.A < quartiles_values[1])) / nrow(ttatgg_mfi) * 100
```

    ## [1] 96.87106

``` r
nrow(subset(ttatgg_mfi, RL1.A > quartiles_values[1] & RL1.A < quartiles_values[2])) / nrow(ttatgg_mfi) * 100
```

    ## [1] 2.746641

``` r
nrow(subset(ttatgg_mfi, RL1.A > quartiles_values[2] & RL1.A < quartiles_values[3])) / nrow(ttatgg_mfi) * 100
```

    ## [1] 0.3414743

``` r
nrow(subset(ttatgg_mfi, RL1.A > quartiles_values[3])) / nrow(ttatgg_mfi) * 100
```

    ## [1] 0.04082845

``` r
stained <- read.csv(file = "Data/Flow_cytometry/F262/1_Primary_and_secondary.csv", header = T, stringsAsFactors = F) %>% mutate(sample = "1_stained")
no_primary <- read.csv(file = "Data/Flow_cytometry/F262/2_Secondary_only.csv", header = T, stringsAsFactors = F) %>% mutate(sample = "2_no_primary")
no_secondary <- read.csv(file = "Data/Flow_cytometry/F262/3_Primary_only.csv", header = T, stringsAsFactors = F) %>% mutate(sample = "3_no_secondary")
no_recomb <- read.csv(file = "Data/Flow_cytometry/F262/4_293T_stained.csv", header = T, stringsAsFactors = F) %>% mutate(sample = "4_no_recombined")

flow_downsampled <- 5000

cell_surface_staining_data <- rbind(stained[1:flow_downsampled,c("sample","BL1.A","RL1.A")], no_primary[1:flow_downsampled,c("sample","BL1.A","RL1.A")], no_secondary[1:flow_downsampled,c("sample","BL1.A","RL1.A")], no_recomb[1:flow_downsampled,c("sample","BL1.A","RL1.A")]) %>% filter(sample != "3_no_secondary")

ace2_staining_scatterplot_cols <- c("1_stained" = "green2", "2_no_primary" = "purple", "4_no_recombined" = "brown")

ACE2_staining_scatterplot <- ggplot() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  labs(x = "Near infrared fluorescence", y = "AF488 (Cell surface ACE2 staining") +
  scale_x_log10(limits = c(1e1,1e5)) + scale_y_log10(limits = c(1e1,3e5)) + 
  scale_color_manual(values = ace2_staining_scatterplot_cols) + 
  geom_point(data = cell_surface_staining_data , aes(x = RL1.A, y = BL1.A, color = sample), alpha = 0.04, size = 0.5)
ACE2_staining_scatterplot
```

![](Kozak_files/figure-gfm/Showing%20how%20near%20infrared%20fluorescence%20correlates%20with%20ACE2%20cell%20surface%20abundance-1.png)<!-- -->

``` r
ggsave(file = "plots/ACE2_staining_scatterplot.pdf", ACE2_staining_scatterplot, height = 3, width = 5)
```

## First, analyzing how the sequences of the Kozak sequences affect protein translation and abundance

``` r
## Import our 4-way sorting data
sort1 <- fourWaySortAnalysis(c(72,73,74,75,76,77,78,79)) %>% arrange(desc(w_ave))
sort2 <- fourWaySortAnalysis(c(80,81,82,83,84,85,86,87)) %>% arrange(desc(w_ave))
sort3 <- fourWaySortAnalysis(c(104,104,105,105,106,106,107,107)) %>% arrange(desc(w_ave))
sort4 <- fourWaySortAnalysis(c(89,89,90,90,91,91,92,92)) %>% arrange(desc(w_ave))

#myfiles[c(72,73,74,75,76,77,78,79)]
#myfiles[c(80,81,82,83,84,85,86,87)]
#myfiles[c(104,104,105,105,106,106,107,107)]
#myfiles[c(89,89,90,90,91,91,92,92)]
```

``` r
## Look at distributions of control sequences
sort_controls <- melt(rbind(
  data.frame(t(t((sort1 %>% filter(sequence == "TTATGGATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])/sum(t((sort1 %>% filter(sequence == "TTATGGATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])))) %>% mutate(replicate = "rep1", sequence = "TTATGGATG"),
  data.frame(t(t((sort2 %>% filter(sequence == "TTATGGATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])/sum(t((sort2 %>% filter(sequence == "TTATGGATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])))) %>% mutate(replicate = "rep2", sequence = "TTATGGATG"),
  data.frame(t(t((sort3 %>% filter(sequence == "TTATGGATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])/sum(t((sort3 %>% filter(sequence == "TTATGGATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])))) %>% mutate(replicate = "rep4", sequence = "TTATGGATG"),
  data.frame(t(t((sort4 %>% filter(sequence == "TTATGGATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])/sum(t((sort4 %>% filter(sequence == "TTATGGATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])))) %>% mutate(replicate = "rep5", sequence = "TTATGGATG"),
  data.frame(t(t((sort1 %>% filter(sequence == "GCCACCATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])/sum(t((sort1 %>% filter(sequence == "GCCACCATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])))) %>% mutate(replicate = "rep1", sequence = "GCCACCATG"),
  data.frame(t(t((sort2 %>% filter(sequence == "GCCACCATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])/sum(t((sort2 %>% filter(sequence == "GCCACCATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])))) %>% mutate(replicate = "rep2", sequence = "GCCACCATG"),
  data.frame(t(t((sort3 %>% filter(sequence == "GCCACCATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])/sum(t((sort3 %>% filter(sequence == "GCCACCATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])))) %>% mutate(replicate = "rep4", sequence = "GCCACCATG"),
  data.frame(t(t((sort4 %>% filter(sequence == "GCCACCATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])/sum(t((sort4 %>% filter(sequence == "GCCACCATG"))[,c("bin1_freq","bin2_freq","bin3_freq","bin4_freq")])))) %>% mutate(replicate = "rep5", sequence = "GCCACCATG")))

sort_controls_summary <- sort_controls %>% mutate(log10_value = log10(value)) %>% filter(log10_value != "-Inf") %>% group_by(sequence, variable) %>% summarize(mean_log10 = mean(log10_value)) %>% mutate(geomean = 10^mean_log10)

Controls_sortseq_plot <- ggplot() + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.text.x = element_text(angle = -90, hjust = 0, vjust = 0.5)) + 
  labs(x = NULL, y = "Percent in bin\n(across four bins)") +
  scale_y_continuous(limits = c(0,75), breaks = seq(0,75,25)) +
  geom_quasirandom(data = sort_controls, aes(x = variable, y = value*100), alpha = 0.2) +
  geom_point(data = sort_controls_summary, aes(x = variable, y = geomean*100), shape = 95, size = 12) + 
  facet_wrap(~sequence, ncol=1)
Controls_sortseq_plot
```

![](Kozak_files/figure-gfm/Finding%20bin-wise%20distributions%20of%20high%20and%20low%20control%20variants-1.png)<!-- -->

``` r
ggsave(file = "Plots/Controls_sortseq_plot.pdf", Controls_sortseq_plot, height = 2, width = 2.5)
```

``` r
## Combining the 4-way sorting data into a single data frame
sort_combined <- merge(sort1[,c("sequence","w_ave","total_count")], sort2[,c("sequence","w_ave","total_count")], by = "sequence", all = T)
sort_combined <- merge(sort_combined, sort3[,c("sequence","w_ave","total_count")], by = "sequence", all = T)
sort_combined <- merge(sort_combined, sort4[,c("sequence","w_ave","total_count")], by = "sequence", all = T)
colnames(sort_combined) <- c("sequence","sort1","sort1_count","sort2","sort2_count","sort3","sort3_count","sort4","sort4_count")
sort_combined$total_count <- rowSums(sort_combined[,c("sort1_count","sort2_count","sort3_count","sort4_count")], na.rm = T)

sort_combined$start <- ""
for(x in 1:nrow(sort_combined)){
  sort_combined$start[x] <- substr(sort_combined$sequence[x],7,9)
}

sort_combined <- sort_combined %>% filter(start == "ATG")
```

``` r
## Filter the data to make sure I'm getting rid of poorly sampled variants
sampling_lod <- read.csv(file = "Data/Sampling_limit_of_detection.csv", header = T, stringsAsFactors = F)

sort_combined2 <- sort_combined

sum(sort_combined2$sort1_count, na.rm = T)
```

    ## [1] 2264587

``` r
paste("There are originally this many non-NA score1 weighted averages: values",sum(!is.na(sort_combined2$sort1)))
```

    ## [1] "There are originally this many non-NA score1 weighted averages: values 4037"

``` r
for(x in 1:nrow(sort_combined2)){
  if(!(is.na(sort_combined2$sort1_count[x]))){
    if(sort_combined2$sort1_count[x] <= 5){
    sort_combined2$sort1[x] <- NA
    }
  }
  if(!(is.na(sort_combined2$sort1[x]))){
    if(sort_combined2$sort1[x] == 0 | sort_combined2$sort1[x] == 1){
      sort_combined2$sort1[x] <- NA
    }
  }
}
paste("There are this many non-NA score1 weighted averages after the count-based filtering: values",sum(!is.na(sort_combined2$sort1)))
```

    ## [1] "There are this many non-NA score1 weighted averages after the count-based filtering: values 3941"

``` r
sum(sort_combined2$sort2_count, na.rm = T)
```

    ## [1] 2721336

``` r
paste("There are originally this many non-NA score2 weighted averages: values",sum(!is.na(sort_combined2$sort2)))
```

    ## [1] "There are originally this many non-NA score2 weighted averages: values 4031"

``` r
for(x in 1:nrow(sort_combined2)){
  if(!(is.na(sort_combined2$sort2_count[x]))){
    if(sort_combined2$sort2_count[x] <= 5){
    sort_combined2$sort2[x] <- NA
    }
  }
  if(!(is.na(sort_combined2$sort2[x]))){
    if(sort_combined2$sort2[x] == 0 | sort_combined2$sort2[x] == 1){
      sort_combined2$sort2[x] <- NA
    }
  }
}
paste("There are this many non-NA score2 weighted averages after the count-based filtering: values",sum(!is.na(sort_combined2$sort2)))
```

    ## [1] "There are this many non-NA score2 weighted averages after the count-based filtering: values 3844"

``` r
sum(sort_combined2$sort3_count, na.rm = T)
```

    ## [1] 14788458

``` r
paste("There are originally this many non-NA score3 weighted averages: values",sum(!is.na(sort_combined2$sort3)))
```

    ## [1] "There are originally this many non-NA score3 weighted averages: values 3876"

``` r
for(x in 1:nrow(sort_combined2)){
  if(!(is.na(sort_combined2$sort3_count[x]))){
    if(sort_combined2$sort3_count[x] <= 7){
    sort_combined2$sort3[x] <- NA
    }
  }
  if(!(is.na(sort_combined2$sort3[x]))){
    if(sort_combined2$sort3[x] == 0 | sort_combined2$sort3[x] == 1){
      sort_combined2$sort3[x] <- NA
    }
  }
}
paste("There are this many non-NA score3 weighted averages after the count-based filtering: values",sum(!is.na(sort_combined2$sort3)))
```

    ## [1] "There are this many non-NA score3 weighted averages after the count-based filtering: values 3786"

``` r
sum(sort_combined2$sort4_count, na.rm = T)
```

    ## [1] 12504792

``` r
paste("There are originally this many non-NA score4 weighted averages: values",sum(!is.na(sort_combined2$sort4)))
```

    ## [1] "There are originally this many non-NA score4 weighted averages: values 4029"

``` r
for(x in 1:nrow(sort_combined2)){
  if(!(is.na(sort_combined2$sort4_count[x]))){
    if(sort_combined2$sort4_count[x] <= 7){
    sort_combined2$sort4[x] <- NA
    }
  }
  if(!(is.na(sort_combined2$sort4[x]))){
    if(sort_combined2$sort4[x] == 0 | sort_combined2$sort4[x] == 1){
      sort_combined2$sort4[x] <- NA
    }
  }
}
paste("There are this many non-NA score4 weighted averages after the count-based filtering: values",sum(!is.na(sort_combined2$sort4)))
```

    ## [1] "There are this many non-NA score4 weighted averages after the count-based filtering: values 3992"

``` r
Replicate_wave_scatterplots <- (ggplot() + geom_point(data = sort_combined2, aes(x = sort1, y = sort2), alpha = 0.02) | ggplot() + geom_point(data = sort_combined2, aes(x = sort1, y = sort3), alpha = 0.02) | ggplot() + geom_point(data = sort_combined2, aes(x = sort1, y = sort4), alpha = 0.02)) / (ggplot() + geom_point(data = sort_combined2, aes(x = sort2, y = sort3), alpha = 0.02) | 
ggplot() + geom_point(data = sort_combined2, aes(x = sort2, y = sort4), alpha = 0.02) | 
ggplot() + geom_point(data = sort_combined2, aes(x = sort3, y = sort4), alpha = 0.02)) +
  NULL; Replicate_wave_scatterplots
```

![](Kozak_files/figure-gfm/Making%20a%20all-by-all%20comparis%20of%20replicate%20weighted%20average%20scores-1.png)<!-- -->

``` r
ggsave(file = "Plots/Replicate_wave_scatterplots.pdf", Replicate_wave_scatterplots, width = 6.5, height = 4)
```

``` r
## Out of curiosity, looking at the average variability of the data
sort_combined2$sd <- 0
for(x in 1:nrow(sort_combined2)){
  sort_combined2$sd[x] <- sd(sort_combined2[x,c("sort1","sort2","sort3","sort4")], na.rm = T)
}

ggplot() + geom_histogram(data = sort_combined2, aes(x = sd))
```

![](Kozak_files/figure-gfm/Getting%20summary%20stats%20for%20the%204-way%20sorting-1.png)<!-- -->

``` r
#median(sort_combined2$sd, na.rm = T)

## Getting summary stats for the 4-way sorting
melt_sort <- melt(sort_combined2[,c("sequence","sort1","sort2","sort3","sort4")], id = "sequence") %>% mutate(n = NA)
melt_sort[!is.na(melt_sort$value),"n"] = 1
melt_sort <- melt_sort %>% filter(!is.na(value))

sort_summary <- melt_sort %>% group_by(sequence) %>% summarize(sort_mean = mean(value, na.rm = T), sort_geomean_log10 = mean(log10(value), na.rm = T), sd_log10 = sd(log10(value), na.rm = T), sort_n = sum(n, na.rm = T), .groups = "drop") %>% arrange(desc(sort_geomean_log10)) %>% mutate(sort_geomean = 10^sort_geomean_log10, sort_upper_conf = 10^(sort_geomean_log10 + sd_log10/sqrt(sort_n-1)*1.96), sort_lower_conf = 10^(sort_geomean_log10 - sd_log10/sqrt(sort_n-1)*1.96))

sort_summary2 <- merge(sort_summary, sort_combined2[,c("sequence","total_count")], by= "sequence") %>% arrange(desc(total_count)) %>% filter(sort_n >= 2)

Mean_weighted_average_histogram <- ggplot() + theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank()) + 
  labs(x = "Mean weighted average", y = "Number of\nvariants") +
  geom_histogram(data = sort_summary2, aes(x = sort_geomean), binwidth = 0.05, color = "black", fill = "grey75")
Mean_weighted_average_histogram
```

![](Kozak_files/figure-gfm/Getting%20summary%20stats%20for%20the%204-way%20sorting-2.png)<!-- -->

``` r
ggsave(file = "plots/Mean_weighted_average_histogram.pdf", Mean_weighted_average_histogram, height = 1.2, width = 3.2)

paste("Weighted ave of low translation rate sample TTATGGATG", round(subset(sort_summary2, sequence == "TTATGGATG")$sort_geomean,2))
```

    ## [1] "Weighted ave of low translation rate sample TTATGGATG 0.3"

``` r
paste("Weighted ave of high translation rate GCCACCATG", round(subset(sort_summary2, sequence == "GCCACCATG")$sort_geomean,2))
```

    ## [1] "Weighted ave of high translation rate GCCACCATG 0.69"

``` r
paste("The minimum weighted average value observed in our dataset", round(min(sort_summary2$sort_geomean),2))
```

    ## [1] "The minimum weighted average value observed in our dataset 0.13"

``` r
paste("The maximum weighted average value observed in our dataset", round(max(sort_summary2$sort_geomean),2))
```

    ## [1] "The maximum weighted average value observed in our dataset 0.91"

## Now bring in the individual data to see how the sort score correlates with actually MFI measured by flow cytometry

``` r
## Import all of my individual testing data
individual_key <- read.csv(file = "Keys/Individual_key.csv", header = T, stringsAsFactors = F)
individual_infection1 <- read.csv(file = "Data/Flow_Cytometry/F168_210809_individual_iRFP670_infection.csv", header = T, stringsAsFactors = F)
individual_infection2 <- read.csv(file = "Data/Flow_Cytometry/F201_211019_individual_iRFP670_infection.csv", header = T, stringsAsFactors = F)
individual_infection3 <- read.csv(file = "Data/Flow_Cytometry/F202_211020_individual_iRFP670_infection.csv", header = T, stringsAsFactors = F)
individual_infection4 <- read.csv(file = "Data/Flow_Cytometry/F203_211025_individual_iRFP670_infection.csv", header = T, stringsAsFactors = F)
individual_infection5 <- read.csv(file = "Data/Flow_Cytometry/F206_211101_individual_iRFP670_infection.csv", header = T, stringsAsFactors = F)
individual_infection6 <- read.csv(file = "Data/Flow_Cytometry/F207_211105_individual_iRFP670_infection.csv", header = T, stringsAsFactors = F)
individual_infection <- rbind(individual_infection1, individual_infection2, individual_infection3, individual_infection4, individual_infection5, individual_infection6)
individual_infection2 <- merge(individual_infection, individual_key, by = "recombined_construct", all.x = T) %>% filter(!is.na(sequence))
individual_infection2$fraction_wt_infection <- 0
individual_infection2$fraction_wt_mfi <- 0

for(x in 1:nrow(individual_infection2)){
  temp_date <- individual_infection2$date[x]
  temp_virus <- individual_infection2$virus[x]
  wt_infection <- individual_infection2 %>% filter(date == temp_date & virus == temp_virus & recombined_construct == "G790A")
  individual_infection2$fraction_wt_infection[x] <- individual_infection2$pct_grn[x] / mean(wt_infection$pct_grn)
  individual_infection2$fraction_wt_mfi[x] <- individual_infection2$mode_nir[x] / mean(wt_infection$mode_nir)
}

individual_infection_summary <- individual_infection2 %>% group_by(virus, cell_label, sequence) %>% 
  summarize(mean_fraction_wt_infection = mean(fraction_wt_infection, na.rm = T), meanfraction_wt_mfi = mean(fraction_wt_mfi, na.rm = T), cell_label = unique(cell_label),
            geomean_fraction_wt_infection = 10^(mean(log10(fraction_wt_infection), na.rm = T)), geomeanfraction_wt_mfi = 10^(mean(log10(fraction_wt_mfi), na.rm = T)), .groups = "drop") %>% 
  filter(cell_label != "Q" & meanfraction_wt_mfi != "NaN") #%>% filter(virus != "G1074A SARS2")
```

``` r
complete_frame2 <- merge(complete_frame, sort_summary2[,c("sequence","sort_geomean_log10","sort_geomean","sort_upper_conf","sort_lower_conf")],all.x = T) %>% filter(!is.na(sort_geomean)) %>% filter(!(sort_geomean %in% c(0,100)))
individual_mfi <- individual_infection_summary %>% group_by(sequence) %>% summarize(mfi_individual = mean(geomeanfraction_wt_mfi, na.rm = T))
complete_frame3 <- merge(complete_frame2, individual_mfi[,c("sequence","mfi_individual")], by = "sequence", all.x = T)
complete_frame3$mfi_individual_log10 <- log10(complete_frame3$mfi_individual)

## Comparing log-transformed 4-way sort scores with the MFI
Individual_vs_4way_sort_scatterplot <- ggplot() + theme(panel.grid = element_blank()) +
  scale_x_log10(limits = c(0.28,0.7)) + scale_y_log10(limits = c(0.0065,1.5)) + 
  labs(y = "Individually assessed MFI\n(Normalized to consensus Kozak)", x = "Mean weighted average") +
  geom_point(data = complete_frame3, aes(x = sort_geomean, y = mfi_individual, )) +
  geom_text_repel(data = complete_frame3, aes(x = sort_geomean, y = mfi_individual, label = sequence), color = "red", alpha = 0.6, size = 2) +
  NULL
Individual_vs_4way_sort_scatterplot
```

![](Kozak_files/figure-gfm/Bring%20in%20the%20individual%20data-1.png)<!-- -->

``` r
ggsave(file = "Plots/Individual_vs_4way_sort_scatterplot.png", Individual_vs_4way_sort_scatterplot, height = 3, width = 4)
ggsave(file = "Plots/Individual_vs_4way_sort_scatterplot.pdf", Individual_vs_4way_sort_scatterplot, height = 3, width = 3.5)
ggsave(file = "Plots/Individual_vs_4way_sort_scatterplot2.pdf", Individual_vs_4way_sort_scatterplot, height = 2.5, width = 3.5)

## Removing the TTAACCATG from the training set since I think there's something weird with that sequence
complete_frame3_for_lm <- complete_frame3 %>% filter(!(is.na(mfi_individual)) & !(sequence %in% "TTAACCATG"))

## Compare 4-way sort data with individual MFI data
mfi_sort_model <- lm(complete_frame3_for_lm$mfi_individual_log10 ~ complete_frame3_for_lm$sort_geomean_log10)

## Now with the linear model added in
Individual_vs_4way_sort_scatterplot2 <- ggplot() + theme(panel.grid = element_blank()) +
  scale_x_continuous(limits = c(-0.56, 0.-0.15)) +
  labs(y = "Log10 individually assessed MFI", x = "Log10 4-way sort score") +
  geom_abline(slope = mfi_sort_model$coefficients[2], intercept = mfi_sort_model$coefficients[1], alpha = 0.2, size = 5) + 
  geom_point(data = complete_frame3 %>% filter(!is.na(mfi_individual_log10)), aes(x = sort_geomean_log10, y = mfi_individual_log10)) +
  geom_text_repel(data = complete_frame3 %>% filter(!is.na(mfi_individual_log10)), aes(x = sort_geomean_log10, y = mfi_individual_log10, label = sequence), color = "red", alpha = 0.6, size = 2) +
  NULL; Individual_vs_4way_sort_scatterplot2
```

![](Kozak_files/figure-gfm/Bring%20in%20the%20individual%20data-2.png)<!-- -->

``` r
ggsave(file = "Plots/Individual_vs_4way_sort_scatterplot2.png", Individual_vs_4way_sort_scatterplot2, height = 3, width = 4)
ggsave(file = "Plots/Individual_vs_4way_sort_scatterplot2.pdf", Individual_vs_4way_sort_scatterplot2, height = 3, width = 4)

paste("Correlation coefficients between the 4-way sort data and individually assessed MFI -->","Pearson's r^2:",
round(cor(complete_frame3$sort_geomean_log10, complete_frame3$mfi_individual_log10, method = "pearson", use = "complete")^2,2),"Spearman's rho^2:",
round(cor(complete_frame3$sort_geomean_log10, complete_frame3$mfi_individual_log10, method = "spearman", use = "complete")^2,2))
```

    ## [1] "Correlation coefficients between the 4-way sort data and individually assessed MFI --> Pearson's r^2: 0.92 Spearman's rho^2: 0.86"

``` r
## Use the above linear model to estimate what the MFI should be based on the 4-way sorting
complete_frame3$mfi_log10_lm <- mfi_sort_model$coefficients[2] * complete_frame3$sort_geomean_log10 + mfi_sort_model$coefficients[1]

## See how the scores look if we use the calculated values based on the linear model
complete_frame3$mfi_lm <- 10^complete_frame3$mfi_log10_lm
complete_frame3$mfi_lm_norm <- complete_frame3$mfi_lm / complete_frame3[complete_frame3$sequence == "GCCACCATG","mfi_lm"]

## The linear model isn't going to work, because it's not bounded by the min (~ 0.01) and max (~1) values. Thus, I'm now using a log equation to achieve that.
L = 1
k = 3.2
xnot = - 0.25

complete_frame3$calibrated_score <- L/(1+exp(-k*(log10(complete_frame3$mfi_lm_norm) - xnot))) + 0.003
complete_frame3$calibrated_score <- complete_frame3$calibrated_score / complete_frame3[complete_frame3$sequence == "GCCACCATG","calibrated_score"]

ggplot() + scale_y_log10(limits = c(0.001,3)) + scale_x_log10(limits = c(0.001,4)) + 
  geom_point(data = complete_frame3, aes(x = mfi_lm_norm, y = calibrated_score))
```

![](Kozak_files/figure-gfm/Use%20this%20linear%20model%20to%20transform%20the%204-way%20sort%20values%20into%20calculated%20MFI%20equivalents-1.png)<!-- -->

``` r
max(complete_frame3$mfi_lm_norm)
```

    ## [1] 3.849882

``` r
min(complete_frame3$mfi_lm_norm)
```

    ## [1] 0.0003126906

``` r
max(complete_frame3$calibrated_score)
```

    ## [1] 1.354221

``` r
min(complete_frame3$calibrated_score)
```

    ## [1] 0.004372397

``` r
## Original linear model calculated data
Individual_vs_4way_sort_lm_scatterplot <- ggplot() + theme(panel.grid.minor = element_blank()) +
  scale_x_log10(limits = c(0.01,1)) + scale_y_log10(limits = c(0.0065,1.5)) + 
  labs(x = "Sortseq predicted MFI (linear adjusted)", y = "Experimentally determined MFI") +
  #geom_abline(slope = mfi_sort_lm_model$coefficients[2], intercept = mfi_sort_lm_model$coefficients[1], alpha = 0.2, size = 5) + 
  geom_point(data = complete_frame3 %>% filter(!is.na(mfi_individual)), aes(x = mfi_lm_norm, y = mfi_individual)) +
  geom_text_repel(data = complete_frame3 %>% filter(!is.na(mfi_individual)), aes(x = mfi_lm_norm, y = mfi_individual, label = sequence), color = "red", alpha = 0.6, size = 2) + NULL; Individual_vs_4way_sort_lm_scatterplot
```

![](Kozak_files/figure-gfm/Use%20this%20model%20to%20transform%20the%204-way%20sort%20values%20into%20calculated%20MFI%20equivalents-1.png)<!-- -->

``` r
ggsave(file = "Plots/Individual_vs_4way_sort_lm_scatterplot.png", Individual_vs_4way_sort_lm_scatterplot, height = 3, width = 4)
ggsave(file = "Plots/Individual_vs_4way_sort_lm_scatterplot.pdf", Individual_vs_4way_sort_lm_scatterplot, height = 3, width = 3.35)

paste("Correlation coefficients between 4-way sort data linear model and individually assessed MFI -->","Pearson's:",
round(cor(log10(complete_frame3$mfi_lm_norm), log10(complete_frame3$mfi_individual), method = "pearson", use = "complete")^2,3),"Spearman's:",
round(cor(log10(complete_frame3$mfi_lm_norm), log10(complete_frame3$mfi_individual), method = "spearman", use = "complete")^2,3))
```

    ## [1] "Correlation coefficients between 4-way sort data linear model and individually assessed MFI --> Pearson's: 0.922 Spearman's: 0.856"

``` r
## Data bounded using log curve
Individual_vs_4way_sort_calcd_scatterplot <- ggplot() + theme(panel.grid.minor = element_blank()) +
  scale_x_log10(limits = c(0.01,1)) + scale_y_log10(limits = c(0.0065,1.5)) + 
  labs(x = "Sortseq predicted MFI (log adjusted)", y = "Experimentally determined MFI") +
  #geom_abline(slope = mfi_sort_calcd_model$coefficients[2], intercept = mfi_sort_calcd_model$coefficients[1], alpha = 0.2, size = 5) + 
  geom_point(data = complete_frame3 %>% filter(!is.na(mfi_individual)), aes(x = calibrated_score, y = mfi_individual)) +
  geom_text_repel(data = complete_frame3 %>% filter(!is.na(mfi_individual)), aes(x = calibrated_score, y = mfi_individual, label = sequence), color = "red", alpha = 0.6, size = 2) + NULL; Individual_vs_4way_sort_calcd_scatterplot
```

![](Kozak_files/figure-gfm/Use%20this%20model%20to%20transform%20the%204-way%20sort%20values%20into%20calculated%20MFI%20equivalents-2.png)<!-- -->

``` r
ggsave(file = "Plots/Individual_vs_4way_sort_calcd_scatterplot.png", Individual_vs_4way_sort_calcd_scatterplot, height = 3, width = 4)
ggsave(file = "Plots/Individual_vs_4way_sort_calcd_scatterplot.pdf", Individual_vs_4way_sort_calcd_scatterplot, height = 3, width = 3.35)

paste("Correlation coefficients between 4-way sort data logistic curve corrected and individually assessed MFI -->","Pearson's:",
round(cor(log10(complete_frame3$calibrated_score), log10(complete_frame3$mfi_individual), method = "pearson", use = "complete")^2,3),"Spearman's:",
round(cor(log10(complete_frame3$calibrated_score), log10(complete_frame3$mfi_individual), method = "spearman", use = "complete")^2,3))
```

    ## [1] "Correlation coefficients between 4-way sort data logistic curve corrected and individually assessed MFI --> Pearson's: 0.92 Spearman's: 0.856"

``` r
density_4way_sort_calcd_plot <- ggplot() + theme_bw() +theme(panel.grid.minor = element_blank()) + 
  scale_x_log10(limits = c(0.003,1.5)) + 
  labs(x = "Predicted MFI calculated by sequencing", y = "Experimentally determined MFI") +
  geom_histogram(data = complete_frame3, aes(x = calibrated_score), color = "black", fill = "grey80", binwidth = 0.05)
density_4way_sort_calcd_plot
```

![](Kozak_files/figure-gfm/Make%20a%20singular%20plot%20showing%20the%20above%20scatterplot%20and%20a%20densityplot%20of%20the%20scores%20by%20sequencing-1.png)<!-- -->

``` r
complete_frame3 <- complete_frame3 %>% filter(!(sequence %in% c("GAGTTAATG","AAACTGATG","AGTAGAATG","CCCATGATG")))
```

``` r
## Next, import the data from Noderer et al
noderer_data <- read.delim(file = "Data/Noderer.tsv", stringsAsFactors = F)
noderer_data$sequence <- substr(noderer_data$sequence, 1, 9)
noderer_data$sequence <- gsub("U","T",noderer_data$sequence)
noderer_data2 <- noderer_data %>% group_by(sequence) %>% summarize(mean_efficiency = mean(efficiency, na.rm = T), .groups = "drop")

## Combining the Noderer and 4-way sort data into a single dataframe
complete_frame4 <- merge(complete_frame3[,c("sequence","sort_geomean_log10","sort_geomean","sort_upper_conf","sort_lower_conf","mfi_lm_norm","calibrated_score","mfi_individual")],
                         noderer_data2[,c("sequence","mean_efficiency")], by = "sequence", all.x = T)
colnames(complete_frame4) <- c("sequence","sort_geomean_log10","sort_geomean","sort_upper_conf","sort_lower_conf","mfi_lm_norm","calibrated_score","mfi_individual","noderer")
```

``` r
Noderer_vs_individual_scatterplot <- ggplot() + 
  labs(x = "MFI individual", y = "Noderer score") + theme(panel.grid = element_blank()) + 
  scale_x_log10(limits = c(0.0065,1.6)) + scale_y_continuous(limits = c(20,125)) +
  geom_point(data = complete_frame4 %>% filter(!is.na(mfi_individual)), aes(x = mfi_individual, y = noderer)) +
  geom_point(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = mfi_individual, y = noderer), fill = "red", color = "black", shape = 21) +
  geom_text_repel(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = mfi_individual, y = noderer, label = sequence), color = "red", size = 3) +
  NULL; Noderer_vs_individual_scatterplot
```

![](Kozak_files/figure-gfm/Noderer%20score%20vs%20individual%20MFI%20graph-1.png)<!-- -->

``` r
ggsave(file = "plots/Noderer_vs_individual_scatterplot.png", Noderer_vs_individual_scatterplot, height = 4, width = 5)
ggsave(file = "plots/Noderer_vs_individual_scatterplot.pdf", Noderer_vs_individual_scatterplot, height = 1.6, width = 3.32)

Noderer_vs_calibrated_score_scatterplot <- ggplot() + theme(panel.grid = element_blank()) + 
  labs(x = "MFI calibrated sort-seq score", y = "Noderer score") +
  scale_x_log10(limits = c(0.0065,1.6)) + scale_y_continuous(limits = c(20,125)) +
  geom_point(data = complete_frame4, aes(x = calibrated_score, y = noderer), alpha = 0.03) +
  geom_point(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = calibrated_score, y = noderer), fill = "red", color = "black", shape = 21) +
  geom_text_repel(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = calibrated_score, y = noderer, label = sequence), color = "red", size = 3) +
  NULL; Noderer_vs_calibrated_score_scatterplot
```

![](Kozak_files/figure-gfm/Noderer%20score%20vs%20individual%20MFI%20graph-2.png)<!-- -->

``` r
ggsave(file = "plots/Noderer_vs_calibrated_score_scatterplot.png", Noderer_vs_calibrated_score_scatterplot, height = 4, width = 4)
ggsave(file = "plots/Noderer_vs_calibrated_score_scatterplot.pdf", Noderer_vs_calibrated_score_scatterplot, height = 1.6, width = 3.32)

paste("Correlation coefficients between Noderer data and our sequencing data -->","Pearson's r^2:",
round(cor(log10(complete_frame4$calibrated_score), complete_frame4$noderer, method = "pearson", use = "complete")^2,2),"Spearman's rho^2:",
round(cor(log10(complete_frame4$calibrated_score), complete_frame4$noderer, method = "spearman", use = "complete")^2,2))
```

    ## [1] "Correlation coefficients between Noderer data and our sequencing data --> Pearson's r^2: 0.67 Spearman's rho^2: 0.6"

## Seeing if we can model what is going on

``` r
## Trying a position weight matrix for the calibrated score

pwm_dataframe <- data.frame("n6" = rep(0,4),"n5" = rep(0,4),"n4" = rep(0,4),"n3" = rep(0,4),"n2" = rep(0,4),"n1" = rep(0,4))
rownames(pwm_dataframe) <- c("A","C","G","T")

for(x in 1:nrow(complete_frame4)){
  temp_seq <- substr(complete_frame4$sequence[x],1,6)
  temp_score <- as.numeric(complete_frame4$calibrated_score[x])
  pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,1,1),"n6"] <- pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,1,1),"n6"] + temp_score
  pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,2,2),"n5"] <- pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,2,2),"n5"] + temp_score
  pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,3,3),"n4"] <- pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,3,3),"n4"] + temp_score
  pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,4,4),"n3"] <- pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,4,4),"n3"] + temp_score
  pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,5,5),"n2"] <- pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,5,5),"n2"] + temp_score
  pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,6,6),"n1"] <- pwm_dataframe[rownames(pwm_dataframe) == substr(temp_seq,6,6),"n1"] + temp_score
}
pwm_dataframe2 <- pwm_dataframe / colSums(pwm_dataframe)
pwm_matrix <- as.matrix(pwm_dataframe2)

## Function for working out the position weight matrix value
pwm <- function(freq, total, bg=0.25){
  #using the formulae above
  p <- (freq + (sqrt(total) * 1/4)) / (total + (4 * (sqrt(total) * 1/4)))
  log2(p/bg)
}
pwm_matrix2 <- pwm(pwm_matrix,6)

## Logo plot 
proportion <- function(x){
   rs <- sum(x);
   return(x / rs);
}

## Bar chart
pwm_matrix_melt <- melt(pwm_matrix)

PWM_barplot <- ggplot() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  labs(x = "Position along Kozak", y = "Weighted\nfrequency") + 
  geom_bar(data = pwm_matrix_melt, aes(x = Var2, y = value, fill = Var1), stat = "identity", color = "black")
PWM_barplot
```

![](Kozak_files/figure-gfm/PSSM%20for%20calibrated_score-1.png)<!-- -->

``` r
ggsave(file = "plots/PWM_barplot.pdf", PWM_barplot, height = 1.2, width = 3)

PWM_pointplot <- ggplot() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position = "top") + 
  labs(x = "Position along Kozak", y = "Weighted\nfrequency") + 
  geom_hline(yintercept = 0.25, linetype = 2, alpha = 0.4) + 
  geom_point(data = pwm_matrix_melt, aes(x = Var2, y = value, color = Var1, fill = Var1), stat = "identity", shape = 21, size = 2, alpha = 0.5) +
  NULL; PWM_pointplot
```

![](Kozak_files/figure-gfm/PSSM%20for%20calibrated_score-2.png)<!-- -->

``` r
ggsave(file = "plots/PWM_pointplot.pdf", PWM_pointplot, height = 2, width = 3.4)
```

``` r
nt_list = c("A","C","G","T")

pos3_ac_list = c()
pos3_ag_list = c()
pos3_at_list = c()
for(x6 in nt_list){
  for(x5 in nt_list){
    for(x4 in nt_list){
      for(x2 in nt_list){
        for(x1 in nt_list){
          pos3_string_A <- paste(x6,x5,x4,"A",x2,x1,"ATG",sep="")
          pos3_string_C <- paste(x6,x5,x4,"C",x2,x1,"ATG",sep="")
          pos3_string_G <- paste(x6,x5,x4,"G",x2,x1,"ATG",sep="")
          pos3_string_T <- paste(x6,x5,x4,"T",x2,x1,"ATG",sep="")
          pos3_ac_list = c(pos3_ac_list,complete_frame4[complete_frame4$sequence == pos3_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos3_string_C,"calibrated_score"])
          pos3_ag_list = c(pos3_ag_list,complete_frame4[complete_frame4$sequence == pos3_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos3_string_G,"calibrated_score"])
          pos3_at_list = c(pos3_at_list,complete_frame4[complete_frame4$sequence == pos3_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos3_string_T,"calibrated_score"])
        }
      }
    }
  }
}
  
pos_3_difference <- data.frame("values" = c(pos3_ac_list,pos3_ag_list,pos3_at_list), "subset" = c(rep("A/C",length(pos3_ac_list)),rep("A/G",length(pos3_ag_list)),rep("A/T",length(pos3_at_list))))
pos_3_difference_summary <- pos_3_difference %>% group_by(subset) %>% summarize(geomean = 10^mean(log10(values)), upper_conf = quantile(values,0.95), lower_conf = quantile(values,0.05))


pos1_ac_list = c(); pos1_ag_list = c(); pos1_at_list = c()
for(x6 in nt_list){
  for(x5 in nt_list){
    for(x4 in nt_list){
      for(x3 in nt_list){
        for(x2 in nt_list){
          pos1_string_A <- paste(x6,x5,x4,x3,x2,"A","ATG",sep="")
          pos1_string_C <- paste(x6,x5,x4,x3,x2,"C","ATG",sep="")
          pos1_string_G <- paste(x6,x5,x4,x3,x2,"G","ATG",sep="")
          pos1_string_T <- paste(x6,x5,x4,x3,x2,"T","ATG",sep="")
          pos1_ac_list = c(pos1_ac_list,complete_frame4[complete_frame4$sequence == pos1_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos1_string_C,"calibrated_score"])
          pos1_ag_list = c(pos1_ag_list,complete_frame4[complete_frame4$sequence == pos1_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos1_string_G,"calibrated_score"])
          pos1_at_list = c(pos1_at_list,complete_frame4[complete_frame4$sequence == pos1_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos1_string_T,"calibrated_score"])
        }
      }
    }
  }
}
pos_1_difference <- data.frame("values" = c(pos1_ac_list,pos1_ag_list,pos1_at_list), "subset" = c(rep("A/C",length(pos1_ac_list)),rep("A/G",length(pos1_ag_list)),rep("A/T",length(pos1_at_list))))
pos_1_difference_summary <- pos_1_difference %>% group_by(subset) %>% summarize(geomean = 10^mean(log10(values)), upper_conf = quantile(values,0.95), lower_conf = quantile(values,0.05))


pos2_ac_list = c(); pos2_ag_list = c(); pos2_at_list = c()
for(x6 in nt_list){
  for(x5 in nt_list){
    for(x4 in nt_list){
      for(x3 in nt_list){
        for(x1 in nt_list){
          pos2_string_A <- paste(x6,x5,x4,x3,"A",x1,"ATG",sep="")
          pos2_string_C <- paste(x6,x5,x4,x3,"C",x1,"ATG",sep="")
          pos2_string_G <- paste(x6,x5,x4,x3,"G",x1,"ATG",sep="")
          pos2_string_T <- paste(x6,x5,x4,x3,"T",x1,"ATG",sep="")
          pos2_ac_list = c(pos2_ac_list,complete_frame4[complete_frame4$sequence == pos2_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos2_string_C,"calibrated_score"])
          pos2_ag_list = c(pos2_ag_list,complete_frame4[complete_frame4$sequence == pos2_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos2_string_G,"calibrated_score"])
          pos2_at_list = c(pos2_at_list,complete_frame4[complete_frame4$sequence == pos2_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos2_string_T,"calibrated_score"])
        }
      }
    }
  }
}
pos_2_difference <- data.frame("values" = c(pos2_ac_list,pos2_ag_list,pos2_at_list), "subset" = c(rep("A/C",length(pos2_ac_list)),rep("A/G",length(pos2_ag_list)),rep("A/T",length(pos2_at_list))))
pos_2_difference_summary <- pos_2_difference %>% group_by(subset) %>% summarize(geomean = 10^mean(log10(values)), upper_conf = quantile(values,0.95), lower_conf = quantile(values,0.05))


pos4_ac_list = c()
pos4_ag_list = c()
pos4_at_list = c()
for(x6 in nt_list){
  for(x5 in nt_list){
    for(x3 in nt_list){
      for(x2 in nt_list){
        for(x1 in nt_list){
          pos4_string_A <- paste(x6,x5,"A",x3,x2,x1,"ATG",sep="")
          pos4_string_C <- paste(x6,x5,"C",x3,x2,x1,"ATG",sep="")
          pos4_string_G <- paste(x6,x5,"G",x3,x2,x1,"ATG",sep="")
          pos4_string_T <- paste(x6,x5,"T",x3,x2,x1,"ATG",sep="")
          pos4_ac_list = c(pos4_ac_list,complete_frame4[complete_frame4$sequence == pos4_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos4_string_C,"calibrated_score"])
          pos4_ag_list = c(pos4_ag_list,complete_frame4[complete_frame4$sequence == pos4_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos4_string_G,"calibrated_score"])
          pos4_at_list = c(pos4_at_list,complete_frame4[complete_frame4$sequence == pos4_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos4_string_T,"calibrated_score"])
        }
      }
    }
  }
}
  
pos_4_difference <- data.frame("values" = c(pos4_ac_list,pos4_ag_list,pos4_at_list), "subset" = c(rep("A/C",length(pos4_ac_list)),rep("A/G",length(pos4_ag_list)),rep("A/T",length(pos4_at_list))))
pos_4_difference_summary <- pos_4_difference %>% group_by(subset) %>% summarize(geomean = 10^mean(log10(values)), upper_conf = quantile(values,0.95), lower_conf = quantile(values,0.05))


pos5_ac_list = c()
pos5_ag_list = c()
pos5_at_list = c()
for(x6 in nt_list){
  for(x4 in nt_list){
    for(x3 in nt_list){
      for(x2 in nt_list){
        for(x1 in nt_list){
          pos5_string_A <- paste(x6,"A",x4,x3,x2,x1,"ATG",sep="")
          pos5_string_C <- paste(x6,"C",x4,x3,x2,x1,"ATG",sep="")
          pos5_string_G <- paste(x6,"G",x4,x3,x2,x1,"ATG",sep="")
          pos5_string_T <- paste(x6,"T",x4,x3,x2,x1,"ATG",sep="")
          pos5_ac_list = c(pos5_ac_list,complete_frame4[complete_frame4$sequence == pos5_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos5_string_C,"calibrated_score"])
          pos5_ag_list = c(pos5_ag_list,complete_frame4[complete_frame4$sequence == pos5_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos5_string_G,"calibrated_score"])
          pos5_at_list = c(pos5_at_list,complete_frame4[complete_frame4$sequence == pos5_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos5_string_T,"calibrated_score"])
        }
      }
    }
  }
}
  
pos_5_difference <- data.frame("values" = c(pos5_ac_list,pos5_ag_list,pos5_at_list), "subset" = c(rep("A/C",length(pos5_ac_list)),rep("A/G",length(pos5_ag_list)),rep("A/T",length(pos5_at_list))))
pos_5_difference_summary <- pos_5_difference %>% group_by(subset) %>% summarize(geomean = 10^mean(log10(values)), upper_conf = quantile(values,0.95), lower_conf = quantile(values,0.05))


pos6_ac_list = c()
pos6_ag_list = c()
pos6_at_list = c()
for(x5 in nt_list){
  for(x4 in nt_list){
    for(x3 in nt_list){
      for(x2 in nt_list){
        for(x1 in nt_list){
          pos6_string_A <- paste("A",x5,x4,x3,x2,x1,"ATG",sep="")
          pos6_string_C <- paste("C",x5,x4,x3,x2,x1,"ATG",sep="")
          pos6_string_G <- paste("G",x5,x4,x3,x2,x1,"ATG",sep="")
          pos6_string_T <- paste("T",x5,x4,x3,x2,x1,"ATG",sep="")
          pos6_ac_list = c(pos6_ac_list,complete_frame4[complete_frame4$sequence == pos6_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos6_string_C,"calibrated_score"])
          pos6_ag_list = c(pos6_ag_list,complete_frame4[complete_frame4$sequence == pos6_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos6_string_G,"calibrated_score"])
          pos6_at_list = c(pos6_at_list,complete_frame4[complete_frame4$sequence == pos6_string_A,"calibrated_score"]/complete_frame4[complete_frame4$sequence == pos6_string_T,"calibrated_score"])
        }
      }
    }
  }
}
  
pos_6_difference <- data.frame("values" = c(pos6_ac_list,pos6_ag_list,pos6_at_list), "subset" = c(rep("A/C",length(pos6_ac_list)),rep("A/G",length(pos6_ag_list)),rep("A/T",length(pos6_at_list))))
pos_6_difference_summary <- pos_6_difference %>% group_by(subset) %>% summarize(geomean = 10^mean(log10(values)), upper_conf = quantile(values,0.95), lower_conf = quantile(values,0.05))
```

``` r
pairwise_fold_diff <- rbind(pos_1_difference %>% mutate(position = "-1"),pos_2_difference %>% mutate(position = "-2"),pos_3_difference %>% mutate(position = "-3"),pos_4_difference %>% mutate(position = "-4"),pos_5_difference %>% mutate(position = "-5"),pos_6_difference %>% mutate(position = "-6"))

pairwise_summary_fold_diff <- rbind(pos_1_difference_summary %>% mutate(position = "-1"),pos_2_difference_summary %>% mutate(position = "-2"),pos_3_difference_summary %>% mutate(position = "-3"),pos_4_difference_summary %>% mutate(position = "-4"),pos_5_difference_summary %>% mutate(position = "-5"),pos_6_difference_summary %>% mutate(position = "-6"))

pairwise_fold_diff$position <- factor(pairwise_fold_diff$position, levels = c("-6","-5","-4","-3","-2","-1"))
pairwise_summary_fold_diff$position <- factor(pairwise_summary_fold_diff$position, levels = c("-6","-5","-4","-3","-2","-1"))

Pairwise_fold_diff <- ggplot() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position = "top") + 
  scale_y_log10(limits = c(1e-2,1e3), breaks = c(0.01,0.1,1,10,100,1000)) + 
  labs(x = "Position along Kozak", y = "Fold difference") + 
  geom_hline(yintercept = 1, pairwise_summary_fold_diff = 2, alpha = 0.2) + 
  geom_jitter(data = pairwise_fold_diff, aes(x = position, y = values, color = subset, fill = subset), position  = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.5), stat = "identity", shape = 21, size = 0.5, alpha = 0.01) + 
  geom_point(data = pairwise_summary_fold_diff, aes(x = position, y = geomean, color = subset, fill = subset), position  = position_dodge(width = 0.5), stat = "identity", shape = 21, size = 1, alpha = 1, color = "black") +
  NULL
Pairwise_fold_diff
```

![](Kozak_files/figure-gfm/Plotting%20the%20pairwise%20differences%20in%20efficiency-1.png)<!-- -->

``` r
ggsave(file = "plots/Pairwise_fold_diff.pdf", Pairwise_fold_diff, height = 2.2, width = 3.4)
```

## See if we can model the protein abundance amount using a random forest model

``` r
complete_frame4_for_rf <- complete_frame4[c("calibrated_score","sequence")] %>% filter(!is.na(calibrated_score))
complete_frame4_for_rf$n6 <- substr(complete_frame4_for_rf$sequence,1,1)
complete_frame4_for_rf$n5 <- substr(complete_frame4_for_rf$sequence,2,2)
complete_frame4_for_rf$n4 <- substr(complete_frame4_for_rf$sequence,3,3)
complete_frame4_for_rf$n3 <- substr(complete_frame4_for_rf$sequence,4,4)
complete_frame4_for_rf$n2 <- substr(complete_frame4_for_rf$sequence,5,5)
complete_frame4_for_rf$n1 <- substr(complete_frame4_for_rf$sequence,6,6)
complete_frame4_for_rf$n65 <- substr(complete_frame4_for_rf$sequence,1,2)
complete_frame4_for_rf$n54 <- substr(complete_frame4_for_rf$sequence,2,3)
complete_frame4_for_rf$n43 <- substr(complete_frame4_for_rf$sequence,3,4)
complete_frame4_for_rf$n32 <- substr(complete_frame4_for_rf$sequence,4,5)
complete_frame4_for_rf$n21 <- substr(complete_frame4_for_rf$sequence,5,6)
complete_frame4_for_rf$n654 <- substr(complete_frame4_for_rf$sequence,1,3)
complete_frame4_for_rf$n543 <- substr(complete_frame4_for_rf$sequence,2,4)
complete_frame4_for_rf$n432 <- substr(complete_frame4_for_rf$sequence,3,5)
complete_frame4_for_rf$n321 <- substr(complete_frame4_for_rf$sequence,4,6)
complete_frame4_for_rf$uatg0a <- substr(complete_frame4_for_rf$sequence,1,3) == "ATG"
complete_frame4_for_rf$uatg1 <- substr(complete_frame4_for_rf$sequence,2,4) == "ATG"
complete_frame4_for_rf$uatg2 <- substr(complete_frame4_for_rf$sequence,3,5) == "ATG"
complete_frame4_for_rf$uatg0b <- substr(complete_frame4_for_rf$sequence,4,6) == "ATG"

# This is going to be done with the RandomForest package
# Calculate the size of each of the data sets:
data_set_size <- floor(nrow(complete_frame4_for_rf)*0.9)
# Generate a random sample of "data_set_size" indexes
indexes <- sample(1:nrow(complete_frame4_for_rf), size = data_set_size)
# Assign the data to the correct sets
training1 <- complete_frame4_for_rf[indexes, !(colnames(complete_frame4_for_rf) %in% "sequence")]
test1 <- complete_frame4_for_rf[-indexes, !(colnames(complete_frame4_for_rf) %in% "sequence")]

rf_regression = randomForest(calibrated_score ~ ., data=training1, ntree=200, mtry=3, importance=TRUE)

print(rf_regression)
```

    ## 
    ## Call:
    ##  randomForest(formula = calibrated_score ~ ., data = training1,      ntree = 200, mtry = 3, importance = TRUE) 
    ##                Type of random forest: regression
    ##                      Number of trees: 200
    ## No. of variables tried at each split: 3
    ## 
    ##           Mean of squared residuals: 0.06790604
    ##                     % Var explained: 63.07

``` r
plot(rf_regression)
```

![](Kozak_files/figure-gfm/Try%20random%20forest-1.png)<!-- -->

``` r
feature_frame <- data.frame("IncMSE" = importance(rf_regression,type = 1))
feature_frame$name <- rownames(feature_frame)
feature_frame <- feature_frame[order(feature_frame$X.IncMSE),]
feature_frame$name <- factor(feature_frame$name, levels = feature_frame$name)

rf_regression_importance_plot <- ggplot() + theme_bw() + 
  labs(x = "% Increase MSE\nupon permutation", y = NULL) +
  geom_point(data = feature_frame, aes(x = X.IncMSE, y = name))
rf_regression_importance_plot
```

![](Kozak_files/figure-gfm/Try%20random%20forest-2.png)<!-- -->

``` r
ggsave(file = "plots/rf_regression_importance_plot.png", rf_regression_importance_plot, height = 3.2, width = 2)
ggsave(file = "plots/rf_regression_importance_plot.pdf", rf_regression_importance_plot, height = 3.2, width = 2)

feature_Frame2 <- data.frame(varImpPlot(rf_regression))
```

![](Kozak_files/figure-gfm/Try%20random%20forest-3.png)<!-- -->

``` r
## We can now use the trained model on the validation dataset
prediction_test <- predict(rf_regression,test1[,colnames(test1) != c("calibrated_score","sequence")])

test1$predicted_scores <- prediction_test

ggplot() + scale_x_log10() + scale_y_log10() + geom_point(data = test1, aes(x = calibrated_score, y = predicted_scores))
```

![](Kozak_files/figure-gfm/Try%20random%20forest-4.png)<!-- -->

``` r
complete_frame4$rf_predict_mfi <- predict(rf_regression,complete_frame4_for_rf[,colnames(test1) != c("calibrated_score","sequence")])

ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = complete_frame4, aes(x = calibrated_score, y = rf_predict_mfi), alpha = 0.2)
```

![](Kozak_files/figure-gfm/Try%20random%20forest-5.png)<!-- -->

``` r
RF_refined_histogram <- ggplot() + theme(panel.grid.major.y = element_blank()) + 
  labs(x = "RF predicted MFI", y = "Number of variants") +
  scale_x_log10() + 
  geom_histogram(data = complete_frame4, aes(x = rf_predict_mfi), binwidth = 0.1, color = "black", fill = "grey75")
RF_refined_histogram
```

![](Kozak_files/figure-gfm/Try%20random%20forest-6.png)<!-- -->

``` r
ggsave(file = "plots/RF_refined_histogram.pdf", RF_refined_histogram, height = 3, width = 4)

## Did the correlation with the individual datapoints improve?
RF_refined_individual_scatterplot <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(x = "RF-refined scores", y = "Individual score") +
  scale_x_log10(limits = c(0.008,1)) + scale_y_log10() + 
  geom_point(data = complete_frame4, aes(x = rf_predict_mfi, y = mfi_individual)) +
  geom_text_repel(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = rf_predict_mfi, y = mfi_individual, label = sequence), color = "orange") +
  NULL; RF_refined_individual_scatterplot
```

![](Kozak_files/figure-gfm/Try%20random%20forest-7.png)<!-- -->

``` r
RF_refined_individual_scatterplot2 <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(x = "Sortseq predicted MFI (linear model)", y = "Individual score") +
  scale_x_log10(limits = c(0.008,1)) + scale_y_log10() + 
  geom_point(data = complete_frame4 , aes(x = mfi_lm_norm, y = mfi_individual)) + 
  geom_text_repel(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = mfi_lm_norm, y = mfi_individual, label = sequence), color = "orange") +
  NULL
RF_refined_individual_scatterplot2
```

![](Kozak_files/figure-gfm/Try%20random%20forest-8.png)<!-- -->

``` r
RF_refined_individual_scatterplot3 <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(x = "Sortseq predicted MFI (log adjusted)", y = "Individual score") +
  scale_x_log10(limits = c(0.008,1)) + scale_y_log10() + 
  geom_point(data = complete_frame4, aes(x = calibrated_score, y = mfi_individual)) +
  geom_text_repel(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = calibrated_score, y = mfi_individual, label = sequence), color = "orange") +
  NULL
RF_refined_individual_scatterplot3
```

![](Kozak_files/figure-gfm/Try%20random%20forest-9.png)<!-- -->

``` r
paste("Correlation coefficients between our RF-refined scores and the individually assessed MFIs -->","Pearson's:",
round(cor(log10(complete_frame4$calibrated_score), log10(complete_frame4$mfi_individual), method = "pearson", use = "complete"),2),"Spearman's:",
round(cor(log10(complete_frame4$calibrated_score), log10(complete_frame4$mfi_individual), method = "spearman", use = "complete"),2))
```

    ## [1] "Correlation coefficients between our RF-refined scores and the individually assessed MFIs --> Pearson's: 0.96 Spearman's: 0.92"

``` r
paste("Correlation coefficients between our RF-refined scores and the individually assessed MFIs -->","Pearson's:",
round(cor(log10(complete_frame4$rf_predict_mfi), log10(complete_frame4$mfi_individual), method = "pearson", use = "complete"),2),"Spearman's:",
round(cor(log10(complete_frame4$rf_predict_mfi), log10(complete_frame4$mfi_individual), method = "spearman", use = "complete"),2))
```

    ## [1] "Correlation coefficients between our RF-refined scores and the individually assessed MFIs --> Pearson's: 0.89 Spearman's: 0.91"

``` r
## Comparing to the Noderer et al dataset again with the RF-refined scores

outlier_lm <- lm(log10(complete_frame4$rf_predict_mfi) ~ complete_frame4$noderer)
complete_frame4$rf_residuals <- outlier_lm$residuals

ggplot() + geom_histogram(data = complete_frame4, aes(x = rf_residuals))
```

![](Kozak_files/figure-gfm/Comparing%20to%20the%20Noderer%20et%20al%20dataset%20again%20with%20the%20RF-refined%20scores-1.png)<!-- -->

``` r
## Try to fit a normal curve to the above distribution? This is done with the MASS package.
fit <- fitdistr(complete_frame4$rf_residuals, "normal")
para <- fit$estimate

normal_dist_fit <- data.frame("prob" = rnorm(1000000, mean = para[1], sd = para[2]))

ggplot() + geom_density(data = complete_frame4, aes(x = rf_residuals)) +
  geom_density(data = normal_dist_fit, aes(x = prob))
```

![](Kozak_files/figure-gfm/Comparing%20to%20the%20Noderer%20et%20al%20dataset%20again%20with%20the%20RF-refined%20scores-2.png)<!-- -->

``` r
residual_outlier_cutoff1 <- quantile(normal_dist_fit$prob,0.01)

outlier_set <- complete_frame4 %>% arrange(rf_residuals) %>% filter(rf_residuals < residual_outlier_cutoff1) %>% mutate(uorf1 = substr(sequence,2,5))

outlier_set$ratio <- outlier_set$noderer / outlier_set$rf_predict_mfi

complete_frame4b <- complete_frame4 %>% mutate(uorf1 = substr(sequence,2,5))

outlier_set_atgg <- complete_frame4b %>% filter(uorf1 == "ATGG")
outlier_set_atgh <- complete_frame4b %>% filter(uorf1 %in% c("ATGA","ATGC","ATGT"))
outlier_set_gatg <- complete_frame4b %>% filter(uorf1 %in% c("GATG","AATG","CATG"))
outlier_set_taa <- complete_frame4b %>% filter(uorf1 %in% c("TAAA","TAAC","TAAG","TAAT"))

outlier_set_remaining <- outlier_set %>% filter(!(uorf1 %in% c("ATGG", "ATGA", "ATGC", "ATGT", "GATG","AATG","CATG")))

RF_refined_noderer_scatterplot <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(x = "RF-refined scores", y = "Noderer score") +
  scale_x_log10(limits = c(0.01,1.1), expand = c(0,0)) + #scale_x_log10(limits = c(0.008,9)) + 
  scale_y_continuous(limits = c(20,125)) +
  geom_point(data = complete_frame4, aes(x = rf_predict_mfi, y = noderer), alpha = 0.05) +
  geom_point(data = outlier_set_atgg, aes(x = rf_predict_mfi, y = noderer), alpha = 0.1, color = "orange") +
  geom_point(data = outlier_set_atgh, aes(x = rf_predict_mfi, y = noderer), alpha = 0.1, color = "blue") +
  #geom_point(data = outlier_set_gatg, aes(x = rf_predict_mfi, y = noderer), alpha = 0.1, color = "magenta") +
  #geom_point(data = outlier_set_gatg, aes(x = rf_predict_mfi, y = noderer), alpha = 0.5, color = "red") +
  geom_text_repel(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = rf_predict_mfi, y = noderer, label = sequence), color = "red") +
  geom_point(data = complete_frame4 %>% filter(sequence %in% c("TTCATCATG","GCGCGCATG")), aes(x = rf_predict_mfi, y = noderer), color = "black", fill = "red", shape = 21) +
  NULL; RF_refined_noderer_scatterplot
```

![](Kozak_files/figure-gfm/Comparing%20to%20the%20Noderer%20et%20al%20dataset%20again%20with%20the%20RF-refined%20scores-3.png)<!-- -->

``` r
ggsave(file = "plots/RF_refined_noderer_scatterplot.png", RF_refined_noderer_scatterplot, height = 4, width = 5)
ggsave(file = "plots/RF_refined_noderer_scatterplot.pdf", RF_refined_noderer_scatterplot, height = 2, width = 3.32)

RF_refined_calibrated_scatterplot <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(x = "RF-imputed score", y = "Calibrated score") +
  scale_x_log10(limits = c(0.01,1.1), expand = c(0,0)) + scale_y_log10(limits = c(0.008,9)) + 
  #scale_y_continuous(limits = c(20,125)) +
  geom_point(data = complete_frame4, aes(x = rf_predict_mfi, y = calibrated_score), alpha = 0.05) +
  geom_point(data = outlier_set_atgg, aes(x = rf_predict_mfi, y = calibrated_score), alpha = 0.1, color = "red") +
  geom_point(data = outlier_set_atgh, aes(x = rf_predict_mfi, y = calibrated_score), alpha = 0.1, color = "blue") +
  NULL; RF_refined_calibrated_scatterplot
```

![](Kozak_files/figure-gfm/Comparing%20to%20the%20Noderer%20et%20al%20dataset%20again%20with%20the%20RF-refined%20scores-4.png)<!-- -->

``` r
paste("Correlation coefficients between Calibrated scores and the Noderer scores -->","Pearson's r^2:",
round(cor(complete_frame4$calibrated_score, complete_frame4$noderer, method = "pearson", use = "complete")^2,2),"Spearman's rho^2:",
round(cor(complete_frame4$calibrated_score, complete_frame4$noderer, method = "spearman", use = "complete")^2,2))
```

    ## [1] "Correlation coefficients between Calibrated scores and the Noderer scores --> Pearson's r^2: 0.56 Spearman's rho^2: 0.6"

``` r
paste("Correlation coefficients between log10 of Calibrated scores and the Noderer scores -->","Pearson's r^2:",
round(cor(log10(complete_frame4$calibrated_score), complete_frame4$noderer, method = "pearson", use = "complete")^2,2),"Spearman's rho^2:",
round(cor(log10(complete_frame4$calibrated_score), complete_frame4$noderer, method = "spearman", use = "complete")^2,2))
```

    ## [1] "Correlation coefficients between log10 of Calibrated scores and the Noderer scores --> Pearson's r^2: 0.67 Spearman's rho^2: 0.6"

``` r
paste("Correlation coefficients between log10 of RF-refined scores and the Noderer scores -->","Pearson's r^2:",
round(cor(log10(complete_frame4$rf_predict_mfi), complete_frame4$noderer, method = "pearson", use = "complete")^2,2),"Spearman's rho^2:",
round(cor(log10(complete_frame4$rf_predict_mfi), complete_frame4$noderer, method = "spearman", use = "complete")^2,2))
```

    ## [1] "Correlation coefficients between log10 of RF-refined scores and the Noderer scores --> Pearson's r^2: 0.85 Spearman's rho^2: 0.76"

``` r
paste("The total number of variants taht we scored was", nrow(complete_frame4))
```

    ## [1] "The total number of variants taht we scored was 4042"

``` r
write.csv(file = "Output_datatables/Supp_table_1_Sortseq_scores.csv", complete_frame4, row.names = F)
```

## OK, this is where we are leaving all of the Kozak scoring. Now onto some applications.

``` r
convenient_Kozaks_table <- subset(complete_frame4, sequence %in% 
                                    c("GCCACCATG","ATTAATATG","CGTCCAATG","TTGCACATG","TATTTCATG","TGTTTTATG","CATTGTATG"))

Convenient_Kozaks_scatterplot_our_construct <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(x = "Individually assessed MFI", y = "Calibrated score") +
  scale_x_log10(limits = c(0.009,1.4)) + scale_y_log10(limits = c(0.009,1.4)) +
  geom_abline(slope = 1, alpha = 0.2, linetype = 2, size = 1.5) +
  geom_point(data = convenient_Kozaks_table, aes(x = mfi_individual, y = calibrated_score), alpha = 0.6, size = 2) +
  geom_text_repel(data = convenient_Kozaks_table, aes(x = mfi_individual, y = calibrated_score, label = sequence), color = "red", alpha = 0.6)
Convenient_Kozaks_scatterplot_our_construct
```

![](Kozak_files/figure-gfm/Coming%20up%20with%20a%20table%20of%20convenient%20Kozak%20sequence%20options-1.png)<!-- -->

``` r
ggsave(file = "plots/Convenient_Kozaks_scatterplot_our_construct.png", Convenient_Kozaks_scatterplot_our_construct, height = 3.6, width = 4)
ggsave(file = "plots/Convenient_Kozaks_scatterplot_our_construct_small.png", Convenient_Kozaks_scatterplot_our_construct, height = 2.2, width = 2.5)
```

``` r
Farrell_Kozaks_table <- subset(complete_frame4, sequence %in% 
                                    c("GCCACCATG","TATCTAATG","TATTTCATG","AATTTTATG"))

Farrell_Kozaks_scatterplot_our_construct <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  labs(x = "Individually assessed MFI", y = "Calibrated score") +
  scale_x_log10(limits = c(0.009,1.4)) + scale_y_log10(limits = c(0.009,1.4)) +
  geom_abline(slope = 1, alpha = 0.2, linetype = 2, size = 1.5) +
  geom_point(data = Farrell_Kozaks_table, aes(x = mfi_individual, y = calibrated_score), alpha = 0.6, size = 2) +
  geom_text_repel(data = Farrell_Kozaks_table, aes(x = mfi_individual, y = calibrated_score, label = sequence), color = "red", alpha = 0.6)
Farrell_Kozaks_scatterplot_our_construct
```

![](Kozak_files/figure-gfm/Looking%20at%20the%20set%20I%20previous%20used%20with%20the%20Bloom%20lab-1.png)<!-- -->

``` r
ggsave(file = "plots/Farrell_Kozaks_scatterplot_our_construct.png", Farrell_Kozaks_scatterplot_our_construct, height = 3.6, width = 4)
ggsave(file = "plots/Farrell_Kozaks_scatterplot_our_construct_small.png", Farrell_Kozaks_scatterplot_our_construct, height = 2.2, width = 2.5)
```

``` r
acevedo <- read.csv(file = "Data/Avecedo_Supp_Table_S1.csv", header = T, stringsAsFactors = F)
acevedo$plus1 <- substr(acevedo$sequence,8,8)
#table(acevedo$plus1)
acevedo$neg4 <- substr(acevedo$sequence,1,4)
complete_acevedo_neg4 <- (data.frame(table(acevedo$neg4)) %>% filter(Freq >= 4) %>% mutate(Var1 = as.character(Var1)))

acevedo_preceding <- acevedo %>% filter(neg4 %in% c(complete_acevedo_neg4$Var1)) %>% group_by(neg4) %>% summarize(acevedo = mean(strength))

complete_frame4_final2 <- complete_frame4

complete_frame4_final2$neg4 <- substr(complete_frame4_final2$sequence, 3, 6)

complete_frame4_final2_acevedo <- complete_frame4_final2 %>% group_by(neg4) %>% summarize(calibrated_score = mean(calibrated_score), imputed_score = mean(rf_predict_mfi), noderer = mean(noderer))
complete_frame4_final2_acevedo2 <- merge(acevedo_preceding, complete_frame4_final2_acevedo, by = "neg4")

Avecedo_scatterplot <- ggplot() + theme(panel.grid = element_blank()) + 
  #scale_x_log10() + scale_y_log10() + 
  labs(x = "Avecedo score", y = "Calibrated score") +
  geom_point(data = complete_frame4_final2_acevedo2, aes(x = acevedo, calibrated_score), alpha = 0.4) +
  NULL; Avecedo_scatterplot
```

![](Kozak_files/figure-gfm/Compare%20to%20the%20Acevedo%20data-1.png)<!-- -->

``` r
  ggsave(file = "Plots/Avecedo_scatterplot.pdf", Avecedo_scatterplot, height = 2, width = 3)

paste("Pearons's r^2 between the calibrated scores and Avecedo data",round((cor(complete_frame4_final2_acevedo2$calibrated_score, complete_frame4_final2_acevedo2$acevedo, method = "pearson"))^2,2))
```

    ## [1] "Pearons's r^2 between the calibrated scores and Avecedo data 0.74"

``` r
paste("Spearman's rho^2 between the calibrated scores and Avecedo data",round((cor(complete_frame4_final2_acevedo2$calibrated_score, complete_frame4_final2_acevedo2$acevedo, method = "spearman"))^2,2))
```

    ## [1] "Spearman's rho^2 between the calibrated scores and Avecedo data 0.77"

``` r
ambrosini <- read.delim(file = "Data/GSE210035_highThroughput_rawcounts_sequences.tsv", sep = "\t", header = T, stringsAsFactors = F)

ambrosini$neg4 <- substr(ambrosini$library_sequences,1,4)

ambrosini_4nt <- ambrosini %>% group_by(neg4) %>% summarize(ag1 = sum(gate1_A), ag2 = sum(gate2_A), ag3 = sum(gate3_A), ag4 = sum(gate4_A),
                                                            bg1 = sum(gate1_B), bg2 = sum(gate2_B), bg3 = sum(gate3_B), bg4 = sum(gate4_B))

ambrosini_4nt_freq <- sweep(ambrosini_4nt[,2:9],2,colSums(ambrosini_4nt[,2:9]),`/`)
ambrosini_4nt_freq$neg4 <- ambrosini_4nt$neg4

ambrosini_4nt_freq$wa1 <- (ambrosini_4nt_freq$ag1 * 0 + ambrosini_4nt_freq$ag2 * 1/3 + ambrosini_4nt_freq$ag3 * 2/3 + ambrosini_4nt_freq$ag4) / (ambrosini_4nt_freq$ag1 + ambrosini_4nt_freq$ag2 + ambrosini_4nt_freq$ag3 + ambrosini_4nt_freq$ag4)

ambrosini_4nt_freq$wa2 <- (ambrosini_4nt_freq$bg1 * 0 + ambrosini_4nt_freq$bg2 * 1/3 + ambrosini_4nt_freq$bg3 * 2/3 + ambrosini_4nt_freq$bg4) / (ambrosini_4nt_freq$bg1 + ambrosini_4nt_freq$bg2 + ambrosini_4nt_freq$bg3 + ambrosini_4nt_freq$bg4)

ambrosini_4nt_freq$ambrosini <- (ambrosini_4nt_freq$wa1 + ambrosini_4nt_freq$wa2)/2

complete_frame4_final2_ambrosini <- merge(complete_frame4_final2_acevedo, ambrosini_4nt_freq[,c("neg4","ambrosini")], by = "neg4")

Ambrosini_scatterplot <- ggplot() + theme(panel.grid = element_blank()) +
  labs(x = "Ambrosini score", y = "Calibrated score") +
  geom_point(data = complete_frame4_final2_ambrosini, aes(x = ambrosini, calibrated_score), alpha = 0.4) +
  NULL; Ambrosini_scatterplot
```

![](Kozak_files/figure-gfm/Compare%20to%20Ambrosini%20data-1.png)<!-- -->

``` r
ggsave(file = "Plots/Ambrosini_scatterplot.pdf", Ambrosini_scatterplot, height = 2, width = 3)

paste("Pearson's r^2 value between our calibrated scores and the Ambrosini scores", cor(log10(complete_frame4_final2_ambrosini$calibrated_score), complete_frame4_final2_ambrosini$ambrosini, method = "pearson")^2)
```

    ## [1] "Pearson's r^2 value between our calibrated scores and the Ambrosini scores 0.551281236816424"

``` r
paste("Spearman's rho^2 value between our calibrated scores and the Ambrosini scores", cor(complete_frame4_final2_ambrosini$calibrated_score, complete_frame4_final2_ambrosini$ambrosini, method = "spearman")^2)
```

    ## [1] "Spearman's rho^2 value between our calibrated scores and the Ambrosini scores 0.562207917435208"

``` r
running_snv_df <- read.csv(file = "Output_datatables/Human_kozak_permutations.csv", header = T, stringsAsFactors = F)

## ORIGINAL SCRIPT FOR MAKING THE ABOVE FILE
## I'm commenting it out b/c it's computation-greedy, and I don't need to regenerate it each time if I have the above file exported

# snv_analysis_key <- complete_frame4[,c("sequence","calibrated_score")]
# snv_analysis <- complete_frame4[,c("sequence","calibrated_score")]
# 
# running_snv_df <- data.frame("orig_sequence" = NA,"sequence" = NA, "hgvs_conseq" = NA, "calibrated_score" = NA, "fold_diff" = NA)
# for(row in 1:nrow(snv_analysis)){
#   temp_wt_seq <- snv_analysis$sequence[row]
#   temp_wt_score <- snv_analysis$calibrated_score[row]
#   hgvs_conseq <- c()
#   sequence <- c()
#   for(pos in 1:6){
#     for(nt in c("A","C","G","T")){
#       hgvs_conseq <- c(hgvs_conseq, paste0("c.",pos-7,substr(temp_wt_seq,pos,pos),">",nt))
#       sequence <- c(sequence,paste0(substr(snv_analysis$sequence[row],0,pos-1),nt,substr(snv_analysis$sequence[row],pos+1,9)))
#     }
#   }
#   temp_combined <- data.frame(cbind(hgvs_conseq,sequence))
#   temp_subset <- merge(temp_combined, snv_analysis_key, by = "sequence") %>% filter(sequence != temp_wt_seq)
#   temp_subset$fold_diff <- temp_subset$calibrated_score / temp_wt_score
#   temp_subset$orig_sequence <- temp_wt_seq
#   ## Collect all data in a dataframe
#   running_snv_df <- rbind(running_snv_df, temp_subset)
# }
# 
# running_snv_df <- running_snv_df %>% filter(!is.na(orig_sequence))
# colnames(running_snv_df) <- c("kozak","mut_kozak","hgvs_conseq","calibrated_score","fold_diff")
# 
# write.csv(file = "Output_datatables/Human_kozak_permutations.csv", running_snv_df, row.names = F)
```

``` r
complete_frame4_likely_low <- complete_frame4 %>% filter(sort_upper_conf < 0.75)

clinvar <- read.csv(file = "Data/ClinVar/250310_clinvar_kozak_mutations.csv", header = T, stringsAsFactors = F) %>% filter(mutation_location == "kozak" & !(old_kozak %in% c("error:no_kozak_found", "error:mutation_not_match_kozak", "error:microsatellite")))

paste("The number of Kozak -6 to -1 nt variants found in ClinVar is:",nrow(clinvar))
```

    ## [1] "The number of Kozak -6 to -1 nt variants found in ClinVar is: 1826"

``` r
colnames(clinvar)[colnames(clinvar) == "GeneSymbol"] <- "gene"
colnames(clinvar)[colnames(clinvar) == "mutation"] <- "nt_change"
clinvar$hgvs_conseq <- paste0("c.",clinvar$nt_change)
clinvar$kozak <- paste0(clinvar$old_kozak,"ATG")
clinvar$mut_kozak <- paste0(clinvar$new_kozak,"ATG")

clinvar_confidently_low <- clinvar %>% filter(mut_kozak %in% complete_frame4_likely_low$sequence) %>% mutate(conf_low = "Yes")
clinvar_confidently_low_mutkozak_list <- unique(clinvar_confidently_low$mut_kozak)

paste("The number of Kozak -6 to -1 nt variants found in ClinVar that are confidently measured as potentially low is:",nrow(clinvar_confidently_low))
```

    ## [1] "The number of Kozak -6 to -1 nt variants found in ClinVar that are confidently measured as potentially low is: 347"

``` r
clinvar2 <- merge(clinvar, running_snv_df, by = c("hgvs_conseq","kozak","mut_kozak"), all.x = T)
clinvar4 <- merge(clinvar2, unique(clinvar_confidently_low[,c("mut_kozak", "conf_low")]), all.x = T)

acmg73 <- read.csv(file = "Data/ClinVar/ACMG73.csv", header = T, stringsAsFactors = F)

gene_omim_extractor <- function(text){
  gsub("[^0-9A-Z]","",strsplit(text,"MIM")[[1]][1])
}

acmg73$gene <- lapply(acmg73$gene_omim,gene_omim_extractor)
acmg73_list <- unique(acmg73$gene)

clinvar2_acmg <- clinvar2 %>% filter(gene %in% acmg73_list)
clinvar2_acmg$gene_kozak <- paste0(clinvar2_acmg$gene,"_",clinvar2_acmg$kozak)

Clinvar_Kozak_permutation_fold_densityplot <- ggplot() + theme(panel.grid.minor = element_blank()) + 
  scale_x_log10(breaks = c(0.01,0.1,1,10,100)) + 
  labs(x = "Fold change from reference Kozak", y = "Density") + 
  geom_hline(yintercept = 0) +
  geom_density(data = running_snv_df, aes(x = fold_diff, y = ..density..), fill = "black", alpha = 0.3, adjust = 0.1) +
  geom_density(data = clinvar2, aes(x = fold_diff, y = ..density..), fill = "red", alpha = 0.3, adjust = 0.5) +
  #geom_histogram(data = transcripts_acmg, aes(x = fold_diff, y = ..density..), fill = "red", alpha = 0.3) +
  NULL; Clinvar_Kozak_permutation_fold_densityplot
```

![](Kozak_files/figure-gfm/Clinvar%20variants%20of%20seemingly%20low%20score-1.png)<!-- -->

``` r
ggsave(file = "Plots/Clinvar_Kozak_permutation_fold_densityplot.pdf", Clinvar_Kozak_permutation_fold_densityplot, height = 1, width = 3)

Clinvar_Kozak_permutation_fold_densityplot <- ggplot() + theme(panel.grid = element_blank()) + 
  scale_x_log10(breaks = c(0.01,0.1,1,10,100), expand = c(0,0)) + 
  labs(x = "Fold change from reference Kozak", y = "Count") + 
  #geom_hline(yintercept = 0) +
  #geom_density(data = running_snv_df, aes(x = fold_diff, y = ..density..), fill = "black", alpha = 0.3, adjust = 0.5) +
  geom_histogram(data = clinvar2, aes(x = fold_diff, y = ..count..), fill = "white", alpha = 0, adjust = 0.8, color = "grey50") +
  geom_histogram(data = clinvar4 %>% filter(fold_diff <= 0.25 & conf_low == "Yes"), aes(x = fold_diff, y = ..count..), fill = "red", alpha = 1, adjust = 0.8, color = "black") +
  #geom_histogram(data = transcripts_acmg, aes(x = fold_diff, y = ..density..), fill = "red", alpha = 0.5, color = "red") +
  NULL; Clinvar_Kozak_permutation_fold_densityplot
```

![](Kozak_files/figure-gfm/Clinvar%20variants%20of%20seemingly%20low%20score-2.png)<!-- -->

``` r
ggsave(file = "Plots/Clinvar_Kozak_permutation_fold_densityplot.pdf", Clinvar_Kozak_permutation_fold_densityplot, height = 1.4, width = 3)
```

``` r
clinvar2_acmg_most_common <- data.frame(table(clinvar2_acmg$gene)) %>% arrange(desc(Freq))
clinvar2_acmg_most_common_3plus <- clinvar2_acmg_most_common %>% filter(Freq >= 3)
clinvar2_acmg_most_common_3plus_levels <- clinvar2_acmg_most_common_3plus$Var1

clinvar2_acmg_3plus <- clinvar2_acmg %>% filter(gene %in% clinvar2_acmg_most_common_3plus$Var1)
clinvar2_acmg_3plus$gene <- factor(clinvar2_acmg_3plus$gene, levels = rev(clinvar2_acmg_most_common_3plus_levels))

clinvar2_acmg_3plus$label <- substr(clinvar2_acmg_3plus$hgvs_conseq, 3, 12)

clinvar2_acmg_3plus_conf_low <- clinvar2_acmg_3plus %>% filter(fold_diff < 0.2 & mut_kozak %in% clinvar_confidently_low_mutkozak_list)

Most_clinvar_grouped_scatterplot <- ggplot() + 
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank()) + 
  labs(x = "Fold abundance of reference Kozak", y = NULL) +
  scale_x_log10(limits = c(0.02,18)) + scale_y_discrete(expand = c(0,1)) +
  geom_vline(xintercept = 1, linetype = 2, alpha = 0.4) +
  geom_point(data = clinvar2_acmg_3plus, aes(x = fold_diff, y = gene), alpha = 0.25) +
  geom_label_repel(data = clinvar2_acmg_3plus_conf_low, 
            aes(x = fold_diff, y = gene, label = label), color = "red", size = 2, segment.color = "orange", segment.alpha = 0.4, min.segment.length = 0, label.padding = 0.2) +
  geom_point(data = clinvar2_acmg_3plus_conf_low, aes(x = fold_diff, y = gene), alpha = 0.25, color = "red") +
  NULL; Most_clinvar_grouped_scatterplot
```

![](Kozak_files/figure-gfm/clinvar%20variants%20subsetted%20to%20acmg59-1.png)<!-- -->

``` r
ggsave(file = "Plots/Most_clinvar_grouped_scatterplot.pdf", Most_clinvar_grouped_scatterplot, height = 5, width = 3.75)
```

``` r
paste("BRCA1 Reference calibrated score:",round(complete_frame4[complete_frame4$sequence == "AAAGAAATG","calibrated_score"],2))
```

    ## [1] "BRCA1 Reference calibrated score: 0.96"

``` r
paste("BRCA1 VCV000869050 calibrated score:",round(complete_frame4[complete_frame4$sequence == "AATGAAATG","calibrated_score"],2))
```

    ## [1] "BRCA1 VCV000869050 calibrated score: 0.09"

``` r
paste("BRCA1 VCV000869047 calibrated score:",round(complete_frame4[complete_frame4$sequence == "ATAGAAATG","calibrated_score"],2))
```

    ## [1] "BRCA1 VCV000869047 calibrated score: 0.12"

``` r
paste("MSH6 Reference calibrated score:",round(complete_frame4[complete_frame4$sequence == "GTCGGTATG","calibrated_score"],2))
```

    ## [1] "MSH6 Reference calibrated score: 1.15"

``` r
paste("MSH6 VCV001736915 calibrated score:",round(complete_frame4[complete_frame4$sequence == "GTCTGTATG","calibrated_score"],2))
```

    ## [1] "MSH6 VCV001736915 calibrated score: 0.03"

``` r
paste("PMS2 Reference calibrated score:",round(complete_frame4[complete_frame4$sequence == "GCATCCATG","calibrated_score"],2))
```

    ## [1] "PMS2 Reference calibrated score: 0.62"

``` r
paste("PMS2 VCV001798654 calibrated score:",round(complete_frame4[complete_frame4$sequence == "GCATGCATG","calibrated_score"],2))
```

    ## [1] "PMS2 VCV001798654 calibrated score: 0.05"

``` r
paste("RAD51C Reference calibrated score:",round(complete_frame4[complete_frame4$sequence == "CCTGCGATG","calibrated_score"],2))
```

    ## [1] "RAD51C Reference calibrated score: 1.11"

``` r
paste("RAD51C VCV000824488 calibrated score:",round(complete_frame4[complete_frame4$sequence == "CCTTCGATG","calibrated_score"],2))
```

    ## [1] "RAD51C VCV000824488 calibrated score: 0.2"

``` r
paste("RAD51C VCV000824488 calibrated score:",round(complete_frame4[complete_frame4$sequence == "CATGCGATG","calibrated_score"],2))
```

    ## [1] "RAD51C VCV000824488 calibrated score: 0.13"

``` r
paste("KCNQ1 Reference calibrated score:",round(complete_frame4[complete_frame4$sequence == "CTCGTTATG","calibrated_score"],2))
```

    ## [1] "KCNQ1 Reference calibrated score: 1.11"

``` r
paste("KCNQ1 VCV000138009 calibrated score:",round(complete_frame4[complete_frame4$sequence == "CCCGTTATG","calibrated_score"],2))
```

    ## [1] "KCNQ1 VCV000138009 calibrated score: 0.17"

``` r
paste("KCNQ1 VCV001736911 calibrated score:",round(complete_frame4[complete_frame4$sequence == "CTCCTTATG","calibrated_score"],2))
```

    ## [1] "KCNQ1 VCV001736911 calibrated score: 0.09"

``` r
paste("KCNH2 Reference calibrated score:",round(complete_frame4[complete_frame4$sequence == "CTCAGGATG","calibrated_score"],2))
```

    ## [1] "KCNH2 Reference calibrated score: 1.13"

``` r
paste("KCNH2 VCV003287580 calibrated score:",round(complete_frame4[complete_frame4$sequence == "CTCTGGATG","calibrated_score"],2))
```

    ## [1] "KCNH2 VCV003287580 calibrated score: 0.02"

``` r
paste("VHL Reference calibrated score:",round(complete_frame4[complete_frame4$sequence == "GAGGGAATG","calibrated_score"],2))
```

    ## [1] "VHL Reference calibrated score: 0.62"

``` r
paste("VHL VCV003332039 calibrated score:",round(complete_frame4[complete_frame4$sequence == "GAGTGAATG","calibrated_score"],2))
```

    ## [1] "VHL VCV003332039 calibrated score: 0.04"

``` r
## Below are manually confirming potentially problematic VUSes
clinvar2_acmg$comment <- NA
clinvar2_acmg[clinvar2_acmg$gene == "MSH6" & clinvar2_acmg$hgvs_conseq == "c.-3G>T" & clinvar2_acmg$kozak == "GTCGGTATG","comment"] <- "VUS that is likely pathogenic"
clinvar2_acmg[clinvar2_acmg$gene == "RAD51C" & clinvar2_acmg$hgvs_conseq == "c.-3G>T" & clinvar2_acmg$kozak == "CCTGCGATG","comment"] <- "VUS that is likely pathogenic"
clinvar2_acmg[clinvar2_acmg$gene == "RAD51C" & clinvar2_acmg$hgvs_conseq == "c.-3G>C" & clinvar2_acmg$kozak == "CCTGCGATG","comment"] <- "VUS that is likely pathogenic"
clinvar2_acmg[clinvar2_acmg$gene == "KCNQ1" & clinvar2_acmg$hgvs_conseq == "c.-3G>C" & clinvar2_acmg$kozak == "CTCGTTATG","comment"] <- "VUS that is likely pathogenic"
clinvar2_acmg[clinvar2_acmg$gene == "SDHD" & clinvar2_acmg$hgvs_conseq == "c.-4C>T" & clinvar2_acmg$kozak == "AACGAGATG","comment"] <- "VUS that is likely pathogenic"
clinvar2_acmg[clinvar2_acmg$gene == "MUTYH" & clinvar2_acmg$hgvs_conseq == "c.-3A>C" & clinvar2_acmg$kozak == "GCCATCATG","comment"] <- "VUS that is likely pathogenic"
clinvar2_acmg[clinvar2_acmg$gene == "SDHA" & clinvar2_acmg$hgvs_conseq == "c.-3G>T" & clinvar2_acmg$kozak == "GCAGACATG","comment"] <- "VUS that is likely pathogenic"
clinvar2_acmg[clinvar2_acmg$gene == "PMS2" & clinvar2_acmg$hgvs_conseq == "c.-2C>G" & clinvar2_acmg$kozak == "GCATCCATG","comment"] <- "VUS that is likely pathogenic"
clinvar2_acmg[clinvar2_acmg$gene == "PALB2" & clinvar2_acmg$hgvs_conseq == "c.-2C>G" & clinvar2_acmg$kozak == "TGCCCGATG","comment"] <- "VUS that is likely pathogenic"
```

## Bringing in the infection data

``` r
individual_infection_summary2 <- individual_infection_summary
individual_infection_summary2[individual_infection_summary2$cell_label == "G790A", "cell_label"] <- "Consensus"
individual_infection_summary2[!(individual_infection_summary2$cell_label %in% c("Consensus","None")), "cell_label"] <- NA
individual_infection_summary2[individual_infection_summary2$virus == "VSVG", "virus"] <- "VSV"
individual_infection_summary2[individual_infection_summary2$virus == "SARS1", "virus"] <- "SARS-CoV"
individual_infection_summary2[individual_infection_summary2$virus == "G928A SARS2", "virus"] <- "SARS-CoV-2"
individual_infection_summary2[individual_infection_summary2$virus == "G1074A SARS2", "virus"] <- "SARS-CoV-2 (Furin mut)"

Individual_VSV_infection_plot <- ggplot() + theme_bw() + 
  labs(x = "Fraction of WT (ACE2-2A-)iRFP670 MFI", y = "Fraction of WT infection", title = "VSV") +
  scale_x_log10(limits = c(3e-3,2), expand = c(0,0)) + scale_y_log10(limits = c(3e-3,2), expand = c(0,0)) + 
  geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.4) +
  geom_point(data = individual_infection_summary2 %>% filter(virus == "VSV"), aes(x = geomeanfraction_wt_mfi, y = geomean_fraction_wt_infection), alpha = 0.5) +
  #geom_text_repel(data = individual_infection_summary2 %>% filter(virus == "VSV"), aes(x = geomeanfraction_wt_mfi, y = geomean_fraction_wt_infection, label = cell_label), color = "red", size = 2, segment.alpha = 0.4) +
  NULL
Individual_VSV_infection_plot
```

![](Kozak_files/figure-gfm/Plotting%20the%20individual%20infection%20data-1.png)<!-- -->

``` r
Individual_SARS_CoV_infection_plot <- ggplot() + theme_bw() + 
  labs(x = "Fraction of WT (ACE2-2A-)iRFP670 MFI", y = "Fraction of WT infection", title = "SARS-CoV") +
  scale_x_log10(limits = c(3e-3,2), expand = c(0,0)) + scale_y_log10(limits = c(3e-3,2), expand = c(0,0)) + 
  geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.4) +
  geom_point(data = individual_infection_summary2 %>% filter(virus == "SARS-CoV"), aes(x = geomeanfraction_wt_mfi, y = geomean_fraction_wt_infection), alpha = 0.5) +
  #geom_text_repel(data = individual_infection_summary2 %>% filter(virus == "SARS-CoV"), aes(x = geomeanfraction_wt_mfi, y = geomean_fraction_wt_infection, label = cell_label), color = "red", size = 2, segment.alpha = 0.4) +
  NULL
Individual_SARS_CoV_infection_plot
```

![](Kozak_files/figure-gfm/Plotting%20the%20individual%20infection%20data-2.png)<!-- -->

``` r
Individual_SARS_CoV_2_infection_plot <- ggplot() + theme_bw() + 
  labs(x = "Fraction of WT (ACE2-2A-)iRFP670 MFI", y = "Fraction of WT infection", title = "SARS-CoV-2") +
  scale_x_log10(limits = c(3e-3,2), expand = c(0,0)) + scale_y_log10(limits = c(3e-3,2), expand = c(0,0)) + 
  geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.4) +
  geom_point(data = individual_infection_summary2 %>% filter(virus == "SARS-CoV-2"), aes(x = geomeanfraction_wt_mfi, y = geomean_fraction_wt_infection), alpha = 0.5) +
  #geom_text_repel(data = individual_infection_summary2 %>% filter(virus == "SARS-CoV-2"), aes(x = geomeanfraction_wt_mfi, y = geomean_fraction_wt_infection, label = cell_label), color = "red", size = 2, segment.alpha = 0.4) +
  NULL
Individual_SARS_CoV_2_infection_plot
```

![](Kozak_files/figure-gfm/Plotting%20the%20individual%20infection%20data-3.png)<!-- -->

``` r
Individual_SARS_CoV_2_Furin_mut_infection_plot <- ggplot() + theme_bw() + 
  labs(x = "Fraction of WT (ACE2-2A-)iRFP670 MFI", y = "Fraction of WT infection", title = "SARS-CoV-2 (Furin mut)") +
  scale_x_log10(limits = c(3e-3,2), expand = c(0,0)) + scale_y_log10(limits = c(3e-3,2), expand = c(0,0)) + 
  geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.4) +
  geom_point(data = individual_infection_summary2 %>% filter(virus == "SARS-CoV-2 (Furin mut)"), aes(x = geomeanfraction_wt_mfi, y = geomean_fraction_wt_infection), alpha = 0.5) +
  #geom_text_repel(data = individual_infection_summary2 %>% filter(virus == "SARS-CoV-2 (Furin mut)"), aes(x = geomeanfraction_wt_mfi, y = geomean_fraction_wt_infection, label = cell_label), color = "red", size = 2, segment.alpha = 0.4) +
  NULL
Individual_SARS_CoV_2_Furin_mut_infection_plot
```

![](Kozak_files/figure-gfm/Plotting%20the%20individual%20infection%20data-4.png)<!-- -->

``` r
Combined_individual <- (Individual_VSV_infection_plot | Individual_SARS_CoV_infection_plot) / (Individual_SARS_CoV_2_infection_plot | Individual_SARS_CoV_2_Furin_mut_infection_plot)

Combined_individual
```

![](Kozak_files/figure-gfm/Plotting%20the%20individual%20infection%20data-5.png)<!-- -->

``` r
ggsave(file = "plots/Combined_individual.pdf", Combined_individual, height = 4, width = 4.25)
```

``` r
s1_experiment_labels <- c("r2e1_10","r2e1_16","r2e2_25","r3e1_8","r3e1_32","r4e1_8","r4e1_32")
s1_experiment_read_labels <- c("r2e1_10_reads","r2e1_16_reads","r2e2_25_reads","r3e1_8_reads","r3e1_32_reads","r4e1_8_reads","r4e1_32_reads")

r2e1_10 <- makeExperimentFrame2(c(1,2,5,6)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r2e1_16 <- makeExperimentFrame2(c(7,8,11,12)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r2e2_25 <- makeExperimentFrame2(c(13,14,17,18)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r3e1_8 <- makeExperimentFrame2(c(124,124,128,128)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r3e1_32 <- makeExperimentFrame2(c(125,125,129,129)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r4e1_8 <- makeExperimentFrame2(c(126,126,130,130)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r4e1_32 <- makeExperimentFrame2(c(127,127,131,131)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))

s1_enrichment <- merge(r2e1_10[,c("sequence","h_enrichment","total_reads")], r2e1_16[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s1_enrichment <- merge(s1_enrichment, r2e2_25[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s1_enrichment <- merge(s1_enrichment, r3e1_8[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s1_enrichment <- merge(s1_enrichment, r3e1_32[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s1_enrichment <- merge(s1_enrichment, r4e1_8[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s1_enrichment <- merge(s1_enrichment, r4e1_32[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
colnames(s1_enrichment) <- c("sequence",c(rbind(s1_experiment_labels, s1_experiment_read_labels))) ## https://coolbutuseless.github.io/2019/01/29/interleaving-vectors-and-matrices-part-1/

s1_enrichment$enrichment_reads <- rowSums(s1_enrichment[,s1_experiment_read_labels], na.rm = T)

for(x in s1_experiment_labels){
  s1_enrichment[,x] <- s1_enrichment[,x] / s1_enrichment[s1_enrichment$sequence == "GCCACCATG",x]
}

## Getting summary stats for the SARS1 enrichment experiments
melt_s1_enrichment <- melt(s1_enrichment[,c("sequence",s1_experiment_labels)], id = "sequence") %>% mutate(n = NA)
melt_s1_enrichment[!is.na(melt_s1_enrichment$value),"n"] = 1

s1_enrichment_summary <- melt_s1_enrichment %>% group_by(sequence) %>% summarize(mean_log10 = mean(log10(value), na.rm = T), sd_log10 = sd(log10(value), na.rm = T), n = sum(n, na.rm = T), .groups = "drop") %>% arrange(desc(mean_log10)) %>% mutate(geomean = 10^mean_log10, upper_conf = 10^(mean_log10 + sd_log10/sqrt(n-1)*1.96), lower_conf = 10^(mean_log10 - sd_log10/sqrt(n-1)*1.96))

s1_enrichment_summary2 <- merge(s1_enrichment_summary, s1_enrichment[,c("sequence","enrichment_reads")], by= "sequence") %>% arrange(desc(enrichment_reads)) %>% filter(n >= 2)
```

``` r
s2_experiment_labels <- c("r1e1_8","r1e1_32","r2e1_8","r2e1_32","r1e2_8","r1e2_32","r2e2_8","r2e2_32")
s2_experiment_read_labels <- c("r1e1_8_reads","r1e1_32_reads","r2e1_8_reads","r2e1_32_reads","r1e2_8_reads","r1e2_32_reads","r2e2_8_reads","r2e2_32_reads")

r1e1_8 <- makeExperimentFrame2(c(108,108,116,116)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r1e1_32 <- makeExperimentFrame2(c(109,109,117,117)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r2e1_8 <- makeExperimentFrame2(c(110,110,118,118)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r2e1_32 <- makeExperimentFrame2(c(111,111,119,119)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r1e2_8 <- makeExperimentFrame2(c(112,112,120,120)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r1e2_32 <- makeExperimentFrame2(c(113,113,121,121)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r2e2_8 <- makeExperimentFrame2(c(114,114,122,122)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r2e2_32 <- makeExperimentFrame2(c(115,115,123,123)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))

s2_enrichment <- merge(r1e1_8[,c("sequence","h_enrichment","total_reads")], r1e1_32[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s2_enrichment <- merge(s2_enrichment, r2e1_8[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s2_enrichment <- merge(s2_enrichment, r2e1_32[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s2_enrichment <- merge(s2_enrichment, r1e2_8[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s2_enrichment <- merge(s2_enrichment, r1e2_32[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s2_enrichment <- merge(s2_enrichment, r2e2_8[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
s2_enrichment <- merge(s2_enrichment, r2e2_32[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
colnames(s2_enrichment) <- c("sequence",c(rbind(s2_experiment_labels, s2_experiment_read_labels))) ## https://coolbutuseless.github.io/2019/01/29/interleaving-vectors-and-matrices-part-1/

s2_enrichment$enrichment_reads <- rowSums(s2_enrichment[,s2_experiment_read_labels], na.rm = T)

for(x in s2_experiment_labels){
  s2_enrichment[,x] <- s2_enrichment[,x] / s2_enrichment[s2_enrichment$sequence == "GCCACCATG",x]
}


 ## Getting summary stats for the SARS1 enrichment experiments
melt_s2_enrichment <- melt(s2_enrichment[,c("sequence",s2_experiment_labels)], id = "sequence")
melt_s2_enrichment <- subset(melt_s2_enrichment, !is.na(value)) %>% mutate(n = 1)

s2_enrichment_summary <- melt_s2_enrichment %>% group_by(sequence) %>% summarize(mean_log10 = mean(log10(value), na.rm = T), sd_log10 = sd(log10(value), na.rm = T), n = sum(n, na.rm = T), .groups = "drop") %>% arrange(desc(mean_log10)) %>% mutate(geomean = 10^mean_log10, upper_conf = 10^(mean_log10 + sd_log10/sqrt(n-1)*1.96), lower_conf = 10^(mean_log10 - sd_log10/sqrt(n-1)*1.96))

s2_enrichment_summary2 <- merge(s2_enrichment_summary, s2_enrichment[,c("sequence","enrichment_reads")], by= "sequence") %>% arrange(desc(enrichment_reads)) %>% filter(n >= 2)
```

``` r
vsv_experiment_labels <- c("r3e1_2","r3e1_8","r4e1_2","r4e1_8")
vsv_experiment_read_labels <- c("r3e1_2_reads","r3e1_8_reads","r4e1_2_reads","r4e1_8_reads")

r3e1_2 <- makeExperimentFrame2(c(32,32,36,36)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r3e1_8 <- makeExperimentFrame2(c(33,148,137,137)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r4e1_2 <- makeExperimentFrame2(c(134,134,138,138)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
r4e1_8 <- makeExperimentFrame2(c(135,135,139,149)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))

vsv_enrichment <- merge(r3e1_2[,c("sequence","h_enrichment","total_reads")], r3e1_8[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
vsv_enrichment <- merge(vsv_enrichment, r4e1_8[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
vsv_enrichment <- merge(vsv_enrichment, r4e1_2[,c("sequence","h_enrichment","total_reads")], by = "sequence", all = T)
colnames(vsv_enrichment) <- c("sequence",c(rbind(vsv_experiment_labels, vsv_experiment_read_labels)))

vsv_enrichment$enrichment_reads <- rowSums(vsv_enrichment[,c("r3e1_2_reads","r3e1_8_reads","r4e1_2_reads","r4e1_8_reads")], na.rm = T)

for(x in vsv_experiment_labels){
  vsv_enrichment[,x] <- vsv_enrichment[,x] / vsv_enrichment[vsv_enrichment$sequence == "GCCACCATG",x]
}

## Getting summary stats for the SARvsv enrichment experiments
melt_vsv_enrichment <- melt(vsv_enrichment[,c("sequence","r3e1_2","r3e1_8","r4e1_2","r4e1_8")], id = "sequence") %>% mutate(n = NA)
melt_vsv_enrichment[!is.na(melt_vsv_enrichment$value),"n"] = 1

vsv_enrichment_summary <- melt_vsv_enrichment %>% group_by(sequence) %>% summarize(mean_log10 = mean(log10(value), na.rm = T), sd_log10 = sd(log10(value), na.rm = T), n = sum(n, na.rm = T), .groups = "drop") %>% arrange(desc(mean_log10)) %>% mutate(geomean = 10^mean_log10, upper_conf = 10^(mean_log10 + sd_log10/sqrt(n-1)*1.96), lower_conf = 10^(mean_log10 - sd_log10/sqrt(n-1)*1.96))

vsv_enrichment_summary2 <- merge(vsv_enrichment_summary, vsv_enrichment[,c("sequence","enrichment_reads")], by= "sequence") %>% arrange(desc(enrichment_reads)) %>% filter(n >= 2)
```

``` r
complete_frame5 <- merge(complete_frame4, s1_enrichment_summary2[,c("sequence","geomean")], by = "sequence", all = T)
complete_frame5 <- merge(complete_frame5, s2_enrichment_summary2[,c("sequence","geomean")], by = "sequence", all.x = T)
complete_frame5 <- merge(complete_frame5, vsv_enrichment_summary2[,c("sequence","geomean")], by = "sequence", all.x = T)
colnames(complete_frame5) <- c("sequence","sort_geomean_log10","sort_geomean","sort_upper_conf","sort_lower_conf","mfi_lm_norm","calibrated_score","mfi_individual","noderer","rf_predict_mfi","residuals","s1","s2","vsv")
```

``` r
Sort_vs_SARS2_scatterplot <- ggplot() + scale_y_log10(limits = c(0.003,10), expand = c(0,0)) +
  labs(x = "Our 4-way sort score", y = "SARS2 infection") +
  geom_point(data= complete_frame5, aes(x = sort_geomean_log10, y = s2), alpha = 0.4)
#ggsave(file = "plots/Sort_vs_SARS2_scatterplot.png", Sort_vs_SARS2_scatterplot, height = 4, width = 6)
Sort_vs_SARS2_scatterplot
```

![](Kozak_files/figure-gfm/Looking%20at%20correlations%20between%20abundance%20and%20infection%20for%20SARS2-1.png)<!-- -->

``` r
Noderer_vs_SARS2_scatterplot <- ggplot() + scale_y_log10(limits = c(0.003,10), expand = c(0,0)) + 
  labs(x = "Noderer score", y = "SARS2 infection") +
  geom_point(data= complete_frame5, aes(x = noderer, y = s2), alpha = 0.4)
#ggsave(file = "plots/Noderer_vs_SARS2_scatterplot.png", Noderer_vs_SARS2_scatterplot, height = 4, width = 6)
Noderer_vs_SARS2_scatterplot
```

![](Kozak_files/figure-gfm/Looking%20at%20correlations%20between%20abundance%20and%20infection%20for%20SARS2-2.png)<!-- -->

``` r
calibrated_score_vs_SARS2_scatterplot <- ggplot() + scale_x_log10(limits = c(0.01,1.4), expand = c(0,0)) + scale_y_log10(limits = c(0.003,10), expand = c(0,0)) + 
  labs(x = "calibrated_score score", y = "SARS2 infection") +
  geom_point(data= complete_frame5, aes(x = calibrated_score, y = s2), alpha = 0.4) +
  NULL; calibrated_score_vs_SARS2_scatterplot
```

![](Kozak_files/figure-gfm/Looking%20at%20correlations%20between%20abundance%20and%20infection%20for%20SARS2-3.png)<!-- -->

``` r
ggsave(file = "plots/calibrated_score_vs_SARS2_scatterplot.png", calibrated_score_vs_SARS2_scatterplot, height = 4, width = 6)
```

``` r
individual_s2 <- individual_infection_summary %>% filter(virus %in% c("G1074A SARS2","G928A SARS2")) %>% group_by(sequence) %>% summarize(individual_s2 = 10^mean(log10(geomean_fraction_wt_infection)))
individual_s2$scaled_ind_s2 <- (individual_s2$individual_s2 - as.numeric(individual_s2[individual_s2$sequence == "None", "individual_s2"])) / 
(as.numeric(individual_s2[individual_s2$sequence == "GCCACCATG", "individual_s2"]) - as.numeric(individual_s2[individual_s2$sequence == "None", "individual_s2"]))

individual_s1 <- individual_infection_summary %>% filter(virus %in% c("SARS1")) %>% group_by(sequence) %>% mutate(individual_s1 = geomean_fraction_wt_infection)
individual_s1$scaled_ind_s1 <- (individual_s1$individual_s1 - as.numeric(individual_s1[individual_s1$sequence == "None", "individual_s1"])) / 
(as.numeric(individual_s1[individual_s1$sequence == "GCCACCATG", "individual_s1"]) - as.numeric(individual_s1[individual_s1$sequence == "None", "individual_s1"]))

individual_vsv <- individual_infection_summary %>% filter(virus %in% c("VSVG")) %>% group_by(sequence) %>% mutate(individual_vsv = geomean_fraction_wt_infection)
individual_vsv$scaled_ind_vsv <- (individual_vsv$individual_vsv - as.numeric(individual_vsv[individual_vsv$sequence == "None", "individual_vsv"])) / 
(as.numeric(individual_vsv[individual_vsv$sequence == "GCCACCATG", "individual_vsv"]) - as.numeric(individual_vsv[individual_vsv$sequence == "None", "individual_vsv"]))

complete_frame6 <- merge(complete_frame5, individual_s2[,c("sequence","individual_s2","scaled_ind_s2")], by = "sequence", all.x = T)
complete_frame6 <- merge(complete_frame6, individual_s1[,c("sequence","individual_s1","scaled_ind_s1")], by = "sequence", all.x = T)
complete_frame6 <- merge(complete_frame6, individual_vsv[,c("sequence","individual_vsv","scaled_ind_vsv")], by = "sequence", all.x = T)

complete_frame6$scaled_s2 <- (complete_frame6$s2 - as.numeric(complete_frame6[complete_frame6$sequence == "XXXXXXXXX", "s2"])) / 
(as.numeric(complete_frame6[complete_frame6$sequence == "GCCACCATG", "s2"]) - as.numeric(complete_frame6[complete_frame6$sequence == "XXXXXXXXX", "s2"]))

complete_frame6$scaled_s1 <- (complete_frame6$s1 - as.numeric(complete_frame6[complete_frame6$sequence == "XXXXXXXXX", "s1"])) / 
(as.numeric(complete_frame6[complete_frame6$sequence == "GCCACCATG", "s1"]) - as.numeric(complete_frame6[complete_frame6$sequence == "XXXXXXXXX", "s1"]))

complete_frame6$scaled_vsv <- (complete_frame6$vsv - as.numeric(complete_frame6[complete_frame6$sequence == "XXXXXXXXX", "vsv"])) / 
(as.numeric(complete_frame6[complete_frame6$sequence == "GCCACCATG", "vsv"]) - as.numeric(complete_frame6[complete_frame6$sequence == "XXXXXXXXX", "vsv"]))

Scaled_individual_multiplex_s2_plot <- ggplot() + theme(panel.grid = element_blank()) + 
  scale_x_log10(limits = c(0.01,1.1), expand = c(0,0)) + scale_y_log10(limits = c(0.003,1.1)) +  #scale_y_continuous(limits = c(0,1)) +#scale_y_log10(limits = c(0.008,10)) + 
  labs(x = "Abundance", y = "Enrichment in infected cells", title = "SARS-CoV-2 spike") +
  stat_smooth(data = complete_frame6 %>% filter(!is.na(scaled_ind_s2) & !is.na(scaled_s2)), aes(x = mfi_individual, y = scaled_ind_s2), geom='line', se=FALSE, alpha = 1, shape = 20, size = 2, color = "cyan", alpha = 0.2) +
  stat_smooth(data = complete_frame6 %>% filter(!is.na(scaled_ind_s2) & !is.na(scaled_s2)), aes(x = calibrated_score, y = scaled_s2), geom='line', se=FALSE, alpha = 1, shape = 20, size = 2, color = "red", line.alpha = 0.2) +
  geom_point(data = complete_frame6 %>% filter(!is.na(scaled_ind_s2) & !is.na(scaled_s2)), aes(x = mfi_individual, y = scaled_ind_s2), alpha = 1, shape = 20, size = 2, color = "blue") +
  geom_point(data = complete_frame6 %>% filter(!is.na(scaled_ind_s2) & !is.na(scaled_s2)), aes(x = calibrated_score, y = scaled_s2), alpha = 1, shape = 20, size = 2, color = "darkred") +
  NULL; Scaled_individual_multiplex_s2_plot
```

![](Kozak_files/figure-gfm/Comparisons%20with%20individual%20infections-1.png)<!-- -->

``` r
ggsave(file = "plots/Scaled_individual_multiplex_s2_plot.pdf", Scaled_individual_multiplex_s2_plot, height = 1.5, width = 1.5)

Scaled_individual_multiplex_s1_plot <- ggplot() + theme(panel.grid = element_blank()) + 
  scale_x_log10(limits = c(0.01,1.1), expand = c(0,0)) + scale_y_log10(limits = c(0.003,1.1)) + #scale_y_continuous(limits = c(0,1)) + #scale_y_log10(limits = c(0.008,10)) + 
  labs(x = "Abundance", y = "Enrichment in infected cells", title = "SARS-CoV spike") +
  stat_smooth(data = complete_frame6 %>% filter(!is.na(scaled_ind_s1) & !is.na(scaled_s1)), aes(x = mfi_individual, y = scaled_ind_s1), geom='line', se=FALSE, alpha = 1, shape = 20, size = 2, color = "cyan", alpha = 0.2) +
  stat_smooth(data = complete_frame6 %>% filter(!is.na(scaled_ind_s1) & !is.na(scaled_s1)), aes(x = calibrated_score, y = scaled_s1), geom='line', se=FALSE, alpha = 1, shape = 20, size = 2, color = "red", line.alpha = 0.2) +
  geom_point(data = complete_frame6 %>% filter(!is.na(scaled_ind_s1) & !is.na(scaled_s1)), aes(x = mfi_individual, y = scaled_ind_s1), alpha = 1, shape = 20, size = 2, color = "blue") +
  geom_point(data = complete_frame6 %>% filter(!is.na(scaled_ind_s1) & !is.na(scaled_s1)), aes(x = calibrated_score, y = scaled_s1), alpha = 1, shape = 20, size = 2, color = "darkred") +
  NULL; Scaled_individual_multiplex_s1_plot
```

![](Kozak_files/figure-gfm/Comparisons%20with%20individual%20infections-2.png)<!-- -->

``` r
ggsave(file = "plots/Scaled_individual_multiplex_s1_plot.pdf", Scaled_individual_multiplex_s1_plot, height = 1.5, width = 1.5)

Scaled_individual_multiplex_vsv_plot <- ggplot() + theme(panel.grid = element_blank()) + 
  scale_x_log10(limits = c(0.01,2), expand = c(0,0)) + scale_y_log10(limits = c(0.003,5)) + ##scale_y_log10(limits = c(0.008,10)) + 
  labs(x = "Abundance", y = "Enrichment in infected cells", title = "VSV-G") +
  stat_smooth(data = complete_frame6 %>% filter(!is.na(scaled_ind_vsv) & !is.na(scaled_vsv)), aes(x = mfi_individual, y = scaled_ind_vsv), geom='line', se=FALSE, alpha = 1, shape = 20, size = 2, color = "cyan", alpha = 0.2) +
  stat_smooth(data = complete_frame6 %>% filter(!is.na(scaled_ind_vsv) & !is.na(scaled_vsv)), aes(x = calibrated_score, y = scaled_vsv), geom='line', se=FALSE, alpha = 1, shape = 20, size = 2, color = "red", line.alpha = 0.2) +
  geom_point(data = complete_frame6 %>% filter(!is.na(scaled_ind_vsv) & !is.na(scaled_vsv)), aes(x = mfi_individual, y = scaled_ind_vsv), alpha = 1, shape = 20, size = 2, color = "blue") +
  geom_point(data = complete_frame6 %>% filter(!is.na(scaled_ind_vsv) & !is.na(scaled_vsv)), aes(x = calibrated_score, y = scaled_vsv), alpha = 1, shape = 20, size = 2, color = "darkred") +
  NULL; Scaled_individual_multiplex_vsv_plot
```

![](Kozak_files/figure-gfm/Comparisons%20with%20individual%20infections-3.png)<!-- -->

``` r
ggsave(file = "plots/Scaled_individual_multiplex_vsv_plot.pdf", Scaled_individual_multiplex_vsv_plot, height = 1.5, width = 1.5)
```

``` r
write.csv(file = "Output_datatables/Supp_table_1_kozak_sortseq_and_infection.csv", complete_frame6[,c("sequence","sort_geomean","sort_upper_conf","sort_lower_conf","mfi_lm_norm","calibrated_score","mfi_individual","noderer","rf_predict_mfi","s1","s2","vsv","individual_s1","individual_s2","individual_vsv")], row.names = F, quote = F)
```

## Calculate derivatives of the slope

``` r
# https://stackoverflow.com/questions/75535848/extracting-values-from-a-geom-smoothmethod-loess-but-some-issues

loess_span <- 0.5
newvals <- data.frame(calibrated_score=seq(0.01,1.2,.01))

complete_frame5_vsvsort <- complete_frame5 %>% filter(!is.na(vsv) & vsv != 0) %>% arrange(calibrated_score)
vsvloess <- loess(log10(vsv)~calibrated_score, data=complete_frame5_vsvsort, span = loess_span)
complete_frame5_vsvloess <- cbind(newvals, value=predict(vsvloess, newvals))

complete_frame5_s1sort <- complete_frame5 %>% filter(!is.na(s1) & s1 != 0) %>% arrange(calibrated_score)
s1loess <- loess(log10(s1)~calibrated_score, data=complete_frame5_s1sort, span = loess_span)
complete_frame5_s1loess <- cbind(newvals, value=predict(s1loess, newvals))

complete_frame5_s2sort <- complete_frame5 %>% filter(!is.na(s2) & s2 != 0) %>% arrange(calibrated_score)
s2loess <- loess(log10(s2)~calibrated_score, data=complete_frame5_s2sort, span = loess_span)
complete_frame5_s2loess <- cbind(newvals, value=predict(s2loess, newvals))

ggplot() + 
  scale_x_log10() + scale_y_log10() + 
  geom_point(data = complete_frame5, aes(x = calibrated_score, y = vsv), alpha = 0.4) +
  geom_line(data = complete_frame5_vsvloess, aes(x = calibrated_score, y = value), color = "red") +
  NULL
```

![](Kozak_files/figure-gfm/Calculate%20derivatives%20of%20the%20slope-1.png)<!-- -->

``` r
calibrated_vsv_loess_plot <- ggplot() + theme(panel.grid = element_blank()) + 
  scale_x_log10(limits = c(0.01,2), expand = c(0,0)) + scale_y_log10(limits = c(0.008,10)) + 
  labs(x = "calibrated abundance", y = "Enrichment in infected cells", title = "VSV-G") +
  #geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.4, color = "black") +
  geom_hline(yintercept = complete_frame5[complete_frame5$sequence == "XXXXXXXXX","vsv"], linetype = 1, alpha = 0.4, color = "purple", size = 2) + 
  geom_point(data= complete_frame5, aes(x = calibrated_score, y = vsv), alpha = 0.04, shape = 20, size = 2) +
  geom_line(data = complete_frame5_vsvloess, aes(x = calibrated_score, y = 10^value), color = "red", alpha = 0.3, size = 3) +
  #stat_smooth(data= complete_frame5, aes(x = calibrated_score, y = vsv), geom='line', alpha=0.1, se=FALSE, size = 3) +
  #geom_point(data= complete_frame6, aes(x = mfi_individual, y = individual_vsv), alpha = 0.4, color = "red", shape = 18, size = 2) +
  NULL; calibrated_vsv_loess_plot
```

![](Kozak_files/figure-gfm/Calculate%20derivatives%20of%20the%20slope-2.png)<!-- -->

``` r
ggsave(file = "plots/calibrated_vsv_loess_plot.pdf", calibrated_vsv_loess_plot, height = 1.5, width = 2)

ggplot() + 
  scale_x_log10() + scale_y_log10() + 
  geom_point(data = complete_frame5, aes(x = calibrated_score, y = s1), alpha = 0.4) +
  geom_line(data = complete_frame5_s1loess, aes(x = calibrated_score, y = value), color = "red") +
  NULL
```

![](Kozak_files/figure-gfm/Calculate%20derivatives%20of%20the%20slope-3.png)<!-- -->

``` r
calibrated_s1_loess_plot <- ggplot() + theme(panel.grid = element_blank()) + 
  scale_x_log10(limits = c(0.01,2), expand = c(0,0)) + scale_y_log10(limits = c(0.008,10)) + 
  labs(x = "calibrated abundance", y = "Enrichment in infected cells", title = "SARS-CoV") +
  #geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.4, color = "black") +
  geom_hline(yintercept = complete_frame5[complete_frame5$sequence == "XXXXXXXXX","s1"], linetype = 1, alpha = 0.4, color = "purple", size = 2) + 
  geom_point(data= complete_frame5 %>% filter(!is.na(s1) & s1 != 0), aes(x = calibrated_score, y = s1), alpha = 0.04, shape = 20, size = 2) +
  geom_line(data = complete_frame5_s1loess, aes(x = calibrated_score, y = 10^value), color = "red", alpha = 0.3, size = 3) +
  #stat_smooth(data= complete_frame5, aes(x = calibrated_score, y = vsv), geom='line', alpha=0.1, se=FALSE, size = 3) +
  #geom_point(data= complete_frame6, aes(x = mfi_individual, y = individual_vsv), alpha = 0.4, color = "red", shape = 18, size = 2) +
  NULL; calibrated_s1_loess_plot
```

![](Kozak_files/figure-gfm/Calculate%20derivatives%20of%20the%20slope-4.png)<!-- -->

``` r
ggsave(file = "plots/calibrated_s1_loess_plot.pdf", calibrated_s1_loess_plot, height = 1.5, width = 2)

ggplot() + 
  scale_x_log10() + scale_y_log10() + 
  geom_point(data = complete_frame5, aes(x = calibrated_score, y = s2), alpha = 0.4) +
  geom_line(data = complete_frame5_s2loess, aes(x = calibrated_score, y = value), color = "red") +
  NULL
```

![](Kozak_files/figure-gfm/Calculate%20derivatives%20of%20the%20slope-5.png)<!-- -->

``` r
calibrated_s2_loess_plot <- ggplot() + theme(panel.grid = element_blank()) + 
  scale_x_log10(limits = c(0.01,2), expand = c(0,0)) + scale_y_log10(limits = c(0.008,10)) + 
  labs(x = "calibrated abundance", y = "Enrichment in infected cells", title = "SARS-CoV-2") +
  #geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.4, color = "black") +
  geom_hline(yintercept = complete_frame5[complete_frame5$sequence == "XXXXXXXXX","s2"], linetype = 1, alpha = 0.4, color = "purple", size = 2) + 
  geom_point(data= complete_frame5 %>% filter(!is.na(s2) & s2 != 0), aes(x = calibrated_score, y = s2), alpha = 0.04, shape = 20, size = 2) +
  geom_line(data = complete_frame5_s2loess, aes(x = calibrated_score, y = 10^value), color = "red", alpha = 0.3, size = 3) +
  #stat_smooth(data= complete_frame5, aes(x = calibrated_score, y = vsv), geom='line', alpha=0.1, se=FALSE, size = 3) +
  #geom_point(data= complete_frame6, aes(x = mfi_individual, y = individual_vsv), alpha = 0.4, color = "red", shape = 18, size = 2) +
  NULL; calibrated_s2_loess_plot
```

![](Kozak_files/figure-gfm/Calculate%20derivatives%20of%20the%20slope-6.png)<!-- -->

``` r
ggsave(file = "plots/calibrated_s2_loess_plot.pdf", calibrated_s2_loess_plot, height = 1.5, width = 2)

ggplot() + 
  scale_x_log10() + #scale_y_log10() + 
  geom_line(data = complete_frame5_vsvloess, aes(x = calibrated_score, y = value)) +
  geom_line(data = complete_frame5_s1loess, aes(x = calibrated_score, y = value), color = "red") +
  geom_line(data = complete_frame5_s2loess, aes(x = calibrated_score, y = value), color = "green") +
  NULL
```

![](Kozak_files/figure-gfm/Calculate%20derivatives%20of%20the%20slope-7.png)<!-- -->

``` r
calibrated_combined_loess_plot <- ggplot() + theme(panel.grid = element_blank()) + 
  scale_x_log10(limits = c(0.01,2), expand = c(0,0)) + scale_y_log10(limits = c(0.008,10)) + 
  labs(x = "calibrated abundance", y = "Enrichment in infected cells", title = "") +
  #geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.4, color = "black") +
  geom_line(data = complete_frame5_vsvloess, aes(x = calibrated_score, y = 10^value), size = 2, alpha = 0.4) +
  geom_line(data = complete_frame5_s1loess, aes(x = calibrated_score, y = 10^value), color = "red", size = 2, alpha = 0.4) +
  geom_line(data = complete_frame5_s2loess, aes(x = calibrated_score, y = 10^value), color = "green", size = 2, alpha = 0.4) +  #stat_smooth(data= complete_frame5, aes(x = calibrated_score, y = vsv), geom='line', alpha=0.1, se=FALSE, size = 3) +
  #geom_point(data= complete_frame6, aes(x = mfi_individual, y = individual_vsv), alpha = 0.4, color = "red", shape = 18, size = 2) +
  NULL; calibrated_combined_loess_plot
```

![](Kozak_files/figure-gfm/Calculate%20derivatives%20of%20the%20slope-8.png)<!-- -->

``` r
ggsave(file = "plots/calibrated_combined_loess_plot.pdf", calibrated_combined_loess_plot, height = 1.5, width = 2)
  
complete_frame5_vsvloess$derivative <- NA
complete_frame5_s1loess$derivative <- NA
complete_frame5_s2loess$derivative <- NA
for(x in 2:nrow(complete_frame5_vsvloess)){
  complete_frame5_vsvloess$derivative[x] <- ((log10(complete_frame5_vsvloess$value[x]) - log10(complete_frame5_vsvloess$value[x-1])) / (log10(complete_frame5_vsvloess$calibrated_score[x]) - log10(complete_frame5_vsvloess$calibrated_score[x-1])))
  complete_frame5_s1loess$derivative[x] <- ((log10(complete_frame5_s1loess$value[x]) - log10(complete_frame5_s1loess$value[x-1])) / (log10(complete_frame5_s1loess$calibrated_score[x]) - log10(complete_frame5_s1loess$calibrated_score[x-1])))
  complete_frame5_s2loess$derivative[x] <- ((log10(complete_frame5_s2loess$value[x]) - log10(complete_frame5_s2loess$value[x-1])) / (log10(complete_frame5_s2loess$calibrated_score[x]) - log10(complete_frame5_s2loess$calibrated_score[x-1])))
}
```

## Can we use position1 as a possible identifier position?

``` r
a1 <- complete_frame4_for_rf %>% filter(n6 == "A") %>% mutate(p5 = paste(n5,n4,n3,n2,n1, sep = ""))
c1 <- complete_frame4_for_rf %>% filter(n6 == "C") %>% mutate(p5 = paste(n5,n4,n3,n2,n1, sep = ""))
g1 <- complete_frame4_for_rf %>% filter(n6 == "G") %>% mutate(p5 = paste(n5,n4,n3,n2,n1, sep = ""))
t1 <- complete_frame4_for_rf %>% filter(n6 == "T") %>% mutate(p5 = paste(n5,n4,n3,n2,n1, sep = ""))

nucleotide1_pairwise_comparisons <- merge(a1[,c("calibrated_score","p5")],c1[,c("calibrated_score","p5")], by = "p5")
nucleotide1_pairwise_comparisons <- merge(nucleotide1_pairwise_comparisons,g1[,c("calibrated_score","p5")], by = "p5")
nucleotide1_pairwise_comparisons <- merge(nucleotide1_pairwise_comparisons,t1[,c("calibrated_score","p5")], by = "p5")
colnames(nucleotide1_pairwise_comparisons) <- c("p5","A","C","G","T")

First_nucleotide_held_constant_densityplot <- ggplot() + theme_bw() + labs(x = "Consensus-normalized predicted MFI", y = "Density of sequence variants") +
  scale_x_log10(limits = c(3e-3,3e0)) +
  geom_density(data = nucleotide1_pairwise_comparisons, aes(x = A), fill = "red", alpha = 0.2, color = "red") +
  geom_density(data = nucleotide1_pairwise_comparisons, aes(x = C), fill = "green", alpha = 0.2, color = "green") +
  geom_density(data = nucleotide1_pairwise_comparisons, aes(x = G), fill = "orange", alpha = 0.2, color = "orange") +
  geom_density(data = nucleotide1_pairwise_comparisons, aes(x = T), fill = "blue", alpha = 0.2, color = "blue")
First_nucleotide_held_constant_densityplot
```

![](Kozak_files/figure-gfm/Using%20position1%20as%20a%20possible%20identifier%20position-1.png)<!-- -->

``` r
ggsave(file = "Plots/First_nucleotide_held_constant_densityplot.pdf", First_nucleotide_held_constant_densityplot, height = 3, width = 5)

#library(reshape)
nucleotide1_pairwise_comparisons_melt <- melt(nucleotide1_pairwise_comparisons)

First_nucleotide_held_constant_histogram <- ggplot() + theme_bw() + 
  theme(panel.grid = element_blank()) + 
  labs(x = "Calibrated abundance", y = "Sequence variant count") + 
  scale_x_log10(limits = c(3e-3,2e0), expand = c(0,0)) +
  scale_y_continuous(breaks = c(0,100)) +
  geom_histogram(data = nucleotide1_pairwise_comparisons_melt, aes(x = value), fill = "grey80", color = "black", bins = 30) +
  facet_grid(rows = vars(variable)) +
  NULL; First_nucleotide_held_constant_histogram
```

![](Kozak_files/figure-gfm/Using%20position1%20as%20a%20possible%20identifier%20position-2.png)<!-- -->

``` r
ggsave(file = "Plots/First_nucleotide_held_constant_histogram.pdf", First_nucleotide_held_constant_histogram, height = 2, width = 2)
```

``` r
# Circle parameters
center_x <- 0
center_y <- 0
radius <- 5
#num_points <- 100

# Generate points to create a circle
theta <- seq(0, 2 * pi, length.out = 100)
circle_x <- center_x + radius * cos(theta)
circle_y <- center_y + radius * sin(theta)
circle_data <- data.frame(x = circle_x, y = circle_y)

# Function to generate random points within a circle
generate_random_points <- function(center_x, center_y, radius, num_points) {
  theta <- runif(num_points, 0, 2 * pi)
  r <- sqrt(runif(num_points, 0, 1)) * radius
  x <- center_x + r * cos(theta)
  y <- center_y + r * sin(theta)
  data.frame(x = x, y = y)
}

# Generate random points
random_points_low <- generate_random_points(center_x, center_y, radius*0.8, 1)
random_points_med <- generate_random_points(center_x, center_y, radius*0.8, 10)
random_points_high <- generate_random_points(center_x, center_y, radius*0.8, 100)

Low_density_cell_image <- ggplot() + theme_minimal() +
  theme(panel.grid = element_blank(), axis.text = element_blank()) +
  labs(x = NULL, y = NULL) +
  geom_path(data = circle_data, aes(x, y), color = "black", size = 1, alpha = 0.4) +
  geom_point(data = random_points_low, aes(x, y), size = 0.5, alpha = 0.3, shape = 16) +
  coord_fixed() +
  xlim(center_x - radius, center_x + radius) +
  ylim(center_y - radius, center_y + radius) +
  NULL
Low_density_cell_image
```

![](Kozak_files/figure-gfm/Make%20a%20visual%20representation%20of%20receptor%20density-1.png)<!-- -->

``` r
ggsave(file = "Plots/Low_density_cell_image.pdf", Low_density_cell_image, height = 0.55, width = 0.55)

Med_density_cell_image <- ggplot() + theme_minimal() +
  theme(panel.grid = element_blank(), axis.text = element_blank()) +
  labs(x = NULL, y = NULL) +
  geom_path(data = circle_data, aes(x, y), color = "black", size = 1, alpha = 0.4) +
  geom_point(data = random_points_med, aes(x, y), size = 0.5, alpha = 0.3, shape = 16) +
  coord_fixed() +
  xlim(center_x - radius, center_x + radius) +
  ylim(center_y - radius, center_y + radius) +
  NULL
Med_density_cell_image
```

![](Kozak_files/figure-gfm/Make%20a%20visual%20representation%20of%20receptor%20density-2.png)<!-- -->

``` r
ggsave(file = "Plots/Med_density_cell_image.pdf", Med_density_cell_image, height = 0.55, width = 0.55)

High_density_cell_image <- ggplot() + theme_minimal() +
  theme(panel.grid = element_blank(), axis.text = element_blank()) +
  labs(x = NULL, y = NULL) +
  geom_path(data = circle_data, aes(x, y), color = "black", size = 1, alpha = 0.4) +
  geom_point(data = random_points_high, aes(x, y), size = 0.5, alpha = 0.3, shape = 16) +
  coord_fixed() +
  xlim(center_x - radius, center_x + radius) +
  ylim(center_y - radius, center_y + radius) +
  NULL
High_density_cell_image
```

![](Kozak_files/figure-gfm/Make%20a%20visual%20representation%20of%20receptor%20density-3.png)<!-- -->

``` r
ggsave(file = "Plots/High_density_cell_image.pdf", High_density_cell_image, height = 0.55, width = 0.55)
```

``` r
kozak_and_variant_key <- read.csv(file = "Keys/Kozak_and_variant_key.csv", header = T)
mini_lib_plasmid <- read.delim(file = "Data/Amplicon_EZ/ACE2-Kozak-mini-library-pooled_joined.tsv", sep = "\t")
mini_lib_plasmid_counted <- count(mini_lib_plasmid, kozak_sequence)

## First compare all of the unselecteds for the first experiment
sa1_neg_1 <- read.delim(file = myfiles[153], sep = "\t")
sa1_neg_2 <- read.delim(file = myfiles[154], sep = "\t")

sa1_negs <- merge(sa1_neg_1, sa1_neg_2, by = "X", all.x = T)
colnames(sa1_negs) <- c("kozak","neg1","neg2")
sa1_negs$n1 <- substr(sa1_negs$kozak,1,1)
sa1_negs <- merge(sa1_negs, kozak_and_variant_key, by = "n1", all.x = T)

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "I0153", y = "I0154") +
  geom_point(data = sa1_negs, aes(x = neg1, y = neg2, color = variant), alpha = 0.2)
```

![](Kozak_files/figure-gfm/Now%20looking%20at%20the%20mini-library%20data-1.png)<!-- -->

``` r
## First compare all of the unselecteds for the second experiment
sa2_neg_1 <- read.delim(file = myfiles[168], sep = "\t")
sa2_neg_2 <- read.delim(file = myfiles[169], sep = "\t")
sa2_neg_3 <- read.delim(file = myfiles[170], sep = "\t")

sa2_negs <- merge(sa2_neg_1, sa2_neg_2, by = "X", all.x = T)
sa2_negs <- merge(sa2_negs, sa2_neg_3, by = "X", all.x = T)
colnames(sa2_negs) <- c("kozak","neg1","neg2","neg3")
sa2_negs$n1 <- substr(sa2_negs$kozak,1,1)
sa2_negs <- merge(sa2_negs, kozak_and_variant_key, by = "n1", all.x = T)

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "I0168", y = "I0169") +
  geom_point(data = sa2_negs, aes(x = neg1, y = neg2, color = variant), alpha = 0.2)
```

![](Kozak_files/figure-gfm/Now%20looking%20at%20the%20mini-library%20data-2.png)<!-- -->

``` r
ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "I0168", y = "I0170") +
  geom_point(data = sa2_negs, aes(x = neg1, y = neg3, color = variant), alpha = 0.2)
```

![](Kozak_files/figure-gfm/Now%20looking%20at%20the%20mini-library%20data-3.png)<!-- -->

``` r
ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "I0169", y = "I0170") +
  geom_point(data = sa2_negs, aes(x = neg2, y = neg3, color = variant), alpha = 0.2)
```

![](Kozak_files/figure-gfm/Now%20looking%20at%20the%20mini-library%20data-4.png)<!-- -->

``` r
## Write tables that can b4 used as neg controls
sa1_neg <- rbind(sa1_neg_1, sa1_neg_2) %>% group_by(X) %>% summarize(count = sum(count))
colnames(sa1_neg) <- c("","count")
write.table(file = "Data/NextSeq003/sa1_unselected.tsv", sa1_neg, row.names = F, quote = F, sep = "\t")

sa2_neg <- rbind(sa2_neg_1, sa2_neg_2, sa2_neg_3) %>% group_by(X) %>% summarize(count = sum(count))
colnames(sa2_neg) <- c("","count")
write.table(file = "Data/NextSeq003/sa2_unselected.tsv", sa2_neg, row.names = F, quote = F, sep = "\t")

sa1_sa2_comb <- merge(sa1_neg, sa2_neg, by = "")
sa1_sa2_comb$kozak_sequence <- substr(sa1_sa2_comb$Var.1, 1,6)

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "SA1", y = "SA2") +
  geom_point(data = sa1_sa2_comb, aes(x = count.x, y = count.y), alpha = 0.2)
```

![](Kozak_files/figure-gfm/Now%20looking%20at%20the%20mini-library%20data-5.png)<!-- -->

``` r
sa1_sa2_comb2 <- merge(sa1_sa2_comb, mini_lib_plasmid_counted,  by = "kozak_sequence")
```

``` r
## Rep 1
sa1_vsv_low <- read.delim(file = myfiles[155], sep = "\t")
sa1_vsv_high <- read.delim(file = myfiles[156], sep = "\t")
sa1_vsv_low2 <- read.delim(file = myfiles[167], sep = "\t")
sa1_vsv_counts <- merge(sa1_vsv_low, sa1_vsv_high, by = "X", all = T)
sa1_vsv_counts <- merge(sa1_vsv_counts, sa1_vsv_low2, by = "X", all = T)
colnames(sa1_vsv_counts) <- c("kozak","low","high","low2")

ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa1_vsv_counts, aes(x = low, y = high), alpha = 0.2)
```

![](Kozak_files/figure-gfm/VSV%20kozak%20and%20sequence%20rep1-1.png)<!-- -->

``` r
ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa1_vsv_counts, aes(x = low, y = low2), alpha = 0.2)
```

![](Kozak_files/figure-gfm/VSV%20kozak%20and%20sequence%20rep1-2.png)<!-- -->

``` r
ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa1_vsv_counts, aes(x = high, y = low2), alpha = 0.2)
```

![](Kozak_files/figure-gfm/VSV%20kozak%20and%20sequence%20rep1-3.png)<!-- -->

``` r
myfiles[c(192,193,155,156,167)]
```

    ## [1] "Data/NextSeq003/sa1_unselected.tsv" "Data/NextSeq003/sa2_unselected.tsv"
    ## [3] "Data/NextSeq003/I0155_lib.tsv"      "Data/NextSeq003/I0156_lib.tsv"     
    ## [5] "Data/NextSeq003/I0167_lib.tsv"

``` r
sa1_vsv_r1 <- makeExperimentFrame2(c(192,193,155,155)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(total_reads))
sa1_vsv_r2 <- makeExperimentFrame2(c(192,193,156,156)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(total_reads))
sa1_vsv_r3 <- makeExperimentFrame2(c(192,193,167,167)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(total_reads))

sa1_vsv <- merge(merge(sa1_vsv_r1[,c("sequence","h_enrichment")], sa1_vsv_r2[,c("sequence","h_enrichment")], by = "sequence", all = T), sa1_vsv_r3[,c("sequence","h_enrichment")], by = "sequence", all = T)
colnames(sa1_vsv) <- c("sequence","low","high","low2")

sa1_vsv$h_enrichment <- (sa1_vsv$low + sa1_vsv$high + sa1_vsv$low2)/3

sa1_vsv <- merge(sa1_vsv, complete_frame6[,c("sequence","calibrated_score")])
sa1_vsv$n1 <- substr(sa1_vsv$sequence,1,1)
sa1_vsv <- merge(sa1_vsv, kozak_and_variant_key, by = "n1", all.x = T)
sa1_vsv_template_henrichment <- (sa1_vsv %>% filter(sequence == "XXXXXXXXX"))$h_enrichment
sa1_vsv <- sa1_vsv %>% filter(variant %in% c("WT","I21N","D355N"))
sa1_vsv$variant <- factor(sa1_vsv$variant, levels = c("WT","I21N","K31D","D355N"))

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_point(data = sa1_vsv, aes(x = calibrated_score, y = h_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/VSV%20kozak%20and%20sequence%20rep1-4.png)<!-- -->

``` r
## Rep 2
sa1_vsv_r4 <- read.delim(file = myfiles[171], sep = "\t")
sa1_vsv_r5 <- read.delim(file = myfiles[172], sep = "\t")
sa1_vsv_r6 <- read.delim(file = myfiles[183], sep = "\t")
sa2_vsv_counts <- merge(sa1_vsv_r4, sa1_vsv_r5, by = "X", all = T)
sa2_vsv_counts <- merge(sa2_vsv_counts, sa1_vsv_r6, by = "X", all = T)
colnames(sa2_vsv_counts) <- c("kozak","r4","r5","r6")
## I0171 is missing data for some reason

sa2_vsv <- makeExperimentFrame2(c(192,193,172,183)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa2_vsv <- merge(sa2_vsv, complete_frame6[,c("sequence","calibrated_score")])
sa2_vsv$n1 <- substr(sa2_vsv$sequence,1,1)
sa2_vsv <- merge(sa2_vsv, kozak_and_variant_key, by = "n1", all.x = T)
sa2_vsv_template_henrichment <- (sa2_vsv %>% filter(sequence == "XXXXXXXXX"))$h_enrichment
sa2_vsv <- sa2_vsv %>% filter(variant %in% c("WT","I21N","D355N"))
sa2_vsv$variant <- factor(sa2_vsv$variant, levels = c("WT","I21N","K31D","D355N"))

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_point(data = sa2_vsv, aes(x = calibrated_score, y = h_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/VSV%20kozak%20and%20sequence%20rep1-5.png)<!-- -->

``` r
## Now average the values
sa_vsv <- merge(sa1_vsv[,c("sequence","calibrated_score","variant","h_enrichment")], sa2_vsv[,c("sequence","h_enrichment")], by = "sequence")
sa_vsv$h_enrichment.x2 <- sa_vsv$h_enrichment.x / mean((sa_vsv %>% filter(variant == "WT" & calibrated_score < 1.1 & calibrated_score > 0.9))$h_enrichment.x, na.rm = T)
sa_vsv$h_enrichment.y2 <- sa_vsv$h_enrichment.y / mean((sa_vsv %>% filter(variant == "WT" & calibrated_score < 1.1 & calibrated_score > 0.9))$h_enrichment.y, na.rm = T)
sa_vsv$ave_enrichment <- (sa_vsv$h_enrichment.x2 + sa_vsv$h_enrichment.y2)/2
sa_vsv_template_henrichment <- (sa1_vsv_template_henrichment + sa2_vsv_template_henrichment)/2

ggplot() + scale_x_log10() + scale_y_log10(limits = c(0.003,10)) + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_hline(yintercept = sa_vsv_template_henrichment, color = "purple", size = 2, alpha = 0.5) +
  geom_point(data = sa_vsv, aes(x = calibrated_score, y = ave_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/VSV%20kozak%20and%20sequence%20rep1-6.png)<!-- -->

``` r
## Custom test pair
test_pair1 <- read.delim(file = myfiles[181], sep = "\t")
test_pair2 <- read.delim(file = myfiles[182], sep = "\t")
test_pair <- merge(test_pair1, test_pair2, by = "X", all = T)

ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = test_pair, aes(x = count.x, y = count.y), alpha = 0.2)
```

![](Kozak_files/figure-gfm/Pairwise%20tests%20for%20SARS1%20samples-1.png)<!-- -->

``` r
sa1_s1_r1 <- makeExperimentFrame2(c(153,154,157,157)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa1_s1_r2 <- makeExperimentFrame2(c(153,154,158,158)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa1_s1_r3 <- makeExperimentFrame2(c(153,154,165,165)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))

sa1_s1 <- merge(merge(sa1_s1_r1[,c("sequence","h_enrichment")], sa1_s1_r2[,c("sequence","h_enrichment")], by = "sequence", all = T), sa1_s1_r3[,c("sequence","h_enrichment")], by = "sequence", all = T)
colnames(sa1_s1) <- c("sequence","low","high","low2")

sa1_s1$h_enrichment <- (sa1_s1$low + sa1_s1$high + sa1_s1$low2)/3

sa1_s1 <- merge(sa1_s1, complete_frame6[,c("sequence","calibrated_score")])
sa1_s1$n1 <- substr(sa1_s1$sequence,1,1)
sa1_s1 <- merge(sa1_s1, kozak_and_variant_key, by = "n1", all.x = T)
sa1_s1_template_henrichment <- (sa1_s1 %>% filter(sequence == "XXXXXXXXX"))$h_enrichment
sa1_s1 <- sa1_s1 %>% filter(variant %in% c("WT","I21N","D355N"))
sa1_s1$variant <- factor(sa1_s1$variant, levels = c("WT","I21N","K31D","D355N"))

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_point(data = sa1_s1, aes(x = calibrated_score, y = h_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/SARS1%20kozak%20and%20sequence%20rep1-1.png)<!-- -->

``` r
## Rep 2
sa2_s1_low <- read.delim(file = myfiles[173], sep = "\t")
sa2_s1_high <- read.delim(file = myfiles[174], sep = "\t")
sa2_s1_low2 <- read.delim(file = myfiles[181], sep = "\t")
sa2_s1_counts <- merge(sa2_s1_low, sa2_s1_high, by = "X", all = T)
sa2_s1_counts <- merge(sa2_s1_counts, sa2_s1_low2, by = "X", all = T)
colnames(sa2_s1_counts) <- c("kozak","low","high","low2")

myfiles[c(169,170,173,174,182)]
```

    ## [1] "Data/NextSeq003/I0169_lib.tsv" "Data/NextSeq003/I0170_lib.tsv"
    ## [3] "Data/NextSeq003/I0173_lib.tsv" "Data/NextSeq003/I0174_lib.tsv"
    ## [5] "Data/NextSeq003/I0182_lib.tsv"

``` r
sa2_s1_r1 <- makeExperimentFrame2(c(169,170,173,173)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa2_s1_r2 <- makeExperimentFrame2(c(169,170,174,174)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa2_s1_r3 <- makeExperimentFrame2(c(169,170,181,181)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))

sa2_s1 <- merge(merge(sa2_s1_r1[,c("sequence","h_enrichment")], sa2_s1_r2[,c("sequence","h_enrichment")], by = "sequence", all = T), sa2_s1_r3[,c("sequence","h_enrichment")], by = "sequence", all = T)
colnames(sa2_s1) <- c("sequence","low","high","low2")

sa2_s1$h_enrichment <- (sa2_s1$low + sa2_s1$high + sa2_s1$low2)/3

sa2_s1 <- merge(sa2_s1, complete_frame6[,c("sequence","calibrated_score")])
sa2_s1$n1 <- substr(sa2_s1$sequence,1,1)
sa2_s1 <- merge(sa2_s1, kozak_and_variant_key, by = "n1", all.x = T)
sa2_s1_template_henrichment <- (sa2_s1 %>% filter(sequence == "XXXXXXXXX"))$h_enrichment
sa2_s1 <- sa2_s1 %>% filter(variant %in% c("WT","I21N","D355N"))
sa2_s1$variant <- factor(sa2_s1$variant, levels = c("WT","I21N","K31D","D355N"))

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_point(data = sa2_s1, aes(x = calibrated_score, y = h_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/SARS1%20kozak%20and%20sequence%20rep1-2.png)<!-- -->

``` r
## Now average the values
sa_s1 <- merge(sa1_s1[,c("sequence","calibrated_score","variant","h_enrichment")], sa2_s1[,c("sequence","h_enrichment")], by = "sequence")
sa_s1$h_enrichment.x2 <- sa_s1$h_enrichment.x / mean((sa_s1 %>% filter(variant == "WT" & calibrated_score < 1.1 & calibrated_score > 0.9))$h_enrichment.x, na.rm = T)
sa_s1$h_enrichment.y2 <- sa_s1$h_enrichment.y / mean((sa_s1 %>% filter(variant == "WT" & calibrated_score < 1.1 & calibrated_score > 0.9))$h_enrichment.y, na.rm = T)
sa_s1$ave_enrichment <- (sa_s1$h_enrichment.x2 + sa_s1$h_enrichment.y2)/2
sa_s1_template_henrichment <- (sa1_s1_template_henrichment + sa2_s1_template_henrichment)/2

ggplot() + scale_x_log10() + scale_y_log10(limits = c(0.003,3)) + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_hline(yintercept = sa_s1_template_henrichment, color = "purple", size = 2, alpha = 0.5) +
  geom_point(data = sa_s1, aes(x = calibrated_score, y = ave_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/SARS1%20kozak%20and%20sequence%20rep1-3.png)<!-- -->

``` r
sa1_s2_low <- read.delim(file = myfiles[159], sep = "\t")
sa1_s2_high <- read.delim(file = myfiles[160], sep = "\t")
sa1_s2_low2 <- read.delim(file = myfiles[166], sep = "\t")
sa1_s2_counts <- merge(sa1_s2_low, sa1_s2_high, by = "X", all = T)
sa1_s2_counts <- merge(sa1_s2_counts, sa1_s2_low2, by = "X", all = T)
colnames(sa1_s2_counts) <- c("kozak","low","high","low2")

ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa1_s2_counts, aes(x = low, y = high), alpha = 0.2)
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-1.png)<!-- -->

``` r
ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa1_s2_counts, aes(x = low, y = low2), alpha = 0.2)
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-2.png)<!-- -->

``` r
ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa1_s2_counts, aes(x = high, y = low2), alpha = 0.2)
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-3.png)<!-- -->

``` r
sa1_s2_r1 <- makeExperimentFrame2(c(153,154,159,159)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa1_s2_r2 <- makeExperimentFrame2(c(153,154,160,160)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa1_s2_r3 <- makeExperimentFrame2(c(153,154,166,166)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))

sa1_s2 <- merge(merge(sa1_s2_r1[,c("sequence","h_enrichment")], sa1_s2_r2[,c("sequence","h_enrichment")], by = "sequence", all = T), sa1_s2_r3[,c("sequence","h_enrichment")], by = "sequence", all = T)
colnames(sa1_s2) <- c("sequence","low","high","low2")

sa1_s2$h_enrichment <- (sa1_s2$low + sa1_s2$high + sa1_s2$low2)/3

sa1_s2 <- merge(sa1_s2, complete_frame6[,c("sequence","calibrated_score")])
sa1_s2$n1 <- substr(sa1_s2$sequence,1,1)
sa1_s2 <- merge(sa1_s2, kozak_and_variant_key, by = "n1", all.x = T)
sa1_s2_template_henrichment <- (sa1_s2 %>% filter(sequence == "XXXXXXXXX"))$h_enrichment
sa1_s2 <- sa1_s2 %>% filter(variant %in% c("WT","I21N","D355N"))
sa1_s2$variant <- factor(sa1_s2$variant, levels = c("WT","I21N","K31D","D355N"))

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_point(data = sa1_s2, aes(x = calibrated_score, y = h_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-4.png)<!-- -->

``` r
## Rep 2
sa2_s2_low <- read.delim(file = myfiles[175], sep = "\t")
sa2_s2_high <- read.delim(file = myfiles[176], sep = "\t")
sa2_s2_low2 <- read.delim(file = myfiles[182], sep = "\t")
sa2_s2_counts <- merge(sa2_s2_low, sa2_s2_high, by = "X", all = T)
sa2_s2_counts <- merge(sa2_s2_counts, sa2_s2_low2, by = "X", all = T)
colnames(sa2_s2_counts) <- c("kozak","low","high","low2")

ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa2_s2_counts, aes(x = low, y = high), alpha = 0.2)
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-5.png)<!-- -->

``` r
ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa2_s2_counts, aes(x = low, y = low2), alpha = 0.2)
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-6.png)<!-- -->

``` r
ggplot() + scale_x_log10() + scale_y_log10() + 
  geom_point(data = sa2_s2_counts, aes(x = high, y = low2), alpha = 0.2)
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-7.png)<!-- -->

``` r
sa2_s2_r1 <- makeExperimentFrame2(c(169,170,175,175)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa2_s2_r2 <- makeExperimentFrame2(c(169,170,176,176)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))
sa2_s2_r3 <- makeExperimentFrame2(c(169,170,182,182)) %>% filter(h_enrichment != "Inf") %>% arrange(desc(h_enrichment))

sa2_s2 <- merge(merge(sa2_s2_r1[,c("sequence","h_enrichment")], sa2_s2_r2[,c("sequence","h_enrichment")], by = "sequence", all = T), sa2_s2_r3[,c("sequence","h_enrichment")], by = "sequence", all = T)
colnames(sa2_s2) <- c("sequence","low","high","low2")

sa2_s2$h_enrichment <- (sa2_s2$low + sa2_s2$high + sa2_s2$low2)/3

sa2_s2 <- merge(sa2_s2, complete_frame6[,c("sequence","calibrated_score")])
sa2_s2$n1 <- substr(sa2_s2$sequence,1,1)
sa2_s2 <- merge(sa2_s2, kozak_and_variant_key, by = "n1", all.x = T)
sa2_s2_template_henrichment <- (sa2_s2 %>% filter(sequence == "XXXXXXXXX"))$h_enrichment
sa2_s2 <- sa2_s2 %>% filter(variant %in% c("WT","I21N","D355N"))
sa2_s2$variant <- factor(sa2_s2$variant, levels = c("WT","I21N","K31D","D355N"))

ggplot() + scale_x_log10() + scale_y_log10() + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_point(data = sa2_s2, aes(x = calibrated_score, y = h_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-8.png)<!-- -->

``` r
## Now average the values
sa_s2 <- merge(sa1_s2[,c("sequence","calibrated_score","variant","h_enrichment")], sa2_s2[,c("sequence","h_enrichment")], by = "sequence")
sa_s2$h_enrichment.x2 <- sa_s2$h_enrichment.x / mean((sa_s2 %>% filter(variant == "WT" & calibrated_score < 1.1 & calibrated_score > 0.9))$h_enrichment.x, na.rm = T)
sa_s2$h_enrichment.y2 <- sa_s2$h_enrichment.y / mean((sa_s2 %>% filter(variant == "WT" & calibrated_score < 1.1 & calibrated_score > 0.9))$h_enrichment.y, na.rm = T)
sa_s2$ave_enrichment <- (sa_s2$h_enrichment.x2 + sa_s2$h_enrichment.y2)/2
sa_s2_template_henrichment <- (sa1_s2_template_henrichment + sa2_s2_template_henrichment)/2

ggplot() + scale_x_log10() + scale_y_log10(limits = c(0.003,3)) + labs(x = "Calibrated abundance level", y = "Rate of infection (enrichment upon hygro treatment)") +
  geom_hline(yintercept = sa_s2_template_henrichment, color = "purple", size = 2, alpha = 0.5) +
  geom_point(data = sa_s2, aes(x = calibrated_score, y = ave_enrichment, color = variant), alpha = 0.2) +
  facet_grid(cols = vars(variant))
```

![](Kozak_files/figure-gfm/SARS2%20kozak%20and%20sequence%20rep1-9.png)<!-- -->

``` r
template_seqabund_dataframe <- data.frame("virus" = c("VSV","SARS-CoV","SARS-CoV-2"),
                                          "template" = c(sa_vsv_template_henrichment,sa_s1_template_henrichment,sa_s2_template_henrichment))
template_seqabund_dataframe$virus <- factor(template_seqabund_dataframe$virus, levels = c("VSV","SARS-CoV","SARS-CoV-2"))

sequence_abundance_compiled <- rbind(sa_vsv[,c("sequence","variant","calibrated_score","ave_enrichment")] %>% mutate(virus = "VSV"),
                                     sa_s1[,c("sequence","variant","calibrated_score","ave_enrichment")] %>% mutate(virus = "SARS-CoV"),
                                     sa_s2[,c("sequence","variant","calibrated_score","ave_enrichment")] %>% mutate(virus = "SARS-CoV-2"))
sequence_abundance_compiled$virus <- factor(sequence_abundance_compiled$virus, levels = c("VSV","SARS-CoV","SARS-CoV-2"))

Sequence_and_abundance_scatterplots <- ggplot() + theme(panel.grid = element_blank(), legend.position = "none") + 
  scale_x_log10() + scale_y_log10(limits = c(0.1,3), breaks = c(0.1,1)) + labs(x = "calibrated abundance", y = "Enrichment upon infection)") +
  geom_hline(data = template_seqabund_dataframe, (aes(yintercept = template)), color = "purple", alpha = 0.5, size = 1) +
  geom_point(data = sequence_abundance_compiled, aes(x = calibrated_score, y = ave_enrichment), alpha = 0.2, size = 1) +
  stat_smooth(data= sequence_abundance_compiled, aes(x = calibrated_score, y = ave_enrichment, color = variant), geom='line', alpha=0.6, se=FALSE, size = 2) +
  facet_grid(cols = vars(virus), rows = vars(variant)) +
  NULL; Sequence_and_abundance_scatterplots
```

![](Kozak_files/figure-gfm/Making%20a%20combined%20graph%20for%20the%20minilibrary%20data-1.png)<!-- -->

``` r
ggsave(file = "Plots/Sequence_and_abundance_scatterplots.pdf", Sequence_and_abundance_scatterplots, height = 2.3, width = 4.5)

Sequence_and_abundance_lineplots <- ggplot() + theme(panel.grid = element_blank(), legend.position = "bottom") + 
  scale_x_log10() + scale_y_log10(limits = c(0.1,3)) + labs(x = "calibrated abundance", y = "Enrichment upon infection") +
  #geom_hline(data = template_seqabund_dataframe, (aes(yintercept = template)), color = "purple", alpha = 0.5, size = 1) +
  geom_point(data = sequence_abundance_compiled, aes(x = calibrated_score, y = ave_enrichment), alpha = 0, size = 1) +
  stat_smooth(data= sequence_abundance_compiled, aes(x = calibrated_score, y = ave_enrichment, color = variant), geom='line', alpha=0.6, se=FALSE, size = 0.75) +
  facet_grid(cols = vars(virus)) +
  NULL; Sequence_and_abundance_lineplots
```

![](Kozak_files/figure-gfm/Making%20a%20combined%20graph%20for%20the%20minilibrary%20data-2.png)<!-- -->

``` r
ggsave(file = "Plots/Sequence_and_abundance_lineplots.pdf", Sequence_and_abundance_lineplots, height = 2, width = 4.25)
```

``` r
loess_span <- 0.75
newvals <- data.frame(calibrated_score=seq(0.01,1.02,.01))

## SARS1 with the ACE2 mutants
sequence_abundance_compiled_s1_wt <- sequence_abundance_compiled %>% filter(virus == "SARS-CoV" & variant == "WT" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
s1_wt_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_s1_wt, span = loess_span)
sequence_abundance_compiled_s1_wt_loess <- cbind(newvals, value=predict(s1_wt_loess, newvals))

sequence_abundance_compiled_s1_i21n <- sequence_abundance_compiled %>% filter(virus == "SARS-CoV" & variant == "I21N" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
s1_i21n_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_s1_i21n, span = loess_span)
sequence_abundance_compiled_s1_i21n_loess <- cbind(newvals, value=predict(s1_i21n_loess, newvals))

sequence_abundance_compiled_s1_d355n <- sequence_abundance_compiled %>% filter(virus == "SARS-CoV" & variant == "D355N" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
s1_d355n_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_s1_d355n, span = loess_span)
sequence_abundance_compiled_s1_d355n_loess <- cbind(newvals, value=predict(s1_d355n_loess, newvals))

ggplot() + labs(title = "SARS-CoV Spike") +
  scale_x_log10(limits = c(0.02,1)) + scale_y_log10(limits = c(0.1,1.5)) + 
  geom_point(data = sequence_abundance_compiled_s1_wt, aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1) +
  geom_line(data = sequence_abundance_compiled_s1_wt_loess, aes(x = calibrated_score, y = value)) +
  geom_point(data = sequence_abundance_compiled_s1_i21n , aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1, color = "red") +
  geom_line(data = sequence_abundance_compiled_s1_i21n_loess, aes(x = calibrated_score, y = value), color = "red") +
  geom_point(data = sequence_abundance_compiled_s1_d355n , aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1, color = "cyan") +
  geom_line(data = sequence_abundance_compiled_s1_d355n_loess, aes(x = calibrated_score, y = value), color = "cyan") +
  NULL
```

![](Kozak_files/figure-gfm/Performing%20LOESS%20off%20minilibrary%20data%20for%20SARS1-1.png)<!-- -->

``` r
## Now SARS2 with the ACE2 mutants
sequence_abundance_compiled_s2_wt <- sequence_abundance_compiled %>% filter(virus == "SARS-CoV-2" & variant == "WT" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
s2_wt_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_s2_wt, span = loess_span)
sequence_abundance_compiled_s2_wt_loess <- cbind(newvals, value=predict(s2_wt_loess, newvals))

sequence_abundance_compiled_s2_i21n <- sequence_abundance_compiled %>% filter(virus == "SARS-CoV-2" & variant == "I21N" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
s2_i21n_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_s2_i21n, span = loess_span)
sequence_abundance_compiled_s2_i21n_loess <- cbind(newvals, value=predict(s2_i21n_loess, newvals))

sequence_abundance_compiled_s2_d355n <- sequence_abundance_compiled %>% filter(virus == "SARS-CoV-2" & variant == "D355N" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
s2_d355n_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_s2_d355n, span = loess_span)
sequence_abundance_compiled_s2_d355n_loess <- cbind(newvals, value=predict(s2_d355n_loess, newvals))

ggplot() + labs(title = "SARS-CoV-2 Spike") +
  scale_x_log10(limits = c(0.02,1)) + scale_y_log10(limits = c(0.1,1.5)) + 
  geom_point(data = sequence_abundance_compiled_s2_wt, aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1) +
  geom_line(data = sequence_abundance_compiled_s2_wt_loess, aes(x = calibrated_score, y = value)) +
  geom_point(data = sequence_abundance_compiled_s2_i21n , aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1, color = "red") +
  geom_line(data = sequence_abundance_compiled_s2_i21n_loess, aes(x = calibrated_score, y = value), color = "red") +
  geom_point(data = sequence_abundance_compiled_s2_d355n , aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1, color = "cyan") +
  geom_line(data = sequence_abundance_compiled_s2_d355n_loess, aes(x = calibrated_score, y = value), color = "cyan") +
  NULL
```

![](Kozak_files/figure-gfm/Performing%20LOESS%20off%20minilibrary%20data%20for%20SARS2-1.png)<!-- -->

``` r
## Now VSV with the ACE2 mutants
sequence_abundance_compiled_vsv_wt <- sequence_abundance_compiled %>% filter(virus == "VSV" & variant == "WT" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
vsv_wt_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_vsv_wt, span = loess_span)
sequence_abundance_compiled_vsv_wt_loess <- cbind(newvals, value=predict(vsv_wt_loess, newvals))

sequence_abundance_compiled_vsv_i21n <- sequence_abundance_compiled %>% filter(virus == "VSV" & variant == "I21N" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
vsv_i21n_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_vsv_i21n, span = loess_span)
sequence_abundance_compiled_vsv_i21n_loess <- cbind(newvals, value=predict(vsv_i21n_loess, newvals))

sequence_abundance_compiled_vsv_d355n <- sequence_abundance_compiled %>% filter(virus == "VSV" & variant == "D355N" & !is.na(ave_enrichment)) %>% arrange(calibrated_score)
vsv_d355n_loess <- loess(ave_enrichment~calibrated_score, data=sequence_abundance_compiled_vsv_d355n, span = loess_span)
sequence_abundance_compiled_vsv_d355n_loess <- cbind(newvals, value=predict(vsv_d355n_loess, newvals))

ggplot() + labs(title = "VSV-G") +
  scale_x_log10(limits = c(0.02,1.5)) + scale_y_log10(limits = c(0.1,1.5)) + 
  geom_point(data = sequence_abundance_compiled_vsv_wt, aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1) +
  geom_line(data = sequence_abundance_compiled_vsv_wt_loess, aes(x = calibrated_score, y = value)) +
  geom_point(data = sequence_abundance_compiled_vsv_i21n , aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1, color = "red") +
  geom_line(data = sequence_abundance_compiled_vsv_i21n_loess, aes(x = calibrated_score, y = value), color = "red") +
  geom_point(data = sequence_abundance_compiled_vsv_d355n , aes(x = calibrated_score, y = ave_enrichment), size = 1, alpha = 0.1, color = "cyan") +
  geom_line(data = sequence_abundance_compiled_vsv_d355n_loess, aes(x = calibrated_score, y = value), color = "cyan") +
  NULL
```

![](Kozak_files/figure-gfm/Performing%20LOESS%20off%20minilibrary%20data%20for%20VSV-G-1.png)<!-- -->

``` r
write.csv(file = "Output_datatables/Supp_table_2_mixed_library_infection.csv", sequence_abundance_compiled, row.names = F, quote = F)
```

## Now bringing in the STIM1 experiment

``` r
klib_raw = read.table("Data/STIM1_Kozak/klib.assembled.tsv", header = TRUE)
kr1_raw = read.table("Data/STIM1_Kozak/k_r1.assembled.tsv", header = TRUE)
kr2_raw = read.table("Data/STIM1_Kozak/k_r2.assembled.tsv", header = TRUE)
kr3_raw = read.table("Data/STIM1_Kozak/k_r3.assembled.tsv", header = TRUE)

klib_temp = klib_raw %>% count(read_class)
kr1_temp = kr1_raw %>% count(read_class)
kr2_temp = kr2_raw %>% count(read_class)
kr3_temp = kr3_raw %>% count(read_class)

klib = klib_raw %>% filter(read_class == "Kozak") %>% count(kozak_sequence)
kr1 = kr1_raw %>% filter(read_class == "Kozak") %>% count(kozak_sequence)
kr2 = kr2_raw %>% filter(read_class == "Kozak") %>% count(kozak_sequence)
kr3 = kr3_raw %>% filter(read_class == "Kozak") %>% count(kozak_sequence)

klib_comb <- rbind(klib,data.frame("kozak_sequence" = c("_OTHER","_TEMPL"), "n" = c(klib_temp$n[2],klib_temp$n[3]))) %>% mutate(log10_count = log10(n))
kr1_comb <- rbind(kr1,data.frame("kozak_sequence" = c("_OTHER","_TEMPL"), "n" = c(kr1_temp$n[2],klib_temp$n[3]))) %>% mutate(log10_count = log10(n))
kr2_comb <- rbind(kr2,data.frame("kozak_sequence" = c("_OTHER","_TEMPL"), "n" = c(kr2_temp$n[2],klib_temp$n[3]))) %>% mutate(log10_count = log10(n))
kr3_comb <- rbind(kr3,data.frame("kozak_sequence" = c("_OTHER","_TEMPL"), "n" = c(kr3_temp$n[2],klib_temp$n[3]))) %>% mutate(log10_count = log10(n))

klib_density <- ggplot() + geom_density(data = klib_comb, aes(x = log10_count))
klib_density_table <- ggplot_build(klib_density)$data[[1]]
klib_minquant <- klib_density_table %>% filter(x < quantile(klib_density_table$x, minquant_fraction))
klib_density_minima <- klib_minquant[klib_minquant$y == min(klib_minquant$y),"x"]
klib_density <- ggplot() + geom_density(data = klib, aes(x = log10_count)) + geom_vline(xintercept = klib_density_minima)
klib_filtered <- klib_comb %>% filter(log10_count > klib_density_minima)

kr1_density <- ggplot() + geom_density(data = kr1_comb, aes(x = log10_count))
kr1_density_table <- ggplot_build(kr1_density)$data[[1]]
kr1_minquant <- kr1_density_table %>% filter(x < quantile(kr1_density_table$x, minquant_fraction))
kr1_density_minima <- kr1_minquant[kr1_minquant$y == min(kr1_minquant$y),"x"]
kr1_density <- ggplot() + geom_density(data = kr1, aes(x = log10_count)) + geom_vline(xintercept = kr1_density_minima)
kr1_filtered <- kr1_comb %>% filter(log10_count > kr1_density_minima)

kr2_density <- ggplot() + geom_density(data = kr2_comb, aes(x = log10_count))
kr2_density_table <- ggplot_build(kr2_density)$data[[1]]
kr2_minquant <- kr2_density_table %>% filter(x < quantile(kr2_density_table$x, minquant_fraction))
kr2_density_minima <- kr2_minquant[kr2_minquant$y == min(kr2_minquant$y),"x"]
kr2_density <- ggplot() + geom_density(data = kr2, aes(x = log10_count)) + geom_vline(xintercept = kr2_density_minima)
kr2_filtered <- kr2_comb %>% filter(log10_count > kr2_density_minima)

kr3_density <- ggplot() + geom_density(data = kr3_comb, aes(x = log10_count))
kr3_density_table <- ggplot_build(kr3_density)$data[[1]]
kr3_minquant <- kr3_density_table %>% filter(x < quantile(kr3_density_table$x, minquant_fraction))
kr3_density_minima <- kr3_minquant[kr3_minquant$y == min(kr3_minquant$y),"x"]
kr3_density <- ggplot() + geom_density(data = kr3, aes(x = log10_count)) + geom_vline(xintercept = kr3_density_minima)
kr3_filtered <- kr3_comb %>% filter(log10_count > kr3_density_minima)
```

``` r
klib_freq = klib_filtered %>% mutate(n0 = n) %>% mutate(f0 = n/sum(n))
kr1_freq = kr1_filtered %>% mutate(n1 = n) %>% mutate(f1 = n/sum(n)) 
kr2_freq = kr2_filtered %>% mutate(n2 = n) %>% mutate(f2 = n/sum(n)) 
kr3_freq = kr3_filtered %>% mutate(n3 = n) %>% mutate(f3 = n/sum(n)) 

k_enrich <- merge(klib_freq[,c("kozak_sequence","n0","f0")], kr1_freq[,c("kozak_sequence","n1","f1")], by = "kozak_sequence", all = T)
k_enrich <- merge(k_enrich, kr2_freq[,c("kozak_sequence","n2","f2")], by = "kozak_sequence", all = T)
k_enrich <- merge(k_enrich, kr3_freq[,c("kozak_sequence","n3","f3")], by = "kozak_sequence", all = T)


k_enrich[is.na(k_enrich)] <- 0
#k_enrich2 <- k_enrich %>% mutate(fcells = (f1 + f2 + f3)/3) %>% mutate(sdcells = sqrt(((f1-fcells)^2 + (f2-fcells)^2 + (f3-fcells)^2)/2)) %>% mutate(cvcells = sdcells / fcells)
k_enrich2 <- k_enrich %>% mutate(fcells = (f2 + f3)/3) %>% mutate(sdcells = sqrt(((f2-fcells)^2 + (f3-fcells)^2)/2)) %>% mutate(cvcells = sdcells / fcells)


#separate based on variants
k_enrich3 = k_enrich2 %>% mutate(code = substr(kozak_sequence, 1, 1))
k_enrich3 = k_enrich3 %>% mutate(variant = case_when(code == "A" ~ "WT",
                                           code == "C" ~ "R429C",
                                           code == "G" ~ "R304W"))

k_enrich3[k_enrich3$kozak_sequence == "_OTHER","variant"] <- "Other"
k_enrich3[k_enrich3$kozak_sequence == "_TEMPL","variant"] <- "Template"
```

``` r
i0710 <- read.csv(file = "Data/STIM1_Kozak/I0710.csv", header = T)
i0711 <- read.csv(file = "Data/STIM1_Kozak/I0711.csv", header = T)
r5t0 <- merge(i0710, i0711, by = "Sequence", all = T)
r5t0$kozak <- substr(r5t0$Sequence,1,6)
r5t0$coding <- substr(r5t0$Sequence,7,20)
r5t0 <- r5t0 %>% filter(coding == "ATGGATGTATGCG")

i0712 <- read.csv(file = "Data/STIM1_Kozak/I0712.csv", header = T)
i0713 <- read.csv(file = "Data/STIM1_Kozak/I0713.csv", header = T)
r5t1 <- merge(i0712, i0713, by = "Sequence", all = T)
r5t1$kozak <- substr(r5t1$Sequence,1,6)
r5t1$coding <- substr(r5t1$Sequence,7,20)
r5t1 <- r5t1 %>% filter(coding == "ATGGATGTATGCG")

i0714 <- read.csv(file = "Data/STIM1_Kozak/I0714.csv", header = T)
i0715 <- read.csv(file = "Data/STIM1_Kozak/I0715.csv", header = T)
r5t2 <- merge(i0714, i0715, by = "Sequence", all = T)
r5t2$kozak <- substr(r5t2$Sequence,1,6)
r5t2$coding <- substr(r5t2$Sequence,7,20)
r5t2 <- r5t2 %>% filter(coding == "ATGGATGTATGCG")

i0716 <- read.csv(file = "Data/STIM1_Kozak/I0716.csv", header = T)
i0717 <- read.csv(file = "Data/STIM1_Kozak/I0717.csv", header = T)
r6t0 <- merge(i0716, i0717, by = "Sequence", all = T)
r6t0$kozak <- substr(r6t0$Sequence,1,6)
r6t0$coding <- substr(r6t0$Sequence,7,20)
r6t0 <- r6t0 %>% filter(coding == "ATGGATGTATGCG")

i0718 <- read.csv(file = "Data/STIM1_Kozak/I0718.csv", header = T)
i0719 <- read.csv(file = "Data/STIM1_Kozak/I0719.csv", header = T)
r6t1 <- merge(i0718, i0719, by = "Sequence", all = T)
r6t1$kozak <- substr(r6t1$Sequence,1,6)
r6t1$coding <- substr(r6t1$Sequence,7,20)
r6t1 <- r6t1 %>% filter(coding == "ATGGATGTATGCG")

i0720 <- read.csv(file = "Data/STIM1_Kozak/I0720.csv", header = T)
i0721 <- read.csv(file = "Data/STIM1_Kozak/I0721.csv", header = T)
r6t2 <- merge(i0720, i0721, by = "Sequence", all = T)
r6t2$kozak <- substr(r6t2$Sequence,1,6)
r6t2$coding <- substr(r6t2$Sequence,7,20)
r6t2 <- r6t2 %>% filter(coding == "ATGGATGTATGCG")
```

``` r
#r5t0[is.na(r5t0)] <- 0.1
r5t0$freq1 <- r5t0$Count.x / sum(r5t0$Count.x, na.rm = T)
r5t0$freq2 <- r5t0$Count.y / sum(r5t0$Count.y, na.rm = T)
r5t0$r5t0 <- rowMeans(r5t0[c("freq1", "freq2")])

#r5t1[is.na(r5t1)] <- 0.1
r5t1$freq1 <- r5t1$Count.x / sum(r5t1$Count.x, na.rm = T)
r5t1$freq2 <- r5t1$Count.y / sum(r5t1$Count.y, na.rm = T)
r5t1$r5t1 <- rowMeans(r5t1[c("freq1", "freq2")])

r5 <- merge(r5t0[,c("kozak","r5t0")], r5t1[,c("kozak","r5t1")])

#r5t2[is.na(r5t2)] <- 0.1
r5t2$freq1 <- r5t2$Count.x / sum(r5t2$Count.x, na.rm = T)
r5t2$freq2 <- r5t2$Count.y / sum(r5t2$Count.y, na.rm = T)
r5t2$r5t2 <- rowMeans(r5t2[c("freq1", "freq2")])

r5 <- merge(r5, r5t2[,c("kozak","r5t2")])
r5$r5 <- (rowMeans(log10(r5[,c("r5t1","r5t2")]), na.rm = T) - log10(r5$r5t0))
colnames(r5)[1] <- "kozak_sequence"

#r6t0[is.na(r6t0)] <- 0.1
r6t0$freq1 <- r6t0$Count.x / sum(r6t0$Count.x, na.rm = T)
r6t0$freq2 <- r6t0$Count.y / sum(r6t0$Count.y, na.rm = T)
r6t0$r6t0 <- rowMeans(r6t0[c("freq1", "freq2")])

#r6t1[is.na(r6t1)] <- 0.1
r6t1$freq1 <- r6t1$Count.x / sum(r6t1$Count.x, na.rm = T)
r6t1$freq2 <- r6t1$Count.y / sum(r6t1$Count.y, na.rm = T)
r6t1$r6t1 <- rowMeans(r6t1[c("freq1", "freq2")])

r6 <- merge(r6t0[,c("kozak","r6t0")], r6t1[,c("kozak","r6t1")])

#r6t2[is.na(r6t2)] <- 0.1
r6t2$freq1 <- r6t2$Count.x / sum(r6t2$Count.x, na.rm = T)
r6t2$freq2 <- r6t2$Count.y / sum(r6t2$Count.y, na.rm = T)
r6t2$r6t2 <- rowMeans(r6t2[c("freq1", "freq2")])

r6 <- merge(r6, r6t2[,c("kozak","r6t2")])
r6$r6 <- (rowMeans(log10(r6[,c("r6t1","r6t2")]), na.rm = T) - log10(r6$r6t0))
colnames(r6)[1] <- "kozak_sequence"
```

``` r
k_enrich3$enrichment <- log10(k_enrich3$fcells / k_enrich3$f0)

k_score0 = k_enrich3 %>% select("kozak_sequence","variant","f0", "fcells","enrichment") %>% filter(!(enrichment %in% c("0","Inf")))

k_score <- merge(merge(k_score0, r5[,c("kozak_sequence","r5")], all = T), r6[,c("kozak_sequence","r6")], all = T)

k_score$survival <- rowMeans(k_score[,c("enrichment","r5","r6")], na.rm = T)

#abund_raw = read.csv("Kozak_values.csv")
abund_raw = complete_frame5[,c("sequence","calibrated_score")]
abund_raw = abund_raw %>% mutate(kozak_sequence = substr(sequence, 1, 6))
abund_stim = abund_raw %>% mutate(code = substr(kozak_sequence, 1,1)) %>% filter(code != "T")
k_score = left_join(k_score, abund_stim)

k_score_samples <- k_score %>% filter(variant %in% c("WT", "R304W","R429C") & !is.na(calibrated_score))

k_score_samples$variant <- factor(k_score_samples$variant, levels = c("WT", "R304W","R429C"))

STIM1_Kozak_histogram <- ggplot() + 
  labs(x = "Enrichment", y = "Number of variants") +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank(), strip.text.y.right = element_text(angle = 0)) + 
  scale_x_log10(limits = c(0.01,100), expand = c(0,0), breaks = c(0.01,1,100)) + 
  scale_y_continuous(breaks = c(0,30)) + 
  geom_vline(xintercept = 10^k_score$survival[2], linetype = 2) +
  geom_histogram(data = k_score_samples, aes(x= 10^survival), bins = 10, color = "black", fill = "grey75") + 
  facet_grid(rows = vars(variant), scales = "free_y") +
  NULL; STIM1_Kozak_histogram
```

![](Kozak_files/figure-gfm/Calculate%20enrichment%20for%20STIM1%20Kozak%20data-1.png)<!-- -->

``` r
ggsave(file = "Plots/STIM1_Kozak_histogram.pdf", STIM1_Kozak_histogram, height = 1.5, width = 1.9)
```

``` r
k_score_samples$variant <- factor(k_score_samples$variant, levels = c("R304W","R429C","WT"))

STIM1_geom_smooth_plot <- ggplot() + theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank()) + 
  scale_x_log10(expand = c(0,0), limits = c(0.007,1.34)) + scale_y_log10(expand = c(0,0)) + 
  labs(x = "Calibrated abundance score", y = "Enrichment score") +
  geom_hline(yintercept = 10^k_score$survival[2], alpha = 0.4) +
  geom_smooth(data = k_score_samples, aes(x= calibrated_score, y= 10^survival, color = variant), alpha = 0.1, span = 1.5) +
  NULL; STIM1_geom_smooth_plot
```

![](Kozak_files/figure-gfm/Plotting%20the%20variant-specific%20STIM1%20functional%20outputs%20dependent%20on%20expression%20level-1.png)<!-- -->

``` r
#ggsave(file = "Plots/STIM1_geom_smooth_plot.pdf", STIM1_geom_smooth_plot, height = 1.5, width = 2.9)
```
