#Project: Comparing drugbank vs C50 (e.g. c50Drugs) scores
#by Jeremy Hunt
#3 October 2017
# modified by Aaron 15 Feb 2018

library(tidyverse)
library(stringr)

###############################################################################################3
# Set defaults
# these numbers were derived from previous optimizations with ChEMBL and BindingDB
# NOTE: These values should match values in web1/drugs/drug_edit.py
c50_for_hi <- 200
c50_for_lo <- 630
hi_evid <- 0.9
lo_evid <- 0.5

ki_for_hi <- 40
ki_for_lo <- 320
ki_hi_evid <- 0.7
ki_lo_evid <- 0.4
###############################################################################################3

log10_plus1 <- function(num){
    return(log10(num+1))
}

convert_lower_confidence <- function(value, min_val, max_val, min_evid=0.5, max_evid=0.9){
    return(linear_confidence_conversion(value, min_val, max_val, min_evid,max_evid))
}
convert_high_confidence <- function(value, min_val, max_val=0, min_evid=0.9, max_evid=1){
    return(linear_confidence_conversion(value, min_val, max_val, min_evid,max_evid))
}

linear_confidence_conversion <- function(value, min_val, max_val, min_evid,max_evid){
    evid_delta <- max_evid-min_evid
    val_delta <- max_val-min_val
    portion_val_delta <- (value-min_val)/val_delta
    return(round(min_evid + portion_val_delta*evid_delta, 4))
}

process_tm <- function(threshold_model){
    threshold_model %>% summarize(accuracy = sum(match) / n())
    accuracy_table <- threshold_model %>% group_by(interaction, prediction) %>% count()
    total <- sum(accuracy_table$n)
    TT <- accuracy_table[which(
               accuracy_table$prediction == T &
               accuracy_table$interaction == T),
               'n']
    TF <- accuracy_table[which(
               accuracy_table$prediction == F &
               accuracy_table$interaction == T),
               'n']
    FT <- accuracy_table[which(
               accuracy_table$prediction == T &
               accuracy_table$interaction == F),
               'n']
    FF <- accuracy_table[which(
               accuracy_table$prediction == F &
               accuracy_table$interaction == F),
               'n']

    FP_frac <- FT/total
    TP_frac <- TT/total
    FN_frac <- TF/total
    TN_frac <- FF/total
    ba <- (TT/(TT+TF) + FF/(FF+FT))/2
    fr <- c(TP_frac, TN_frac, FP_frac, FN_frac)
    return(list(ba = ba, fr = fr))
}

plot_results <- function(balancedAccs,
                         log10_c50plus1_thresholds,
                         drugbank_thresh,
                         dont_show_existing_threshold,
                         existing_threshold,
                         perc = 1.0
                         ){
    balancedAccs <- unlist(balancedAccs)
### rather than going with max I'll go with
### the first instance that reaches the Xth percentile.
### That's done to give more control.
### Putting perc = 1.0 is the same as max
    best_balancedAcc <- quantile(balancedAccs, perc)
    thresh <- log10_c50plus1_thresholds[which(balancedAccs>=best_balancedAcc)[1]]
    plot(log10_c50plus1_thresholds,
         balancedAccs,
         type = 'l',
         main = paste('Balanced accuracy of drugbank confidence from c50Drugs data.\n',
                      'Threshold =', drugbank_thresh
                      )
    )
    abline(v = thresh, col = 'red')
    legend('topright',
           paste(thresh,
                 round(best_balancedAcc, 4),
                 sep = ': '
                ),
           text.col='red'
          )
    if (! dont_show_existing_threshold){
        abline(v = existing_threshold, col = 'blue')
        legend('topleft',
               paste(round(existing_threshold, 3),
                     round(balancedAccs[which(log10_c50plus1_thresholds==existing_threshold)],
                           4),
                     sep = ': '
                    ),
               text.col='blue'
              )
    }
    return(thresh)
}


# Load and combine data {{{ --------------------------------------------------------------------


args <- commandArgs(trailingOnly = TRUE)
c50Drugs <- read_tsv(args[1]) %>%
                   rename(c50Drugs_direction = direction) %>%
                   select(c50Drugs_id = !! quo(names(.)[1]), everything())
drug_collection <- strsplit(args[1], ".", fixed = TRUE)[[1]][2]
# the evidence column is imported as a string instead of numeric
# to make ggplot graph by catagories
# (makes violin and box plotting easier)
drugbank <- read_tsv(args[2], col_types = 'ccci')
c50Drugs_to_db <- read_tsv(args[3], col_names = c('c50Drugs_id', 'drugbank_id'))
dpi_type <- args[4]
if (dpi_type != 'ki' && dpi_type != 'c50'){
    print('WARNING: Unrecognized DPI type. Quitting.')
    quit()
}else if(dpi_type == 'ki'){
    colnames(c50Drugs)[grep('Ki_final', colnames(c50Drugs))] <- 'C50_final'
    c50_for_hi <- ki_for_hi
    c50_for_lo <- ki_for_lo
    hi_evid <- ki_hi_evid
    lo_evid <- ki_lo_evid
}# else it's c50

optimize <- FALSE
if (length(args) > 4){
    optimize <- as.logical(args[5])
}
if (optimize){
    print( paste(
        "WARNING:",
        dpi_type,
        "to evidence values are being determined empircally.",
        "This can cause consistency issues and is no longer suggested."
    ))
}else{
    print( paste(
        "Preset thresholds optimized from previous experiments and databases are being used.",
        "The threshold for high evidence, a value of",
        hi_evid,
        ", is:",
        c50_for_hi,
        "nM, and for low evidence (",
        lo_evid,
        "):",
        c50_for_lo,
        "nM. Note that all plots will be generated for completeness, but the thresholds will be used."
    ))
}
logc50_for_hi <- log10_plus1(c50_for_hi)
logc50_for_lo <- log10_plus1(c50_for_lo)

pdf(paste0(drug_collection, '_',dpi_type,'Drugs_parsing_plots.pdf'))
#check redundancy:
print( paste( 
    nrow(c50Drugs) - nrow(distinct(c50Drugs, c50Drugs_id, uniprot_id)),
    'redudant DPI in',
    drug_collection
))
print( paste( 
    nrow(drugbank) - nrow(distinct(drugbank, drugbank_id, uniprot_id)),
   'redudant DPI in drugbank'
))

#drug intersection of databases
db_d_inter <- drugbank %>% inner_join(c50Drugs_to_db, by = c('drugbank_id')) 
c_d_inter <- c50Drugs %>% inner_join(c50Drugs_to_db, by = 'c50Drugs_id') 

# drug-protein pair intersection of databases (2 objects should have same info)
db_dp_inter <- db_d_inter %>% inner_join(c50Drugs, by = c('c50Drugs_id','uniprot_id'))
c_dp_inter <- c_d_inter%>% inner_join(drugbank, by = c('drugbank_id', 'uniprot_id'))
#check that joined correctly
paste('these numbers should match:', nrow(db_dp_inter), nrow(c_dp_inter))

# drug_collection dpi that match a drug in drugbank (d_m),
# but no match for full drug protein pair (dp_nm)
dp_nm_d_m <- c_d_inter %>% anti_join(c_dp_inter, by = c('c50Drugs_id', 'uniprot_id'))

# drug_collection drug intersection with drugbank, including DPI info from drugbank
c_d_inter_all_data <- dp_nm_d_m %>%
                      mutate(evidence = '0', direction = NA) %>%
                      bind_rows(c_dp_inter)

# gets rid of 9 points (.4 and .7 hardly have any points)
cdi_clean <- c_d_inter_all_data %>% filter(evidence != '0.4', evidence != '0.7')
cdi_clean_noNA <- cdi_clean
cdi_clean_noNA[is.na(cdi_clean_noNA)] <- 0
#cdi_clean_noNA %>% write_csv(paste0(drug_collection,'_drugbank_drug_intersection_cleaned.csv'))
# }}}

# plots {{{ ---------------------------------------------------------------------------------

# explore if drug_collection and db even have the same directions when we are comparing them!
print('drugbank')
drugbank %>% count(direction) # most of drugbank has 0, an unknown direction.
print(paste(drug_collection,'in drugbank'))
cdi_clean %>% count(c50Drugs_direction)
# chebl mostly has IC50, because it is easier to break proteins than help
print(paste('all', drug_collection))
c50Drugs %>% count(c50Drugs_direction)

#see if the directions match
cdi_dir <- cdi_clean %>% mutate(direction_match = c50Drugs_direction == direction)
print('direction match across all DPI')
cdi_dir %>% filter(!is.na(direction_match)) %>% summarize(frac_match = sum(direction_match)/n())
# this asks if ANY measurement of a drug-protien pair
# as a different direction as listed in drugbank
print('direction match across DPI where drugbank is 0.9 and has direction')
cdi_dir %>% filter(evidence == '0.9', direction !=0) %>% 
    summarize(fracion_matched_direction = sum(direction_match)/n())
# this asks if any ONE measurement of a drug-protien pair has the SAME direction
# as listed in drugbank
print(paste('same as above, but deals with multiple drugbanks mapping to a single', drug_collection))
temp <- cdi_dir %>%
    select(c50Drugs_id, uniprot_id, drugbank_id, direction, c50Drugs_direction, direction_match)%>%
    filter(!is.na(direction)) %>% # keep only entries in drugbank
    filter(direction != 0) # keep only drugbank entries that have a known direction
temp[is.na(temp)] <- 0
temp %>% summarize(fraction_matched_direction = sum(direction_match)/n())


#plot just IC50 of intersection set and see if better overlap.
cdi_dir %>% mutate(x_axis = paste0(evidence, direction_match)) %>%
    ggplot(aes(x_axis, log10_plus1(C50_final))) + geom_violin()
cdi_dir %>% mutate(x_axis = paste0(evidence, direction_match, c50Drugs_direction)) %>%
    ggplot(aes(x_axis, log10_plus1(C50_final))) + geom_violin()

# plot to explore effect of drugbank direction on log10(C50+1) data from drug_collection
cdi_clean %>% mutate(evidence_dir = paste0(evidence,'_', direction)) %>%
    ggplot(aes(evidence_dir, log10_plus1(C50_final))) + geom_violin()

# plot to explore effect of drug_collection direction on log10(C50+1) data from drug_collection   
cdi_clean %>% ggplot(aes(c50Drugs_direction, log10_plus1(C50_final))) + geom_violin()
cdi_clean %>% ggplot(aes(c50Drugs_direction, log10_plus1(C50_final))) + geom_violin(scale = 'count')

c50Drugs %>% ggplot(aes(c50Drugs_direction, log10_plus1(C50_final))) + geom_violin()
c50Drugs %>% ggplot(aes(c50Drugs_direction, log10_plus1(C50_final))) + geom_violin(scale = 'count')

cdi_clean %>% ggplot(aes(evidence, log10_plus1(C50_final))) +
  geom_violin(scale = 'area') +
  ggtitle(paste('zero point is', drug_collection, 'data with drug match but no dpi match in drugbank'))

c_d_inter_all_data %>% ggplot(aes(evidence, log10_plus1(C50_final))) +
  geom_violin(scale = 'count') +
  ggtitle(paste('zero point is', drug_collection, 'data with drug match but no dpi match in drugbank'))

c_d_inter_all_data %>% ggplot(aes(log10_plus1(C50_final), color = evidence)) + geom_freqpoly(bins = 20)
c_d_inter_all_data %>% filter(evidence != .4, evidence != .7) %>%
  ggplot(aes(log10_plus1(C50_final), ..density.., color = evidence)) + geom_freqpoly(bins = 20)

#graph zero data as all drug_collection data except dpi matches in drugbank
c_dp_nm <- c50Drugs %>% anti_join(c_dp_inter, by = c('c50Drugs_id','uniprot_id'))

all_data <- c_dp_nm %>% mutate(evidence = '0', direction = NA) %>% bind_rows(c_dp_inter)

all_data %>% ggplot(aes(evidence, log10_plus1(C50_final))) +
  geom_violin(scale = 'area') +
  ggtitle(paste('zero point is', drug_collection, 'all', drug_collection, 'data except DPI in drugbank'))


log10_c50plus1_thresholds <- sort(c(seq(0.1, 4, 0.01), logc50_for_hi, logc50_for_lo))
drugbank_threshs <- c(0.9, 0.5)

# run this 2x
drugbank_thresh <- drugbank_threshs[1]
balancedAccs <- log10_c50plus1_thresholds # To be set
full_results <- list()
for (i in 1:length(log10_c50plus1_thresholds)){
    threshold_model <- c_d_inter_all_data %>% mutate(interaction = (
                                                 (evidence>= drugbank_thresh)
                                                 ),
                              prediction = log10_plus1(C50_final) <= log10_c50plus1_thresholds[i],
                              match = interaction == prediction)
    temp <- process_tm(threshold_model)
    balancedAccs[i] <- temp$ba
    full_results[[i]] <- temp$fr
}

temp <- plot_results(balancedAccs, log10_c50plus1_thresholds, drugbank_thresh, optimize, logc50_for_hi)
if (optimize){
    logc50_for_hi <- temp
}

# re-run, but tweak the evidence thresholding
drugbank_thresh <- drugbank_threshs[2]
balancedAccs <- log10_c50plus1_thresholds # To be set
full_results <- list()
for (i in 1:length(log10_c50plus1_thresholds)){
    threshold_model <- c_d_inter_all_data %>% mutate(interaction = (
                                                 (evidence>= drugbank_thresh &
                                                  evidence < drugbank_threshs[1]
                                                 )
                                                 ),
                              prediction = log10_plus1(C50_final) <= log10_c50plus1_thresholds[i],
                              match = interaction == prediction)
    temp <- process_tm(threshold_model)
    balancedAccs[i] <- temp$ba
    full_results[[i]] <- temp$fr
}
temp <- plot_results(balancedAccs, log10_c50plus1_thresholds, drugbank_thresh, optimize, logc50_for_lo)

if (optimize){
    logc50_for_lo <- temp
}


# }}}
dev.off()
# convert {{{ ---------------------------------------------------------------------------------
# and convert the C50 to confidence values optimized using drugbank
final_id <- paste0(drug_collection, '_id')

### This is where we put in the evidence scores from C50 values
c50Drugs_dir_subset <- c50Drugs %>% rename(direction = c50Drugs_direction) %>%
                     mutate(log10C50 = log10_plus1(C50_final)) %>%
                     mutate(evidence = ifelse(log10C50 <= logc50_for_hi,
# We tried this approach, but it noticeably harmed scores.
# so the logic remains for a later date, but it's purposefully disabled
#                                              convert_high_confidence(C50_final, c50_for_hi),
                                              hi_evid,
                                              ifelse(log10C50 <= logc50_for_lo,
#                                                     convert_lower_confidence(C50_final, c50_for_lo, c50_for_hi),
                                                     lo_evid,
                                                     ifelse(log10C50 > logc50_for_lo,
                                                            0,
                                                            NA
                                                           )
                                                    )
                                             )
                           ) %>%
                     filter(evidence > 0) %>%
                     select(c50Drugs_id, uniprot_id, evidence, direction) %>%
                     rename(!!final_id:=c50Drugs_id)
c50Drugs_dir_subset %>% write_tsv(paste0('dpi.',drug_collection, '.drugbankOptimd.tsv'),col_names=FALSE)

# }}}
