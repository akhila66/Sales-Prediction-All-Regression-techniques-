library(tidyverse) 
library(lubridate)
library(caret)
require(MASS)
library(pls)
library(mltools)
library(elasticnet)
library('R2')
library(earth)
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

dfTrain <- read_csv("~/Desktop/SEM2/IDE/HW6/Train.csv")
dfTest <- read_csv("~/Desktop/SEM2/IDE/HW6/Test.csv")

#get train and test customer ids so that I can keep track of which customers belong in which data set..

trainIds <- unique(dfTrain$custId)  
testIds <- unique(dfTest$custId)    

#combine the test and train data 
fullData <- dfTrain %>% dplyr::select(-revenue) %>% bind_rows(dfTest)

#get earliest and latest dates in data 
#-- since the test and train data come from the same time period, I can just do this together
tf_maxdate = max(ymd(fullData$date))
tf_mindate = min(ymd(fullData$date))


#lots of transforms, etc.
#use of lubridate package to change dates to R dates
#use of forcats package to lump categorical variables
#grouping by customer id
#note: this summarizatoin is by no means the smartest bit of summarization, but it works...

tfull <- fullData %>% 
  mutate(date=ymd(date)) %>%                                              
  mutate(country = fct_lump(fct_explicit_na(country), n = 11)) %>%         
  mutate(subContinent = fct_lump(fct_explicit_na(subContinent), n = 10)) %>%
  mutate(region = fct_lump(fct_explicit_na(region), n = 8)) %>%
  mutate(networkDomain = fct_lump(fct_explicit_na(networkDomain), n = 8)) %>%
  mutate(source= fct_lump(fct_explicit_na(source), n = 5)) %>%
  mutate(medium = fct_lump(fct_explicit_na(medium), n = 5)) %>%
  mutate(browser = fct_lump(fct_explicit_na(browser), n = 4)) %>%
  group_by(custId) %>%                                   
  summarize(                                            
    channelGrouping = max(ifelse(is.na(channelGrouping) == TRUE, -9999, channelGrouping)),
    first_ses_from_the_period_start = min(date) - tf_mindate,
    last_ses_from_the_period_end = tf_maxdate - max(date),
    interval_dates = max(date) - min(date),
    unique_date_num = length(unique(date)),
    maxVisitNum = max(visitNumber, na.rm = TRUE),
    browser = first(browser),
    operatingSystem = first(operatingSystem),
    deviceCategory = first(deviceCategory),
    subContinent = first(subContinent),
    country = first(country),
    region = first(region),
    networkDomain = first(networkDomain),
    source = first(source),
    medium = first(medium),
    isVideoAd_mean = mean(ifelse(is.na(adwordsClickInfo.isVideoAd) == TRUE, 0, 1)),
    isMobile = mean(ifelse(isMobile == TRUE, 1 , 0)),
    isTrueDirect = mean(ifelse(is.na(isTrueDirect) == TRUE, 0, 1)),
    bounce_sessions = sum(ifelse(is.na(bounces) == TRUE, 0, 1)),
    pageviews_sum = sum(pageviews, na.rm = TRUE),
    pageviews_mean = mean(ifelse(is.na(pageviews), 0, pageviews)),
    pageviews_min = min(ifelse(is.na(pageviews), 0, pageviews)),
    pageviews_max = max(ifelse(is.na(pageviews), 0, pageviews)),
    pageviews_median = median(ifelse(is.na(pageviews), 0, pageviews)),
    session_cnt = NROW(visitStartTime)
  )

tfull$operatingSystem<- fct_explicit_na(tfull$operatingSystem, na_level = Mode(tfull$operatingSystem))

tfull$channelGrouping <- as.factor(tfull$channelGrouping)
tfull$deviceCategory <- as.factor(tfull$deviceCategory)




#get the transformed train data
tf<-tfull %>% filter(custId %in% trainIds)

#compute the target variable
outcome<-dfTrain %>% 
  group_by(custId) %>%
  summarize(
    transactionRevenue = sum(revenue)
  ) %>% 
  mutate(logSumRevenue = log(transactionRevenue+1)) %>% dplyr::select(-transactionRevenue)

#join the transformed target variable to the aggregated training data
tf <- tf %>% inner_join(outcome, by = "custId")

#separate out the test data
tstf <-tfull %>% filter(custId %in% testIds)

check <- tf %>% mutate_all(is.na) %>% summarise_all(mean)

# --------------------------------------------------------- LM ------------------------------

mdl01<-lm(formula = logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
            as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
            log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
            country + region + networkDomain + source + 
            log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
            log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
            pageviews_median + log(session_cnt), 
          data = tf)

summary(mdl01)
model.summary = summary(mdl01)$coefficients
model.summary
summary(mdl01)$r.squared # more is better

res <- predict(mdl01, tf)
rmse((tf$logSumRevenue),res)
# cross validation
data_ctrl <- trainControl(method = "cv", number = 5)
model_caret <- train(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
                       as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
                       log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
                       country + region + networkDomain + source + 
                       log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
                       log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
                       pageviews_median + log(session_cnt),   # model to fit
                     data = tf,                        
                     trControl = data_ctrl,              # folds
                     method = "lm",                      # specifying regression model
                     na.action = na.pass)                # pass missing data to model - some models will handle this

model_caret
model_caret$finalModel
model_caret$resample
model_caret$finalModel$coefficients


# -------- output file
res <- predict(mdl01, tstf)
predicted<-data.frame(custId = tstf$custId,predRevenue = res)
predicted %>% select_if(is.numeric) %>% mutate_all(is.na) %>% summarise_all(mean)
#predicted[is.na(predicted)] <- 0
write.csv(predicted, file = paste("~/Desktop/SEM2/IDE/HW5/submit",Sys.time(),".csv"), row.names=FALSE)

# ----------------------------------------- RLM (warnings not working)---------------

tf %>% select_if(is.numeric) %>% mutate_all(is.na) %>% summarise_all(mean)
summary(fitrlm <- rlm(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
                        as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
                        log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
                        country + region + networkDomain + source + 
                        log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
                        log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
                        pageviews_median + log(session_cnt), data = tf))

summary(fitrlm)
model.summary1 = summary(fitrlm)$coefficients
model.summary1
summary(fitrlm)$r.squared # more is better

res1 <- predict(fitrlm, tf)
rmse((tf$logSumRevenue),res1)

data_ctrl <- trainControl(method = "cv", number = 5)
model_caret1 <- train(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
                       as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
                       log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
                       country + region + networkDomain + source + 
                       log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
                       log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
                       pageviews_median + log(session_cnt),   # model to fit
                     data = tf,                        
                     trControl = data_ctrl,              # folds
                     method = "rlm",                      # specifying regression model
                     )                # pass missing data to model - some models will handle this

model_caret1
model_caret1$finalModel
model_caret1$resample
model_caret1$finalModel$coefficients

# --------------------------  PLS -------------

summary(plsFit <- plsr(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
                as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
                log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
                country + region + networkDomain + source + 
                log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
                log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
                pageviews_median + log(session_cnt), data = tf))
summary(plsFit)
model.summary2 = summary(plsFit)$coefficients


res2 <- predict(plsFit, tf)
rmse((tf$logSumRevenue),res2)

# cross validation
data_ctrl <- trainControl(method = "cv", number = 5)
plscvmodel <- train(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
                        as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
                        log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
                        country + region + networkDomain + source + 
                        log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
                        log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
                        pageviews_median + log(session_cnt),   # model to fit
                      data = tf,    
                      tuneLength = 10,
                      trControl = data_ctrl,              # folds
                      method = "pls",                      # specifying regression model
)                # pass missing data to model - some models will handle this

plscvmodel
plscvmodel$finalModel
plscvmodel$resample
plscvmodel$finalModel$coefficients


# ------------------------  ridge , lasso and elastic net regression / elastic net

tf_cat <- tf[ ,sapply(tf, is.factor)]
catcols <- colnames(tf_cat)
for(i in catcols)
{
  print(i)
  #tf_cat[i] <- str_replace_all(tf_cat[,i], "[^[:alnum:]]", " ") # to remove special char
  d <- dummyVars(" ~ .", data = tf_cat[i])
  onehot <- data.frame(predict(d, newdata = tf_cat[i]))
  tf_cat <- cbind(tf_cat,onehot)
}
tf_cat <- tf_cat[ , !(names(tf_cat) %in% catcols)]
tf_cat_num<-cbind(tf_cat,tf[ ,!sapply(tf, is.factor)])        # complete data.frame with all variables put together



#http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/153-penalized-regression-essentials-ridge-lasso-elastic-net/#ridge-regression
#https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
#https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
library(glmnet)  

#Computing ridge regression
x <- model.matrix(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
                    as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
                    log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
                    country + region + networkDomain + source + 
                    log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
                    log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
                    pageviews_median + log(session_cnt),tf)[,-1]
y <- tf$logSumRevenue
lambda <- seq(0.01,0.1 ,length = 100)
# Build the model
ridgeall <- glmnet(x, y, alpha = 0, lambda = lambda, standardize = FALSE)
plot(ridgeall, xvar = 'lambda')


set.seed(123)
# Find the best lambda using cross-validation
ridge <- train(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
    as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
    log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
    country + region + networkDomain + source + 
    log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
    log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
    pageviews_median + log(session_cnt),   # model to fit
  data = tf, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda)
)
plot(ridge)
ridge$bestTune$lambda
# Model coefficients
coef(ridge$finalModel, ridge$bestTune$lambda)
plot(ridge$finalModel, xvar = 'lambda')
plot(ridge$finalModel, xvar = 'dev')
plot(caret::varImp(ridge,scale = T))
# Make predictions
predictions <- ridge %>% predict(tf)
# Model prediction performance
data.frame(
  RMSE = RMSE(predictions, tf$logSumRevenue),
  Rsquare = caret::R2(predictions, tf$logSumRevenue)
)
ridge$resample
coef(ridge$finalModel, ridge$bestTune$lambda)



#Computing lasso regression
lassoall <- glmnet(x, y, alpha = 0, lambda = lambda, standardize = FALSE)
plot(lassoall, xvar = 'lambda')
# Find the best lambda using cross-validation
set.seed(123) 
lasso <- train(
  logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
    as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
    log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
    country + region + networkDomain + source + 
    log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
    log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
    pageviews_median + log(session_cnt),   # model to fit
  data = tf, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda)
)

plot(lasso)
# Model coefficients
lasso$bestTune$lambda
coef(lasso$finalModel, lasso$bestTune$lambda)
plot(lasso$finalModel, xvar = 'lambda')
# Make predictions
predictions <- lasso %>% predict(tf)
# Model prediction performance
data.frame(
  RMSE = RMSE(predictions, tf$logSumRevenue),
  Rsquare = caret::R2(predictions, tf$logSumRevenue)
)
lasso$resample



#Computing elastic net regession
# Build the model using the training set
set.seed(123)
enetmodel <- train(
  logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
    as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
    log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
    country + region + networkDomain + source + 
    log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
    log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
    pageviews_median + log(session_cnt),   # model to fit
    data = tf, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = seq(0,1,length=10), lambda = lambda))
# Best tuning parameter
plot(enetmodel)
enetmodel$bestTune
plot(enetmodel$finalModel)
coef(enetmodel$finalModel, enetmodel$bestTune$lambda)
# Make predictions on the test data
predictions <- enetmodel %>% predict(tf) 
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, tf$logSumRevenue),
  Rsquare = caret::R2(predictions, tf$logSumRevenue)
)
enetmodel$resample

# ------------------------  MASS
#http://uc-r.github.io/mars
mars1 <- earth(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
                 as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
                 log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
                 country + region + networkDomain + source + 
                 log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
                 log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
                 pageviews_median + log(session_cnt),  
                 data = tf   
)

summary(mars1) %>% .$coefficients
plot(mars1, which = 1)


mars2 <- earth(logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
                 as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
                 log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
                 country + region + networkDomain + source + 
                 log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
                 log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
                 pageviews_median + log(session_cnt),
        data = tf,
        degree = 2
)
mars2
mars2$rsq
# check out the coefficient terms
summary(mars2) %>% .$coefficients
plot(mars2, which = 1)

# for reproducibiity
set.seed(123)
hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)
# cross validated model
set.seed(123)
tuned_mars <- train(
   logSumRevenue ~ channelGrouping + as.numeric(first_ses_from_the_period_start) + 
    as.numeric(last_ses_from_the_period_end) +  log(unique_date_num+1) + 
    log(maxVisitNum+1) + browser + operatingSystem + deviceCategory + 
    country + region + networkDomain + source + 
    log(bounce_sessions+1) + bounce_sessions*pageviews_sum +
    log(pageviews_sum+1) + log(pageviews_mean+1) + pageviews_min + 
    pageviews_median + log(session_cnt),
  data = tf,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 3),
  tuneGrid = hyper_grid
)

# best model
tuned_mars$bestTune
summary(tuned_mars)
tuned_mars$resample
tuned_mars$finalModel$coefficients



ptuned_marsres<-tibble(custId=tstf$custId, predRevenue=predict(tuned_mars,newdata=tstf))
head(ptuned_marsres)
ptuned_marsres %>% select_if(is.numeric) %>% mutate_all(is.na) %>% summarise_all(mean)
ptuned_marsres <-ptuned_marsres %>% mutate(predRevenue = replace(predRevenue, which(predRevenue<0), 0))

#predicted[is.na(predicted)] <- 0
write.csv(ptuned_marsres, file = paste("~/Desktop/SEM2/IDE/HW6/submitMars",Sys.time(),".csv"), row.names=FALSE)



# -----------------------------------------------
# cleaning
rm(list = ls(all.names = TRUE))
gc()

