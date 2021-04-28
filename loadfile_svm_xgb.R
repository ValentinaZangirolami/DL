#Load packages
library(tensorflow)
library(keras)
library(e1071)
library(funModeling)
library(caret)
library(xgboost)

#seed
seed=123
set.seed(seed)
set_random_seed(123)

#Load training and test set
base_dir <- "~/archive/Alzheimer_s Dataset"
train <- file.path(base_dir, "train")
train_mild <- file.path(train, "MildDemented")
train_moderate <- file.path(train, "ModerateDemented")
train_non <- file.path(train, "NonDemented")
train_very <- file.path(train, "VeryMildDemented")

test <- file.path(base_dir, "test")
test_mild <- file.path(test, "MildDemented")
test_moderate <- file.path(test, "ModerateDemented")
test_non <- file.path(test, "NonDemented")
test_very <- file.path(test, "VeryMildDemented")

#Creating folders for the future partition in train/val/test
original_dataset_dir <- "~/archive/Alzheimer_s Dataset"
base_dir <- "~/Statisca_learning/progetto"
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_mild_dir <- file.path(train_dir, "MildDemented")
dir.create(train_mild_dir)
train_non_dir <- file.path(train_dir, "NonDemented")
dir.create(train_non_dir)
train_moderate_dir <- file.path(train_dir, "ModerateDemented")
dir.create(train_moderate_dir)
train_very_dir <- file.path(train_dir, "VeryMildDemented")
dir.create(train_very_dir)
validation_mild_dir <- file.path(validation_dir, "MildDemented")
dir.create(validation_mild_dir)
validation_non_dir <- file.path(validation_dir, "NonDemented")
dir.create(validation_non_dir)
validation_moderate_dir <- file.path(validation_dir, "ModerateDemented")
dir.create(validation_moderate_dir)
validation_very_dir <- file.path(validation_dir, "VeryMildDemented")
dir.create(validation_very_dir)
test_mild_dir <- file.path(test_dir, "MildDemented")
dir.create(test_mild_dir)
test_non_dir <- file.path(test_dir, "NonDemented")
dir.create(test_non_dir)
test_moderate_dir <- file.path(test_dir, "ModerateDemented")
dir.create(test_moderate_dir)
test_very_dir <- file.path(test_dir, "VeryMildDemented")
dir.create(test_very_dir)

#save directory for all images
fnames <- list.files(train_mild, full.names = TRUE)
fnames1 <- list.files(train_moderate, full.names = TRUE)
fnames2 <- list.files(train_non, full.names = TRUE)
fnames3 <- list.files(train_very, full.names = TRUE)

fname <- list.files(test_mild, full.names = TRUE)
fname1 <- list.files(test_moderate, full.names = TRUE)
fname2 <- list.files(test_non, full.names = TRUE)
fname3 <- list.files(test_very, full.names = TRUE)

#split train in train/val
ids <- sample(1:length(fnames), round(length(fnames)*.67), replace=FALSE)
file.copy(file.path(fnames[ids]), file.path(train_mild_dir))
file.copy(file.path(fnames[-ids]), file.path(validation_mild_dir))

ids <- sample(1:length(fnames1), round(length(fnames1)*.67), replace=FALSE)
file.copy(file.path(fnames1[ids]), file.path(train_moderate_dir))
file.copy(file.path(fnames1[-ids]), file.path(validation_moderate_dir))

ids <- sample(1:length(fnames2), round(length(fnames2)*.67), replace=FALSE)
file.copy(file.path(fnames2[ids]), file.path(train_non_dir))
file.copy(file.path(fnames2[-ids]), file.path(validation_non_dir))

ids <- sample(1:length(fnames3), round(length(fnames3)*.67), replace=FALSE)
file.copy(file.path(fnames3[ids]), file.path(train_very_dir))
file.copy(file.path(fnames3[-ids]), file.path(validation_very_dir))

#test
file.copy(file.path(fname), file.path(test_mild_dir))
file.copy(file.path(fname1), file.path(test_moderate_dir))
file.copy(file.path(fname2), file.path(test_non_dir))
file.copy(file.path(fname3), file.path(test_very_dir))

#preprocessing
datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  directory= train_dir,
  target_size = c(224,224),
  color_mode = 'rgb',
  class_mode = 'categorical',
  generator = datagen,
  batch_size = 32,
  shuffle = TRUE,
  seed=1
)
validation_generator <- flow_images_from_directory(
  directory= validation_dir,
  target_size = c(224,224),
  color_mode = 'rgb',
  class_mode = 'categorical',
  shuffle = TRUE,
  generator = datagen,
  seed=1)

test_generator <- flow_images_from_directory(
  directory= test_dir,
  target_size = c(224,224),
  color_mode = 'rgb',
  class_mode = 'categorical',
  shuffle = TRUE,
  generator = datagen,
  seed=1)

#load vgg16
base_model<- application_vgg16(input_shape = c(224, 224, 3), weights = 'imagenet', include_top = FALSE, pooling = 'avg')
base_model

#feature extraction
model_fit<- keras_model(inputs = base_model$input, outputs = base_model$output)
feature_train <- model_fit$predict(train_generator)
feature_val<- model_fit$predict(validation_generator)
feature_test<- model_fit$predict(test_generator)

#labels
train_labels<- as.factor(train_generator$classes)
val_labels<- as.factor(validation_generator$classes)
test_labels<- as.factor(test_generator$classes)
barplot(prop.table(table(train_labels))) #unbalanced data

#Follow lines include a vector of weights for each label. It is used to determine the cost of misclassification.
#It is similar to compute_class_weight (python- sklearn)

counter=freq(train_generator$classes, plot=F) 
majority=max(counter$frequency)
counter$weight=ceil(majority/counter$frequency)
l_weights=setNames(as.list(counter$weight), counter$var)

#dataframe of train, validation and test
train_over<- data.frame(train_labels, feature_train)
val_over<- data.frame(val_labels, feature_val)
test_over<- data.frame(test_labels, feature_test)

#best tuning of svm
svm_fit<- svm(train_labels~.,train_over,class_weight=l_weights,kernel = 'radial', type='C-classification',cost=1,gamma=1)

y_pred_train<- predict(svm_fit)
confusionMatrix(train_labels, y_pred_train)
y_pred_val<- predict(svm_fit, newdata = feature_val)
confusionMatrix(val_labels, y_pred_val)
y_pred_test <- predict(svm_fit, newdata = feature_test)
confusionMatrix(test_labels, y_pred_test)

#xgb
#xgb require weights for each line of dataset
pesi=c(rep(4,times=480),rep(49,times=35),rep(1,times=1715),rep(2,times=1201))
#transform dataset into xgb.DMatrix
t_label<- as.numeric(train_labels)
val_label<-as.numeric(val_labels)
te_label<-as.numeric(test_labels)
dtrain <- xgb.DMatrix(data = feature_train, label= t_label,weight=pesi)
dval <- xgb.DMatrix(data = feature_val, label= val_label)
dtest <- xgb.DMatrix(data = feature_test, label= te_label)

#best tuning of xgb
xgb_params <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = 5)
model<- xgb.train(data=dtrain,eta=0.2, params = xgb_params, nrounds = 200, early_stopping_rounds = 6,watchlist=list(val1=dtrain,val2=dval))

xgb.pred = predict(model,dtrain,reshape=T)
result = sum(xgb.pred==t_label)/length(xgb.pred)
print(paste("Train Accuracy =",sprintf("%1.2f%%", 100*result)))

xgb.pred = predict(model,dval,reshape=T)
result = sum(xgb.pred==val_label)/length(xgb.pred)
print(paste("Validation Accuracy =",sprintf("%1.2f%%", 100*result))) 

xgb.pred = predict(model,dtest,reshape=T)
result = sum(xgb.pred==te_label)/length(xgb.pred)
print(paste("Test Accuracy =",sprintf("%1.2f%%", 100*result)))

#In this case, svm and xgb not perform well. We try with CNN and pre-trained neural network
