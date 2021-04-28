#Inception-Resnet-V2
#preprocessing
datagen <- image_data_generator(
  preprocessing_function = inception_resnet_v2_preprocess_input
)
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

val_datagen <- image_data_generator(preprocessing_function = inception_resnet_v2_preprocess_input)
validation_generator <- flow_images_from_directory(
  directory= validation_dir,
  target_size = c(224,224),
  color_mode = 'rgb',
  class_mode = 'categorical',
  shuffle = TRUE,
  generator = val_datagen,
  seed=1)

test_generator <- flow_images_from_directory(
  directory= test_dir,
  target_size = c(224,224),
  color_mode = 'rgb',
  class_mode = 'categorical',
  shuffle = TRUE,
  generator = val_datagen,
  seed=1)

#load inception_resnet_v2
conv_base_inc<- application_inception_resnet_v2(weights = "imagenet",include_top = FALSE,input_shape = c(224, 224, 3))

#fine tuning
unfreeze_weights(conv_base_inc, from = "conv2d_88")
model2 <- keras_model_sequential() %>%
  conv_base_inc %>%
  layer_flatten() %>%
  layer_dense(units = 1024, activation = 'relu',kernel_initializer = 'he_uniform', kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 512, activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units = 512, activation = 'relu',kernel_initializer = 'he_uniform', kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dense(units = 4, activation = "softmax")

early = callback_early_stopping(monitor='val_loss', patience=6, verbose=1)
lr_decay = callback_reduce_lr_on_plateau(monitor='val_loss', patience=3, verbose=1, factor=0.5)
mod_check <- callback_model_checkpoint(filepath = 'incres_pesi.h5', monitor = "val_loss", save_best_only = TRUE)

model2 %>% compile(
  optimizer = optimizer_adam(lr = 1e-4),
  loss = "categorical_crossentropy",
  metrics = c("accuracy","AUC","Precision","Recall"))

history <- model2 %>% fit_generator(
  train_generator,
  steps_per_epoch = train_generator$n/train_generator$batch_size,
  epochs = 100,
  class_weight = list('0'=4,'1'=49,'2'=1,'3'=2),
  callbacks = list(early, lr_decay, mod_check),
  validation_data = validation_generator,
  validation_steps = validation_generator$n/validation_generator$batch_size
)

#best epoch
model2<- load_model_hdf5('incres_pesi.h5')
model2%>% evaluate_generator(test_generator, steps = test_generator$n/test_generator$batch_size)

#MOBILENET
#preprocessing with data augmentation (no weights)
datagen <- image_data_generator(
  rotation_range = 20,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  brightness_range = c(0.5, 1.5),
  horizontal_flip = TRUE,
  preprocessing_function = mobilenet_preprocess_input,
  fill_mode = "nearest"
)
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

val_datagen <- image_data_generator(preprocessing_function = mobilenet_preprocess_input)
validation_generator <- flow_images_from_directory(
  directory= validation_dir,
  target_size = c(224,224),
  color_mode = 'rgb',
  class_mode = 'categorical',
  shuffle = TRUE,
  generator = val_datagen,
  seed=1)

test_generator <- flow_images_from_directory(
  directory= test_dir,
  target_size = c(224,224),
  color_mode = 'rgb',
  class_mode = 'categorical',
  shuffle = TRUE,
  generator = val_datagen,
  seed=1)
#load mobile net

conv_base_mobile <- application_mobilenet(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224, 224, 3)
)
conv_base_mobile

#fine tuning
unfreeze_weights(conv_base_mobile, from = "conv_dw_3")
model <- keras_model_sequential() %>%
  conv_base_mobile %>%
  layer_flatten() %>%
  layer_dense(units = 1024, kernel_initializer = 'he_uniform', kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(rate=0.1) %>%
  layer_activation_parametric_relu()%>%
  layer_dense(units = 512,kernel_initializer = 'he_uniform', kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(rate=0.5) %>%
  layer_activation_parametric_relu()%>%
  layer_dense(units = 512,kernel_initializer = 'he_uniform', kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(rate=0.5) %>%
  layer_activation_parametric_relu()%>%
  layer_dense(units = 4, activation = "softmax")

early = callback_early_stopping(monitor='val_loss', patience=6, verbose=1)
lr_decay = callback_reduce_lr_on_plateau(monitor='val_loss', patience=3, verbose=1, factor=0.5)
mod_check <- callback_model_checkpoint(filepath = 'mobile_dataug.h5', monitor = "val_loss", save_best_only = TRUE)

model %>% compile(
  optimizer = optimizer_adam(lr = 1e-4),
  loss = "categorical_crossentropy",
  metrics = c("accuracy","AUC","Precision","Recall"))

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = train_generator$n/train_generator$batch_size,
  epochs = 100,
  callbacks = list(early, lr_decay, mod_check),
  validation_data = validation_generator,
  validation_steps = validation_generator$n/validation_generator$batch_size
)

#best epoch
model<- load_model_hdf5('mobile_dataug.h5')
model%>%evaluate_generator(test_generator, steps = test_generator$n/test_generator$batch_size)

