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

#best structure of CNN
model6 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), kernel_regularizer = regularizer_l2(l=0.001), input_shape = c(224, 224 , 3)) %>%
  layer_batch_normalization() %>%
  layer_activation_parametric_relu() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), kernel_regularizer = regularizer_l2(l=0.001)) %>%
  layer_batch_normalization() %>%
  layer_activation_parametric_relu() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), kernel_regularizer = regularizer_l2(l=0.001)) %>%
  layer_batch_normalization() %>%
  layer_activation_parametric_relu() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), kernel_regularizer = regularizer_l2(l=0.001)) %>%
  layer_batch_normalization() %>%
  layer_activation_parametric_relu() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), kernel_regularizer = regularizer_l2(l=0.001)) %>%
  layer_batch_normalization() %>%
  layer_activation_parametric_relu() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, kernel_regularizer = regularizer_l2(l=0.001)) %>%
  layer_batch_normalization() %>%
  layer_activation_parametric_relu() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 4) %>%
  layer_activation_softmax()

#optimizer, loss and metrics
model6 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4), metrics = c("acc","AUC","Precision","Recall")
)

#callbacks
early = callback_early_stopping(monitor='val_loss', patience=6, verbose=1) #ferma epoche se val_loss aumenta per 6 epoche consecutive
lr_decay = callback_reduce_lr_on_plateau(monitor='val_loss', patience=3, verbose=1, factor=0.5) #riduce learning rate quando val_loss cresce per 3 epoche consecutive
salva = callback_model_checkpoint(filepath = "C:/Users/feder/Downloads/best_model6.h5", monitor = "val_loss", save_best_only = TRUE) 

history6<- model6 %>% fit_generator(
  train_generator,
  steps_per_epoch = train_generator$n/train_generator$batch_size,
  epochs = 100,  
  class_weight= list('0'=4,'1'=49, '2'=1, '3'=2),
  callbacks = list(lr_decay,salva,early),
  validation_data = validation_generator,
  validation_step = validation_generator$n/validation_generator$batch_size
)
#best epoch
model = load_model_hdf5("best_model6.h5")
model %>% evaluate(test_generator, steps= test_generator$n/test_generator$batch_size)

