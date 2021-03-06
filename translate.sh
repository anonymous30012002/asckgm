CUDA_VISIBLE_DEVICES=1 python translate${TRAIN_TYPE}.py \
-checkpoint_path $CHECKPOINT_PATH \
-vocab_path $VOCAB_PATH \
-test_pkl_path $TEST_PKL \
-config_file_path $CONFIG_FILE_PATH \
-out_file_path $OUT_FILE_PATH \
-annoy_file_path $ANNOY_FILE \
-annoy_pkl_path $ANNOY_PKL \
-model_type $MODEL_TYPE \
-context_size $CONTEXT_SIZE \
-test_state_pkl_path $TEST_STATES_PKL \
-test_kb_path $TEST_KB_PKL \
-test_aspect_path $TEST_ASPECT_PKL \
-test_sentiment_path $TEST_SENTIMENT_PKL \
-test_celeb_path $TEST_CELEB_PKL \
-kb_vocab_path $KB_VOCAB_PATH \
-celeb_vocab_path $CELEB_VOCAB_PATH \
-num_states $NUM_STATES \
-out_class_file_path $OUT_CLASS_FILE \
-use_kb $USE_KB \
-use_aspect $USE_ASPECT \
-use_sentiment $USE_SENTIMENT \
-use_review $USE_REVIEW