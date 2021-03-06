read -p "Is train: (1 - Yes, 0 - no): " IS_TRAIN

## We had srun queuing system to run models on server with
## 2 types of queues and multiple gpus
## Ignore this
#read -p "Enter queue: i:intel / a:amd  " Q
#case $Q in
#    a )
#		read -p "Enter gpu (write as 07/08):  " GPU
#		NODE=gpu$GPU
#		QUEUE=amd-longq
#		;;
#    i )
#		NODE=dgx01
#		QUEUE=intel-longq
#		;;
#esac

read -p "Please enter data version (as v2). Default v2: " DATA_VERSION
read -p "Please enter context size. Default 2: " CONTEXT_SIZE
# For now we train and evaluate on the whole dataset. #Future work
read -p "Please enter state data directory. Blank for all: " DATA_STATE_DIR
echo $DATA_STATE_DIR
read -p "Please enter state model saved directory. Blank for all: " MODEL_STATE_DIR

# Model variant 
read -p "Enter model: m:multimodelHRED / h:hred / t:Transformer /M:MTransformer" MODEL_INPUT
case $MODEL_INPUT in
    m ) 
		MODEL_TYPE=MultimodalHRED
		;;
    h ) 
		MODEL_TYPE=HRED # Text based HRED
		;;
    t )
    MODEL_TYPE=Transformer
    ;;
    M )
    MODEL_TYPE=MTransformer
    ;;
    #MKb )
    #MODEL_TYPE=MKbTransformer
    #;;
esac

read -p "Please enter config version (as _v4). Default _v4: " CONFIG_VERSION
read -p "Please enter checkpoint epoch (like 20/30). Default 10: " CHECKPOINT_EPOCH
read -p "Please enter max len for data directory: (as _20). Default 20: " MAX_LEN

read -p "Use KB: t:true / f:false (Default: False)" KB_INPUT
case $KB_INPUT in
    t ) 
		USE_KB=True
		;;
    f ) 
		USE_KB=False
		;;
esac
export USE_KB=${USE_KB:-False}

read -p "Use ASPECT: t:true / f:false (Default: False)" ASPECT_INPUT
case $ASPECT_INPUT in
    t ) 
		USE_ASPECT=True
		;;
    f ) 
		USE_ASPECT=False
		;;
esac
export USE_ASPECT=${USE_ASPECT:-False}

read -p "Use REVIEW: t:true / f:false (Default: False)" REVIEW_INPUT
case $REVIEW_INPUT in
    t ) 
		USE_REVIEW=True
		;;
    f ) 
		USE_REVIEW=False
		;;
esac
export USE_REVIEW=${USE_REVIEW:-False}

read -p "Use SENTIMENT: t:true / f:false (Default: False)" SENTIMENT_INPUT
case $SENTIMENT_INPUT in
    t ) 
		USE_SENTIMENT=True
		;;
    f ) 
		USE_SENTIMENT=False
		;;
esac
export USE_SENTIMENT=${USE_SENTIMENT:-False}

MAX_LEN=${MAX_LEN:-_20}
CONFIG_VERSION=${CONFIG_VERSION:-_v4}
CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH:-10}
CONTEXT_SIZE=${CONTEXT_SIZE:-2}
DATA_VERSION=${DATA_VERSION:-v2}
NUM_STATES=${NUM_STATES:-10}

export NUM_STATES=$NUM_STATES
export MODEL_TYPE=$MODEL_TYPE
export DATA_STATE_DIR=$DATA_STATE_DIR
export MODEL_STATE_DIR=$MODEL_STATE_DIR
export CHECKPOINT_EPOCH=$CHECKPOINT_EPOCH
export MAX_LEN=$MAX_LEN
export CONFIG_VERSION=$CONFIG_VERSION
export CONTEXT_SIZE=$CONTEXT_SIZE
export DATA_VERSION=$DATA_VERSION

export PROJECT_DIR=${PWD}
export DATA_DIR=$PROJECT_DIR/data/dataset/${DATA_VERSION}/dialogue_data/
export CONTEXT_DATA_DIR=$DATA_DIR/context_${CONTEXT_SIZE}${MAX_LEN}/
# Actual data dir. could be only related to a particular intent state  
export DIR_PKL=$CONTEXT_DATA_DIR/$DATA_STATE_DIR

# Common
export HRED_CODE_DIR=$PROJECT_DIR/
export HRED_MODEL_DIR=$PROJECT_DIR/models
export MODEL_DIR=$HRED_MODEL_DIR/context_${CONTEXT_SIZE}${MAX_LEN}/$MODEL_TYPE/$MODEL_STATE_DIR/$CONFIG_VERSION/

echo "Making model dir"
echo $MODEL_DIR
mkdir -p $MODEL_DIR

# Config related variables
export CONFIG_FILE_PATH=$HRED_CODE_DIR/config/config_hred_mmd${CONFIG_VERSION}.json
export TRAIN_PKL=$DIR_PKL/train.pkl
export VALID_PKL=$DIR_PKL/valid.pkl
export TEST_PKL=$DIR_PKL/test.pkl
export VOCAB_PATH=$CONTEXT_DATA_DIR/vocab.pkl

# KB related path
export TRAIN_KB_PKL=$DIR_PKL/train_kb_text_both.pkl
export VALID_KB_PKL=$DIR_PKL/valid_kb_text_both.pkl
export TEST_KB_PKL=$DIR_PKL/test_kb_text_both.pkl
export TRAIN_CELEB_PKL=$DIR_PKL/train_celeb_text_both.pkl
export VALID_CELEB_PKL=$DIR_PKL/valid_celeb_text_both.pkl
export TEST_CELEB_PKL=$DIR_PKL/test_celeb_text_both.pkl
export TRAIN_ASPECT_PKL=$DIR_PKL/train_aspect_text_both.pkl
export VALID_ASPECT_PKL=$DIR_PKL/valid_aspect_text_both.pkl
export TEST_ASPECT_PKL=$DIR_PKL/test_aspect_text_both.pkl
export TRAIN_SENTIMENT_PKL=$DIR_PKL/train_sentiment_text_both.pkl
export VALID_SENTIMENT_PKL=$DIR_PKL/valid_sentiment_text_both.pkl
export TEST_SENTIMENT_PKL=$DIR_PKL/test_sentiment_text_both.pkl
export CELEB_VOCAB_PATH=$CONTEXT_DATA_DIR/celeb_vocab.pkl
export KB_VOCAB_PATH=$CONTEXT_DATA_DIR/kb_vocab.pkl

# States related path 
export TRAIN_STATES_PKL=$CONTEXT_DATA_DIR/train_states.pkl
export VALID_STATES_PKL=$CONTEXT_DATA_DIR/valid_states.pkl
export TEST_STATES_PKL=$CONTEXT_DATA_DIR/test_states.pkl


echo $MODEL_TYPE
echo "Model saved in: "
echo $MODEL_DIR
echo "Following config: "
echo $CONFIG_FILE_PATH

export ANNOY_PATH=${PWD}/data/raw_catlog/image_annoy_index
export ANNOY_FILE=$ANNOY_PATH/annoy.ann
export ANNOY_PKL=$ANNOY_PATH/ImageUrlToIndex.pkl
# export ANNOY_PKL=$ANNOY_PATH/FileNameMapToIndex.pkl

export CHECKPOINT_PATH=$MODEL_DIR/model_params_${CHECKPOINT_EPOCH}.pkl
export RESULTS_DIR=$MODEL_DIR/$DATA_STATE_DIR
export OUT_FILE_PATH=$RESULTS_DIR/pred_100samp_${CHECKPOINT_EPOCH}.txt
export OUT_CLASS_FILE=$RESULTS_DIR/class_${CHECKPOINT_EPOCH}.txt

export TEST_TARGET=$DIR_PKL/test_target_text.txt
export TEST_CONTEXT=$DIR_PKL/test_context_text.txt
export LOG_BLEU_FILE=$RESULTS_DIR/bleu_${CHECKPOINT_EPOCH}.txt
export TEST_TARGET_TOKENIZED=$RESULTS_DIR/test_tokenized.txt
export LOG_BLEU_TOKENIZED=$RESULTS_DIR/bleu_tokenized_${CHECKPOINT_EPOCH}.txt

cp $TEST_CONTEXT $RESULTS_DIR
cp $CONFIG_FILE_PATH $RESULTS_DIR
export RESULTS_FILE=$RESULTS_DIR/results_${CHECKPOINT_EPOCH}.txt

echo "Saving results to" 
echo $RESULTS_DIR
mkdir -p $RESULTS_DIR

if [ $IS_TRAIN == 1 ]; then
echo "Training"
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME"
echo "-model_path $MODEL_DIR -vocab_path $VOCAB_PATH -train_pkl_path $TRAIN_PKL -valid_pkl_path $VALID_PKL -config_file_path $CONFIG_FILE_PATH -annoy_file_path $ANNOY_FILE -annoy_pkl_path $ANNOY_PKL -model_type $MODEL_TYPE -context_size $CONTEXT_SIZE -train_state_pkl_path $TRAIN_STATES_PKL -valid_state_pkl_path $VALID_STATES_PKL -train_kb_path $TRAIN_KB_PKL -valid_kb_path $VALID_KB_PKL -train_aspect_path $TRAIN_ASPECT_PKL -valid_aspect_path $VALID_ASPECT_PKL -train_sentiment_path $TRAIN_SENTIMENT_PKL -valid_sentiment_path $VALID_SENTIMENT_PKL  -train_celeb_path $TRAIN_CELEB_PKL -valid_celeb_path $VALID_CELEB_PKL -kb_vocab_path $KB_VOCAB_PATH -celeb_vocab_path $CELEB_VOCAB_PATH -num_states $NUM_STATES -use_kb $USE_KB -use_aspect $USE_ASPECT -use_sentiment $USE_SENTIMENT -use_review $USE_REVIEW "

#srun --partition $QUEUE --nodelist=$NODE --gres=gpu -c4 \
bash train.sh > $MODEL_DIR/logs.txt
fi
echo "Training done"

# Printing time
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME"

echo "-checkpoint_path $CHECKPOINT_PATH \
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
-use_review $USE_REVIEW"
#srun --partition $QUEUE --nodelist=$NODE --gres=gpu -c4 \
bash translate.sh > $MODEL_DIR/logs_translate_100.txt
echo "Translation done"

# Printing time
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME"

python utils/convert_test_for_bleu.py \
-input_file $TEST_TARGET \
-tokenized_file $TEST_TARGET_TOKENIZED
echo "Test tokenized"

#srun --partition $QUEUE --nodelist=$NODE --gres=gpu -c4 \
bash evaluation/nlg_eval.sh > $RESULTS_DIR/metrics_tokenized_${CHECKPOINT_EPOCH}.txt
echo "Bleu done"

