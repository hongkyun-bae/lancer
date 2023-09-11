import os
#######################          model_name


class BaseConfig():
    """
    General configurations appiled to all models
    """

    impre_ratio = 0.5
    num_epochs = 1
    early_stop_patience = 2



#######################          candidate_type
#######################          loss_function
#######################          lifetime
#######################          negative_sampling_ratio
#######################          numbering
#######################          data    
#######################          testdata    
#######################          testfilter    
#######################          history_type    
    
    
    
    # candidate_type = "impre"
    ##uniform / fame / overlap/ overlap_fame / random / impre / 10popFilter_overlap
    ## logoverlap_hour / logoverlap_sec / logpop /50,75,80,85,90,95,97,overlapFilter_pop /
    ## overlap_logpop / random_pop / overlap_test1,2 / 90overlap1Filter_pop / 90overlap2Filter_pop
    our_type = "onetype"   #combine # onetype
    # loss_function = "CEL" #BPR_soft #BPR_sig #CEL



    num_batches_show_loss = 1000  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    # num_batches_validate = 1000
    batch_size = 50
    learning_rate = 0.0001
    num_workers = 0  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 1
    # entity_freq_threshold = 2
    # entity_confidence_threshold = 0.5
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 330899
    num_categories = 1 + 127
    # num_entities = 1 + 15587
    num_users = 1 + 229061
    word_embedding_dim = 100
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200
    num_words_cat = 5
    behaviors_target_file = 'behaviors_parsed_ns'+str(negative_sampling_ratio)+'_lt'+str(lifetime)+'.tsv'



class FIM_randomConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title', 'category', 'subcategory'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3
    HDC_window_size = 3
    HDC_filter_num = 150
    conv3D_filter_num_first = 32
    conv3D_kernel_size_first = 3
    conv3D_filter_num_second = 16
    conv3D_kernel_size_second = 3
    maxpooling3D_size = 3
    maxpooling3D_stride = 3

class FIMConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title', 'category_word'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3
    HDC_window_size = 3
    HDC_filter_num = 150
    conv3D_filter_num_first = 32
    conv3D_kernel_size_first = 3
    conv3D_filter_num_second = 16
    conv3D_kernel_size_second = 3
    maxpooling3D_size = 3
    maxpooling3D_stride = 3

class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For multi-head self-attention
    num_attention_heads = 10


class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 100
    window_size = 3
    long_short_term_method = 'ini'
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5


class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['title', 'title_entities'], "record": []}
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False


class HiFiArkConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    num_pooling_heads = 5
    regularizer_loss_weight = 0.1


class TANRConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1

