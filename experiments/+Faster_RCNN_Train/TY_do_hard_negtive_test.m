function imdb_new = TY_do_hard_negtive_test(conf, model_stage, imdb, roidb, ignore_cache)
    if ~exist('ignore_cache', 'var')
        ignore_cache            = false;
    end

    imdb_new{1}                         = TY_hard_negtive_test(conf, imdb{1}, roidb, ...
                                    'net_def_file',     model_stage.test_net_def_file, ...
                                    'net_file',         model_stage.output_model_file, ...
                                    'cache_name',       model_stage.cache_name, ...
                                    'ignore_cache',     ignore_cache);
end