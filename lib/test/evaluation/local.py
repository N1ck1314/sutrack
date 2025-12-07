from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/lls/sjj/sutrack/data/got10k_lmdb'
    settings.got10k_path = '/media/lls/新加卷/dataset/GOT-10k/full_data/test_data/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/lls/sjj/sutrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/lls/sjj/sutrack/data/lasot_lmdb'
    settings.lasot_path = '/home/lls/sjj/sutrack/data/lasot'
    settings.lasotlang_path = '/home/lls/sjj/sutrack/data/lasot'
    settings.network_path = '/home/lls/sjj/sutrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/lls/sjj/sutrack/data/nfs'
    settings.otb_path = '/home/lls/sjj/sutrack/data/OTB2015'
    settings.otblang_path = '/home/lls/sjj/sutrack/data/otb_lang'
    settings.prj_dir = '/home/lls/sjj/sutrack'
    settings.result_plot_path = '/home/lls/sjj/sutrack/test/result_plots'
    settings.results_path = '/home/lls/sjj/sutrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/lls/sjj/sutrack'
    settings.segmentation_path = '/home/lls/sjj/sutrack/test/segmentation_results'
    settings.tc128_path = '/home/lls/sjj/sutrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/lls/sjj/sutrack/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/lls/新加卷/dataset/trackingnet/TEST'
    settings.uav_path = '/media/lls/新加卷/dataset/Dataset-UAV-123/data/UAV123'
    settings.vot_path = '/media/lls/新加卷/dataset/VOT2019-RGBD/VOT2019-RGBD'
    settings.youtubevos_dir = ''

    return settings
