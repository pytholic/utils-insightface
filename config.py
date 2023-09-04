from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4) 
config.network = "mbf"
config.resume = False
config.suffix_run_name = config.network + "-GNAP-emore-512" # network|last module|dataset|batch
config.output = "work_dirs/" + config.suffix_run_name 
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 512
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/home/jovyan/haseeb-dataset-3090-1/face-data/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 40
config.warmup_epoch = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
config.result_dir = "./results"

config.notes = ""
config.wandb_entity = None
config.wandb_project = "insightface"

config.num_workers = 6
config.global_layer = "GDC" # "GDC" or "GNAP"
config.optimizer = "sgd"
