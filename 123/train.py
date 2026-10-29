# # 8通道整体
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 引入 thop 计算 FLOPs (如果没有请 pip install thop)
# try:
#     from thop import profile
# except ImportError:
#     print("【警告】未安装 thop 库，无法自动计算 FLOPs。请运行 pip install thop")
#     profile = None

# from nets.segformer import SegFormer
# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
#                                      weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config,
#                          worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   辅助函数：计算 Params 和 FLOPs
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         return 0, 0, 0
    
#     # 切换到评估模式
#     model_eval = model.eval()
#     # 创建伪造输入 (Batch=1, Channels, Height, Width)
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1])
#     if torch.cuda.is_available():
#         dummy_input = dummy_input.cuda()
#         model_eval = model_eval.cuda()

#     try:
#         flops, params = profile(model_eval, inputs=(dummy_input, ), verbose=False)
#     except Exception as e:
#         print(f"计算 FLOPs 失败: {e}")
#         return 0, 0, 0
    
#     # 恢复训练模式
#     model.train()
    
#     # 转换单位
#     flops_g = flops / 1e9             # GFLOPs
#     params_m = params / 1e6           # Million (10^6)
#     params_mb = params * 4 / (1024 * 1024) # MB (假设 float32, 4 bytes)
    
#     return flops_g, params_m, params_mb

# if __name__ == "__main__":
#     # ---------------------------------#
#     #   Cuda    是否使用Cuda
#     # ---------------------------------#
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
    
#     # -----------------------------------------------------#
#     #   num_classes     分类个数+1
#     # -----------------------------------------------------#
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   【设置】输入通道数 (双时相4波段 = 8)
#     # -------------------------------------------------------------------#
#     input_channels  = 8  

#     phi             = "b5"
#     pretrained      = True
#     model_path      = "model_data/segformer_b5_backbone_weights.pth"
    
#     input_shape     = [512, 512]
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # ------------------------------------------------------------------#
#     #   训练参数
#     # ------------------------------------------------------------------#
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
#     save_dir        = 'logs'
    
#     # ------------------------------------------------------------------#
#     #   【修改核心】评估设置
#     #   1. eval_flag = True (必须开启才能计算精度)
#     #   2. eval_period = 5 (每5轮计算一次，符合你的打印要求)
#     # ------------------------------------------------------------------#
#     eval_flag       = True 
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     if pretrained:
#         if distributed:
#             if local_rank == 0:
#                 download_weights(phi)  
#             dist.barrier()
#         else:
#             download_weights(phi)

#     # 构建模型
#     model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained, in_chans=input_channels)
    
#     # -------------------------------------------------------------------#
#     #   【新增功能1】计算静态指标 (FLOPs, Params)
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0
#     if local_rank == 0:
#         print("\n正在计算模型复杂度 (FLOPs & Params)...")
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
        
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location = device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # 记录 Loss
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     # -------------------------------------------------------------------#
#     #   【新增功能2】初始化详细指标 CSV
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         # 写入表头
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch', 
#                     'Train Time (min)', 
#                     'Params Size (MB)', 
#                     'Total Params (10^6)', 
#                     'FLOPs (G)', 
#                     'mIoU (%)', 
#                     'mPA (%)', 
#                     'Accuracy (%)', 
#                     'F1 Score'
#                 ]
#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     if True:
#         UnFreeze_flag = False
        
#         if Freeze_Train:
#             for param in model.backbone.parameters():
#                 param.requires_grad = False

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                     drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#         gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                     drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#         if local_rank == 0:
#             eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
#                                             eval_flag=eval_flag, period=eval_period)
#         else:
#             eval_callback   = None
        
#         # ---------------------------------------#
#         #   Epoch 循环
#         # ---------------------------------------#
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             # 记录本轮开始时间
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size

#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#                 for param in model.backbone.parameters():
#                     param.requires_grad = True
                            
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#                 gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                             drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#                 gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                             drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#                 UnFreeze_flag   = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
#                 dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   【核心功能】每5轮记录一次详细表格并打印
#             #   只有 eval_period=5 的时候，eval_callback 才会计算出新的精度
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#                 # 获取最新计算的指标 (从 eval_callback 实例中读取)
#                 # 注意：这里假设 EvalCallback 已经计算完并存入了列表中
#                 # 如果没有数据，给默认值
#                 current_miou = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 # F1 Score 这种库可能没有直接算，如果没有你可以用 mIoU 代替或者自己加
#                 # 这里假设 EvalCallback 有 f1s 属性，如果没有，显示 0
#                 current_f1 = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0

#                 # 1. 写入 CSV
#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}"
#                     ]
#                     writer.writerow(row_data)

#                 # 2. 打印漂亮的表格
#                 table_data = {
#                     "Metric": ["Epoch", "Train Time (min)", "Params Size (MB)", "Total Params (10^6)", "FLOPs (G)", "mIoU (%)", "mPA (%)", "Accuracy (%)", "F1 Score"],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1}")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()


# 最新八通道整体添加俩指标
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 引入 thop 计算 FLOPs (如果没有请 pip install thop)
# try:
#     from thop import profile
# except ImportError:
#     print("【警告】未安装 thop 库，无法自动计算 FLOPs。请运行 pip install thop")
#     profile = None

# from nets.segformer import SegFormer
# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
#                                      weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config,
#                          worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   辅助函数：计算 Params 和 FLOPs
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         return 0, 0, 0
    
#     # 切换到评估模式
#     model_eval = model.eval()
#     # 创建伪造输入 (Batch=1, Channels, Height, Width)
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1])
#     if torch.cuda.is_available():
#         dummy_input = dummy_input.cuda()
#         model_eval = model_eval.cuda()

#     try:
#         flops, params = profile(model_eval, inputs=(dummy_input, ), verbose=False)
#     except Exception as e:
#         print(f"计算 FLOPs 失败: {e}")
#         return 0, 0, 0
    
#     # 恢复训练模式
#     model.train()
    
#     # 转换单位
#     flops_g = flops / 1e9             # GFLOPs
#     params_m = params / 1e6           # Million (10^6)
#     params_mb = params * 4 / (1024 * 1024) # MB (假设 float32, 4 bytes)
    
#     return flops_g, params_m, params_mb

# if __name__ == "__main__":
#     # ---------------------------------#
#     #   Cuda    是否使用Cuda
#     # ---------------------------------#
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
    
#     # -----------------------------------------------------#
#     #   num_classes     分类个数+1
#     # -----------------------------------------------------#
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   【设置】输入通道数 (双时相4波段 = 8)
#     # -------------------------------------------------------------------#
#     input_channels  = 8  

#     phi             = "b5"
#     pretrained      = True
#     model_path      = "model_data/segformer_b5_backbone_weights.pth"
    
#     input_shape     = [512, 512]
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # ------------------------------------------------------------------#
#     #   训练参数
#     # ------------------------------------------------------------------#
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
#     save_dir        = 'logs'
    
#     # ------------------------------------------------------------------#
#     #   【修改核心】评估设置
#     #   1. eval_flag = True (必须开启才能计算精度)
#     #   2. eval_period = 5 (每5轮计算一次，符合打印要求)
#     # ------------------------------------------------------------------#
#     eval_flag       = True 
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     if pretrained:
#         if distributed:
#             if local_rank == 0:
#                 download_weights(phi)  
#             dist.barrier()
#         else:
#             download_weights(phi)

#     # 构建模型
#     model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained, in_chans=input_channels)
    
#     # -------------------------------------------------------------------#
#     #   【新增功能1】计算静态指标 (FLOPs, Params)
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0
#     if local_rank == 0:
#         print("\n正在计算模型复杂度 (FLOPs & Params)...")
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
        
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location = device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # 记录 Loss
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     # -------------------------------------------------------------------#
#     #   【新增功能2】初始化详细指标 CSV (包含 clDice 和 CC-F1)
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         # 写入表头
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch', 
#                     'Train Time (min)', 
#                     'Params Size (MB)', 
#                     'Total Params (10^6)', 
#                     'FLOPs (G)', 
#                     'mIoU (%)', 
#                     'mPA (%)', 
#                     'Accuracy (%)', 
#                     'F1 Score',
#                     'clDice (%)',   # 新增
#                     'CC-F1 (%)'     # 新增
#                 ]
#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     if True:
#         UnFreeze_flag = False
        
#         if Freeze_Train:
#             for param in model.backbone.parameters():
#                 param.requires_grad = False

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                     drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#         gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                     drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#         if local_rank == 0:
#             eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
#                                             eval_flag=eval_flag, period=eval_period)
#         else:
#             eval_callback   = None
        
#         # ---------------------------------------#
#         #   Epoch 循环
#         # ---------------------------------------#
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             # 记录本轮开始时间
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size

#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#                 for param in model.backbone.parameters():
#                     param.requires_grad = True
                            
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#                 gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                             drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#                 gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                             drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#                 UnFreeze_flag   = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
#                 dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   【核心功能】每5轮记录一次详细表格并打印 (已适配 clDice 和 CC-F1)
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#                 # 安全获取最新计算的指标，避免列表越界
#                 current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
                
#                 # 新增指标的提取
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 # 1. 写入 CSV
#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",  # 转换为百分比
#                         f"{current_ccf1 * 100:.2f}"     # 转换为百分比
#                     ]
#                     writer.writerow(row_data)

#                 # 2. 打印漂亮的表格
#                 table_data = {
#                     "Metric": ["Epoch", "Train Time (min)", "Params Size (MB)", "Total Params (10^6)", "FLOPs (G)", "mIoU (%)", "mPA (%)", "Accuracy (%)", "F1 Score", "clDice (%)", "CC-F1 (%)"],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1}")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()


# 完整8通道俩指标五次随机种子
import argparse
import datetime
import os
import time
import csv
import copy
import pandas as pd
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

# 引入 thop 计算 FLOPs (如果没有请 pip install thop)
try:
    from thop import profile
except ImportError:
    print("【警告】未安装 thop 库，无法自动计算 FLOPs。请运行 pip install thop")
    profile = None

from nets.segformer import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

# -------------------------------------------------------------------#
#   辅助函数：计算 Params 和 FLOPs
#   使用深拷贝 + CPU，避免 thop hook 污染真实训练模型
# -------------------------------------------------------------------#
def get_model_complexity(model, input_channels, input_shape):
    if profile is None:
        return 0, 0, 0

    try:
        model_copy = copy.deepcopy(model).cpu().eval()
    except Exception as e:
        print(f"计算 FLOPs 时模型深拷贝失败: {e}")
        return 0, 0, 0

    dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1]).cpu()

    try:
        flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"计算 FLOPs 失败: {e}")
        return 0, 0, 0
    finally:
        del model_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    flops_g = flops / 1e9
    params_m = params / 1e6
    params_mb = params * 4 / (1024 * 1024)

    return flops_g, params_m, params_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    parser.add_argument("--save_dir", type=str, default="logs", help="directory to save logs/checkpoints")
    args = parser.parse_args()

    # ---------------------------------#
    #   Cuda    是否使用Cuda
    # ---------------------------------#
    Cuda            = True
    seed            = args.seed
    distributed     = False
    sync_bn         = False
    fp16            = False
    
    # -----------------------------------------------------#
    #   num_classes     分类个数+1
    # -----------------------------------------------------#
    num_classes     = 2
    
    # -------------------------------------------------------------------#
    #   输入通道数 (双时相4波段 = 8)
    # -------------------------------------------------------------------#
    input_channels  = 8

    phi             = "b5"
    pretrained      = True
    model_path      = "model_data/segformer_b5_backbone_weights.pth"
    
    input_shape     = [512, 512]
    
    Init_Epoch      = 0
    Freeze_Epoch    = 50
    Freeze_batch_size = 8
    UnFreeze_Epoch  = 100
    Unfreeze_batch_size = 4
    
    Freeze_Train    = False

    # ------------------------------------------------------------------#
    #   训练参数
    # ------------------------------------------------------------------#
    Init_lr         = 1e-4
    Min_lr          = Init_lr * 0.01
    optimizer_type  = "adamw"
    momentum        = 0.9
    weight_decay    = 1e-2
    lr_decay_type   = 'cos'
    save_period     = 5
    save_dir        = args.save_dir
    
    # ------------------------------------------------------------------#
    #   评估设置
    # ------------------------------------------------------------------#
    eval_flag       = True 
    eval_period     = 5

    VOCdevkit_path  = 'VOCdevkit'
    dice_loss       = True
    focal_loss      = False
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers     = 4

    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ngpus_per_node  = torch.cuda.device_count()
    
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
    else:
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank  = 0
        rank        = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)
            dist.barrier()
        else:
            download_weights(phi)

    # 构建模型
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained, in_chans=input_channels)
    
    # -------------------------------------------------------------------#
    #   计算静态指标 (FLOPs, Params)
    # -------------------------------------------------------------------#
    flops_g, params_m, params_mb = 0, 0, 0
    if local_rank == 0:
        print("\n正在计算模型复杂度 (FLOPs & Params)...")
        flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
        print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

    if not pretrained:
        weights_init(model)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}

        if isinstance(pretrained_dict, dict):
            if "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]
            elif "model" in pretrained_dict:
                pretrained_dict = pretrained_dict["model"]

        cleaned_dict = {}
        for k, v in pretrained_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned_dict[nk] = v

        for k, v in cleaned_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict, strict=False)
        
        if local_rank == 0:
            print("\nSuccessful Load Key Num:", len(load_key))
            print("Fail To Load Key Num:", len(no_load_key))

    # 记录 Loss
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = False
            model_train = model_train.cuda()
    
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, phi=phi, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    # -------------------------------------------------------------------#
    #   初始化详细指标 CSV (包含 clDice 和 CC-F1)
    # -------------------------------------------------------------------#
    metrics_csv_path = ""
    if local_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
        if not os.path.exists(metrics_csv_path):
            with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = [
                    'Epoch', 
                    'Train Time (min)', 
                    'Params Size (MB)', 
                    'Total Params (10^6)', 
                    'FLOPs (G)', 
                    'mIoU (%)', 
                    'mPA (%)', 
                    'Accuracy (%)', 
                    'F1 Score',
                    'clDice (%)',
                    'CC-F1 (%)'
                ]
                writer.writerow(header)

    # ------------------------------------------------------#
    #   开始训练
    # ------------------------------------------------------#
    UnFreeze_flag = False
        
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    nbs             = 16
    lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
    epoch_step      = num_train // batch_size
    epoch_step_val  = max(1, num_val // batch_size)
        
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset   = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size    = batch_size // ngpus_per_node
        train_shuffle = False
        val_shuffle   = False
    else:
        train_sampler = None
        val_sampler   = None
        train_shuffle = True
        val_shuffle   = False

    gen = DataLoader(
        train_dataset,
        shuffle=train_shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
    )

    gen_val = DataLoader(
        val_dataset,
        shuffle=val_shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=seg_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
    )

    if local_rank == 0:
        eval_callback = EvalCallback(
            model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
            eval_flag=eval_flag, period=eval_period
        )
    else:
        eval_callback = None
        
    # ---------------------------------------#
    #   Epoch 循环
    # ---------------------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        epoch_start_time = time.time()

        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            nbs             = 16
            lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
            lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
            Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
            for param in model.backbone.parameters():
                param.requires_grad = True
                            
            epoch_step      = num_train // batch_size
            epoch_step_val  = max(1, num_val // batch_size)

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            gen = DataLoader(
                train_dataset,
                shuffle=train_shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=seg_dataset_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
            )
            gen_val = DataLoader(
                val_dataset,
                shuffle=val_shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=seg_dataset_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
            )

            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer, epoch,
            epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
            dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
            save_period, save_dir, local_rank
        )

        if distributed:
            dist.barrier()

        # -------------------------------------------------------------------#
        #   每5轮记录一次详细表格并打印
        # -------------------------------------------------------------------#
        if local_rank == 0 and (epoch + 1) % eval_period == 0:
            epoch_end_time = time.time()
            train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
            current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
            current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
            current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
            current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
            current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
            current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

            with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row_data = [
                    epoch + 1,
                    f"{train_time_min:.2f}",
                    f"{params_mb:.2f}",
                    f"{params_m:.3f}",
                    f"{flops_g:.3f}",
                    f"{current_miou * 100:.2f}",
                    f"{current_mpa * 100:.2f}",
                    f"{current_acc * 100:.2f}",
                    f"{current_f1:.4f}",
                    f"{current_cldice * 100:.2f}",
                    f"{current_ccf1 * 100:.2f}"
                ]
                writer.writerow(row_data)

            table_data = {
                "Metric": [
                    "Epoch",
                    "Train Time (min)",
                    "Params Size (MB)",
                    "Total Params (10^6)",
                    "FLOPs (G)",
                    "mIoU (%)",
                    "mPA (%)",
                    "Accuracy (%)",
                    "F1 Score",
                    "clDice (%)",
                    "CC-F1 (%)"
                ],
                "Value": row_data
            }
            df = pd.DataFrame(table_data)
            print("\n" + "="*50)
            print(f"Performance Summary at Epoch {epoch + 1}")
            print("="*50)
            print(df.to_string(index=False))
            print("="*50 + "\n")

    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()




# unet++
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# import copy
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 引入 thop 计算 FLOPs
# try:
#     from thop import profile
# except ImportError:
#     print("【警告】未安装 thop 库，无法自动计算 FLOPs。请运行 pip install thop")
#     profile = None

# # -------------------------------------------------------------------#
# #   导入模型 (UNet++)
# # -------------------------------------------------------------------#
# from nets.unetplusplus import UNetPlusPlus
# from nets.segformer_training import (
#     get_lr_scheduler,
#     set_optimizer_lr,
#     weights_init
# )
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (
#     download_weights,
#     seed_everything,
#     show_config,
#     worker_init_fn
# )
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   鲁棒的 FLOPs 计算函数
# #   强制使用 CPU 计算，避免显存干扰，且不影响原模型
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         return 0, 0, 0

#     # 1. 深拷贝模型并转为 CPU、Eval 模式
#     #    这样做最安全，绝对不会影响训练过程
#     try:
#         model_copy = copy.deepcopy(model).to('cpu').eval()
#     except Exception as e:
#         # 如果模型无法深拷贝（极少数情况），则直接使用原模型
#         print(f"【注意】模型深拷贝失败，将使用原模型计算: {e}")
#         model_copy = model.eval().cpu()

#     # 2. 创建 CPU 上的虚拟输入
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1]).to('cpu')

#     try:
#         # 3. 计算 FLOPs 和 Params
#         flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
#     except Exception as e:
#         print(f"【错误】FLOPs 计算失败: {e}")
#         return 0, 0, 0
#     finally:
#         # 4. 释放临时模型，清理内存
#         del model_copy
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # 5. 单位转换
#     flops_g = flops / 1e9
#     params_m = params / 1e6
#     params_mb = params * 4 / (1024 * 1024)

#     # 6. 恢复原模型状态
#     if next(model.parameters()).is_cuda:
#         model.train().cuda()
#     else:
#         model.train()

#     return flops_g, params_m, params_mb


# if __name__ == "__main__":
#     # ---------------------------------#
#     #   Cuda 是否使用Cuda
#     # ---------------------------------#
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False

#     # -----------------------------------------------------#
#     #   num_classes 分类个数
#     # -----------------------------------------------------#
#     num_classes     = 2

#     # -------------------------------------------------------------------#
#     #   输入通道数 (双时相4波段 = 8)
#     # -------------------------------------------------------------------#
#     input_channels  = 8

#     # -------------------------------------------------------------------#
#     #   模型配置
#     # -------------------------------------------------------------------#
#     phi             = "b5"          # UNet++ 不用这个参数，保留也不影响
#     pretrained      = True          # True 代表加载 encoder 预训练权重
#     model_path      = ""            # 若 Resume 训练则填完整权重路径，否则留空

#     # 训练时的输入尺寸
#     input_shape     = [512, 512]

#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size   = 8
#     UnFreeze_Epoch      = 200
#     Unfreeze_batch_size = 4
#     Freeze_Train        = False

#     # ------------------------------------------------------------------#
#     #   训练参数
#     # ------------------------------------------------------------------#
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
#     save_dir        = 'logs_unet_plusplus'

#     # ------------------------------------------------------------------#
#     #   评估设置
#     # ------------------------------------------------------------------#
#     eval_flag       = True
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node = torch.cuda.device_count()

#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   构建 UNet++ 模型
#     # -------------------------------------------------------------------#
#     model = UNetPlusPlus(num_classes=num_classes, in_chans=input_channels, pretrained=pretrained)

#     # -------------------------------------------------------------------#
#     #   计算静态指标 (FLOPs, Params)
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0
#     if local_rank == 0:
#         print("\n正在计算模型复杂度 (FLOPs & Params)...")
#         # 统一用 512×512 作为 FLOPs 统计基准
#         flops_input_shape = [512, 512]
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, flops_input_shape)
#         print(f"计算基准尺寸: {flops_input_shape}")
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)

#     # -------------------------------------------------------------------#
#     #   加载自定义权重（如果是 Resume 训练）
#     # -------------------------------------------------------------------#
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))

#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location=device)
#         load_key, no_load_key, temp_dict = [], [], {}

#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)

#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)

#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # 记录 Loss
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None

#     if fp16:
#         from torch.cuda.amp import GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train = model.train()

#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(
#                 model_train, device_ids=[local_rank], find_unused_parameters=True
#             )
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()

#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
#         val_lines = f.readlines()

#     num_train = len(train_lines)
#     num_val   = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes=num_classes, phi=phi, model_path=model_path, input_shape=input_shape,
#             Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
#             Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
#             Freeze_Train=Freeze_Train,
#             Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
#             lr_decay_type=lr_decay_type,
#             save_period=save_period, save_dir=save_dir,
#             num_workers=num_workers, num_train=num_train, num_val=num_val
#         )

#     # -------------------------------------------------------------------#
#     #   初始化详细指标 CSV（已加入 clDice 和 CC-F1）
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")

#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch',
#                     'Train Time (min)',
#                     'Params Size (MB)',
#                     'Total Params (10^6)',
#                     'FLOPs (G)',
#                     'mIoU (%)',
#                     'mPA (%)',
#                     'Accuracy (%)',
#                     'F1 Score',
#                     'clDice (%)',
#                     'CC-F1 (%)'
#                 ]
#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     if True:
#         UnFreeze_flag = False

#         if Freeze_Train:
#             try:
#                 # 尝试冻结 backbone / encoder
#                 for param in model.model.encoder.parameters():
#                     param.requires_grad = False
#             except:
#                 print("【注意】无法自动冻结 Backbone，将进行全参数训练。")
#                 pass

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

#         epoch_step     = num_train // batch_size
#         epoch_step_val = num_val // batch_size

#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#         train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset   = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

#         if distributed:
#             train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
#             val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
#             batch_size    = batch_size // ngpus_per_node
#             shuffle       = False
#         else:
#             train_sampler = None
#             val_sampler   = None
#             shuffle       = True

#         gen = DataLoader(
#             train_dataset,
#             shuffle=shuffle,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             pin_memory=True,
#             drop_last=True,
#             collate_fn=seg_dataset_collate,
#             sampler=train_sampler,
#             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#         )

#         gen_val = DataLoader(
#             val_dataset,
#             shuffle=False,   # 验证集固定不打乱
#             batch_size=batch_size,
#             num_workers=num_workers,
#             pin_memory=True,
#             drop_last=False, # 验证集不丢最后一个 batch
#             collate_fn=seg_dataset_collate,
#             sampler=val_sampler,
#             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#         )

#         if local_rank == 0:
#             eval_callback = EvalCallback(
#                 model, input_shape, num_classes, val_lines, VOCdevkit_path,
#                 log_dir, Cuda, eval_flag=eval_flag, period=eval_period
#             )
#         else:
#             eval_callback = None

#         # ---------------------------------------#
#         #   Epoch 循环
#         # ---------------------------------------#
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size

#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

#                 # 解冻 Backbone
#                 try:
#                     for param in model.backbone.parameters():
#                         param.requires_grad = True
#                 except:
#                     for param in model.parameters():
#                         param.requires_grad = True

#                 epoch_step     = num_train // batch_size
#                 epoch_step_val = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#                 gen = DataLoader(
#                     train_dataset,
#                     shuffle=shuffle,
#                     batch_size=batch_size,
#                     num_workers=num_workers,
#                     pin_memory=True,
#                     drop_last=True,
#                     collate_fn=seg_dataset_collate,
#                     sampler=train_sampler,
#                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#                 )

#                 gen_val = DataLoader(
#                     val_dataset,
#                     shuffle=False,
#                     batch_size=batch_size,
#                     num_workers=num_workers,
#                     pin_memory=True,
#                     drop_last=False,
#                     collate_fn=seg_dataset_collate,
#                     sampler=val_sampler,
#                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#                 )

#                 UnFreeze_flag = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(
#                 model_train, model, loss_history, eval_callback, optimizer,
#                 epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch,
#                 Cuda, dice_loss, focal_loss, cls_weights, num_classes,
#                 fp16, scaler, save_period, save_dir, local_rank
#             )

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   每5轮记录一次详细表格并打印（已加入 clDice 和 CC-F1）
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0

#                 # 获取最新计算的指标
#                 current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 # 1. 写入 CSV
#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",
#                         f"{current_ccf1 * 100:.2f}"
#                     ]
#                     writer.writerow(row_data)

#                 # 2. 打印漂亮的表格
#                 table_data = {
#                     "Metric": [
#                         "Epoch",
#                         "Train Time (min)",
#                         "Params Size (MB)",
#                         "Total Params (10^6)",
#                         "FLOPs (G)",
#                         "mIoU (%)",
#                         "mPA (%)",
#                         "Accuracy (%)",
#                         "F1 Score",
#                         "clDice (%)",
#                         "CC-F1 (%)"
#                     ],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "=" * 50)
#                 print(f"Performance Summary at Epoch {epoch + 1}")
#                 print("=" * 50)
#                 print(df.to_string(index=False))
#                 print("=" * 50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()
# DeepLabV3+
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# import copy  # 用于模型深拷贝
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 引入 thop 计算 FLOPs
# try:
#     from thop import profile
# except ImportError:
#     print("【警告】未安装 thop 库，无法自动计算 FLOPs。请运行 pip install thop")
#     profile = None

# # -------------------------------------------------------------------#
# #   导入 DeepLabV3+
# # -------------------------------------------------------------------#
# from nets.deeplabv3plus import DeepLabV3Plus

# from nets.segformer_training import (
#     get_lr_scheduler,
#     set_optimizer_lr,
#     weights_init
# )
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (
#     seed_everything,
#     show_config,
#     worker_init_fn
# )
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   鲁棒的 FLOPs 计算函数 (CPU模式)
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         return 0, 0, 0
    
#     try:
#         model_copy = copy.deepcopy(model).to("cpu").eval()
#     except Exception as e:
#         print(f"【注意】模型深拷贝失败 ({e})，尝试直接使用原模型计算...")
#         model_copy = model.eval().cpu()

#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1]).to("cpu")

#     try:
#         flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
#     except Exception as e:
#         print(f"【错误】FLOPs 计算失败: {e}")
#         return 0, 0, 0
#     finally:
#         if model_copy is not model:
#             del model_copy
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     flops_g = flops / 1e9
#     params_m = params / 1e6
#     params_mb = params * 4 / (1024 * 1024)

#     if next(model.parameters()).is_cuda:
#         model.train().cuda()
#     else:
#         model.train()

#     return flops_g, params_m, params_mb


# if __name__ == "__main__":
#     # ---------------------------------#
#     #   Cuda    是否使用Cuda
#     # ---------------------------------#
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
    
#     # -----------------------------------------------------#
#     #   num_classes     分类个数
#     # -----------------------------------------------------#
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   输入通道数 (8通道)
#     # -------------------------------------------------------------------#
#     input_channels  = 8

#     # -------------------------------------------------------------------#
#     #   模型配置
#     # -------------------------------------------------------------------#
#     phi             = "b5"   # 占位参数
#     pretrained      = True
#     model_path      = ""
    
#     # 训练时的输入尺寸
#     input_shape     = [512, 512]
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # ------------------------------------------------------------------#
#     #   训练参数
#     # ------------------------------------------------------------------#
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = "cos"
#     save_period     = 5
#     save_dir        = "logs_deeplabv3plus_resnest"
    
#     # ------------------------------------------------------------------#
#     #   评估设置
#     # ------------------------------------------------------------------#
#     eval_flag       = True
#     eval_period     = 5

#     VOCdevkit_path  = "VOCdevkit"
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   构建 DeepLabV3+ 模型
#     # -------------------------------------------------------------------#
#     model = DeepLabV3Plus(
#         num_classes=num_classes,
#         in_chans=input_channels,
#         pretrained=pretrained
#     )
    
#     # -------------------------------------------------------------------#
#     #   计算静态指标 (FLOPs, Params)
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0
    
#     if local_rank == 0:
#         print("\n正在计算 DeepLabV3+ 的复杂度...")
#         flops_input_shape = [512, 512]
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, flops_input_shape)
#         print(f"计算基准尺寸: {flops_input_shape}")
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)
        
#     # -------------------------------------------------------------------#
#     #   加载自定义权重 (如果是 Resume 训练)
#     # -------------------------------------------------------------------#
#     if model_path != "":
#         if local_rank == 0:
#             print("Load weights {}.".format(model_path))
        
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location=device)
#         load_key, no_load_key, temp_dict = [], [], {}

#         if "state_dict" in pretrained_dict:
#             pretrained_dict = pretrained_dict["state_dict"]

#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)

#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # 记录 Loss
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(
#                 model_train,
#                 device_ids=[local_rank],
#                 find_unused_parameters=True
#             )
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     # -------------------------------------------------------------------#
#     #   读取数据集
#     # -------------------------------------------------------------------#
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
#         val_lines = f.readlines()

#     num_train = len(train_lines)
#     num_val   = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes=num_classes,
#             phi=phi,
#             model_path=model_path,
#             input_shape=input_shape,
#             Init_Epoch=Init_Epoch,
#             Freeze_Epoch=Freeze_Epoch,
#             UnFreeze_Epoch=UnFreeze_Epoch,
#             Freeze_batch_size=Freeze_batch_size,
#             Unfreeze_batch_size=Unfreeze_batch_size,
#             Freeze_Train=Freeze_Train,
#             Init_lr=Init_lr,
#             Min_lr=Min_lr,
#             optimizer_type=optimizer_type,
#             momentum=momentum,
#             lr_decay_type=lr_decay_type,
#             save_period=save_period,
#             save_dir=save_dir,
#             num_workers=num_workers,
#             num_train=num_train,
#             num_val=num_val
#         )

#     # -------------------------------------------------------------------#
#     #   初始化详细指标 CSV（已加入 clDice / CC-F1）
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 header = [
#                     "Epoch",
#                     "Train Time (min)",
#                     "Params Size (MB)",
#                     "Total Params (10^6)",
#                     "FLOPs (G)",
#                     "mIoU (%)",
#                     "mPA (%)",
#                     "Accuracy (%)",
#                     "F1 Score",
#                     "clDice (%)",
#                     "CC-F1 (%)"
#                 ]
#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     UnFreeze_flag = False
        
#     if Freeze_Train:
#         try:
#             for param in model.backbone.parameters():
#                 param.requires_grad = False
#         except:
#             print("【注意】无法自动冻结 Backbone，将进行全参数训练。")

#     batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#     nbs             = 16
#     lr_limit_max    = 1e-4 if optimizer_type in ["adam", "adamw"] else 5e-2
#     lr_limit_min    = 3e-5 if optimizer_type in ["adam", "adamw"] else 5e-4
#     Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#     Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#     optimizer = {
#         "adam":  optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
#         "adamw": optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
#         "sgd":   optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
#     }[optimizer_type]

#     lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#     epoch_step      = num_train // batch_size
#     epoch_step_val  = max(1, (num_val + batch_size - 1) // batch_size)
        
#     if epoch_step == 0 or epoch_step_val == 0:
#         raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
#     train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#     val_dataset   = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#     if distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
#         val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
#         batch_size    = batch_size // ngpus_per_node
#         train_shuffle = False
#         val_shuffle   = False
#     else:
#         train_sampler = None
#         val_sampler   = None
#         train_shuffle = True
#         val_shuffle   = False

#     gen = DataLoader(
#         train_dataset,
#         shuffle=train_shuffle,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#         collate_fn=seg_dataset_collate,
#         sampler=train_sampler,
#         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#     )

#     gen_val = DataLoader(
#         val_dataset,
#         shuffle=val_shuffle,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False,
#         collate_fn=seg_dataset_collate,
#         sampler=val_sampler,
#         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#     )

#     if local_rank == 0:
#         eval_callback = EvalCallback(
#             model,
#             input_shape,
#             num_classes,
#             val_lines,
#             VOCdevkit_path,
#             log_dir,
#             Cuda,
#             eval_flag=eval_flag,
#             period=eval_period
#         )
#     else:
#         eval_callback = None
        
#     # ---------------------------------------#
#     #   Epoch 循环
#     # ---------------------------------------#
#     for epoch in range(Init_Epoch, UnFreeze_Epoch):
#         epoch_start_time = time.time()

#         if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#             batch_size = Unfreeze_batch_size

#             nbs             = 16
#             lr_limit_max    = 1e-4 if optimizer_type in ["adam", "adamw"] else 5e-2
#             lr_limit_min    = 3e-5 if optimizer_type in ["adam", "adamw"] else 5e-4
#             Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#             Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#             lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#             try:
#                 for param in model.backbone.parameters():
#                     param.requires_grad = True
#             except:
#                 for param in model.parameters():
#                     param.requires_grad = True
                            
#             epoch_step      = num_train // batch_size
#             epoch_step_val  = max(1, (num_val + batch_size - 1) // batch_size)

#             if epoch_step == 0 or epoch_step_val == 0:
#                 raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#             gen = DataLoader(
#                 train_dataset,
#                 shuffle=train_shuffle,
#                 batch_size=batch_size,
#                 num_workers=num_workers,
#                 pin_memory=True,
#                 drop_last=True,
#                 collate_fn=seg_dataset_collate,
#                 sampler=train_sampler,
#                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#             )
#             gen_val = DataLoader(
#                 val_dataset,
#                 shuffle=val_shuffle,
#                 batch_size=batch_size,
#                 num_workers=num_workers,
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=seg_dataset_collate,
#                 sampler=val_sampler,
#                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#             )

#             UnFreeze_flag = True

#         if distributed:
#             train_sampler.set_epoch(epoch)

#         set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#         fit_one_epoch(
#             model_train,
#             model,
#             loss_history,
#             eval_callback,
#             optimizer,
#             epoch,
#             epoch_step,
#             epoch_step_val,
#             gen,
#             gen_val,
#             UnFreeze_Epoch,
#             Cuda,
#             dice_loss,
#             focal_loss,
#             cls_weights,
#             num_classes,
#             fp16,
#             scaler,
#             save_period,
#             save_dir,
#             local_rank
#         )

#         if distributed:
#             dist.barrier()

#         # -------------------------------------------------------------------#
#         #   每5轮记录一次详细表格并打印（已加入 clDice / CC-F1）
#         # -------------------------------------------------------------------#
#         if local_rank == 0 and (epoch + 1) % eval_period == 0:
#             epoch_end_time = time.time()
#             train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#             current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, "mious") and len(eval_callback.mious) > 0 else 0
#             current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, "mPAs") and len(eval_callback.mPAs) > 0 else 0
#             current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, "Accs") and len(eval_callback.Accs) > 0 else 0
#             current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, "f1s") and len(eval_callback.f1s) > 0 else 0
#             current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, "cldices") and len(eval_callback.cldices) > 0 else 0
#             current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, "cc_f1s") and len(eval_callback.cc_f1s) > 0 else 0

#             with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 row_data = [
#                     epoch + 1,
#                     f"{train_time_min:.2f}",
#                     f"{params_mb:.2f}",
#                     f"{params_m:.3f}",
#                     f"{flops_g:.3f}",
#                     f"{current_miou * 100:.2f}",
#                     f"{current_mpa * 100:.2f}",
#                     f"{current_acc * 100:.2f}",
#                     f"{current_f1:.4f}",
#                     f"{current_cldice * 100:.2f}",
#                     f"{current_ccf1 * 100:.2f}"
#                 ]
#                 writer.writerow(row_data)

#             table_data = {
#                 "Metric": [
#                     "Epoch",
#                     "Train Time (min)",
#                     "Params Size (MB)",
#                     "Total Params (10^6)",
#                     "FLOPs (G)",
#                     "mIoU (%)",
#                     "mPA (%)",
#                     "Accuracy (%)",
#                     "F1 Score",
#                     "clDice (%)",
#                     "CC-F1 (%)"
#                 ],
#                 "Value": row_data
#             }
#             df = pd.DataFrame(table_data)
#             print("\n" + "=" * 50)
#             print(f"Performance Summary at Epoch {epoch + 1}")
#             print("=" * 50)
#             print(df.to_string(index=False))
#             print("=" * 50 + "\n")

#     if local_rank == 0 and loss_history is not None:
#         loss_history.writer.close()

# trans-unet
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# import copy  # 【新增】必须导入 copy 库用于模型深拷贝
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 引入 thop 计算 FLOPs
# try:
#     from thop import profile
# except ImportError:
#     print("【警告】未安装 thop 库，无法自动计算 FLOPs。请运行 pip install thop")
#     profile = None

# # -------------------------------------------------------------------#
# #   导入 TransUNet
# # -------------------------------------------------------------------#
# from nets.transunet import TransUNet

# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
#                                      weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config,
#                          worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   鲁棒的 FLOPs 计算函数 (CPU模式)
# #   针对 TransUNet 这种混合架构，强制 CPU 计算最安全
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         return 0, 0, 0
    
#     # 1. 深拷贝模型并转为 CPU、Eval 模式
#     #    TransUNet 的 ViT 部分在 GPU 上初始化可能占用较多显存，
#     #    计算 FLOPs 时移至 CPU 可防止后续训练 OOM。
#     try:
#         model_copy = copy.deepcopy(model).to('cpu').eval()
#     except Exception as e:
#         print(f"【注意】模型深拷贝失败 ({e})，尝试直接使用原模型计算...")
#         model_copy = model.eval().cpu()

#     # 2. 创建 CPU 上的虚拟输入 [1, C, H, W]
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1]).to('cpu')

#     try:
#         # 3. 计算 FLOPs 和 Params
#         flops, params = profile(model_copy, inputs=(dummy_input, ), verbose=False)
#     except Exception as e:
#         print(f"【错误】FLOPs 计算失败: {e}")
#         return 0, 0, 0
#     finally:
#         # 4. 释放临时模型，清理内存
#         if model_copy != model:
#             del model_copy
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # 5. 单位转换
#     flops_g = flops / 1e9              # GFLOPs
#     params_m = params / 1e6            # Million
#     params_mb = params * 4 / (1024 * 1024) # MB (Float32)

#     # 6. 恢复原模型状态 (如果刚才因为 deepcopy 失败移动了原模型)
#     if next(model.parameters()).is_cuda:
#         model.train().cuda()
#     else:
#         model.train()

#     return flops_g, params_m, params_mb

# if __name__ == "__main__":
#     # ---------------------------------#
#     #   Cuda    是否使用Cuda
#     # ---------------------------------#
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
    
#     # -----------------------------------------------------#
#     #   num_classes     分类个数+1
#     # -----------------------------------------------------#
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   【设置】输入通道数 (8通道)
#     # -------------------------------------------------------------------#
#     input_channels  = 8  

#     # -------------------------------------------------------------------#
#     #   模型配置
#     #   TransUNet 通常基于 ViT-B/16 或 R50-ViT-B/16
#     # -------------------------------------------------------------------#
#     phi             = "b5"   # 占位
#     pretrained      = True   # True 代表加载 ImageNet 预训练
#     model_path      = ""     # 设为空，自动下载
    
#     # 训练时的输入尺寸
#     # 注意：TransUNet 对输入尺寸敏感，尽量保持为 16 或 32 的倍数
#     input_shape     = [512, 512]
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # ------------------------------------------------------------------#
#     #   训练参数
#     # ------------------------------------------------------------------#
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
#     save_dir        = 'logs_transunet' 
    
#     # ------------------------------------------------------------------#
#     #   评估设置
#     # ------------------------------------------------------------------#
#     eval_flag       = True 
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   构建 TransUNet 模型
#     # -------------------------------------------------------------------#
#     # 第一次运行会自动下载权重
#     model = TransUNet(num_classes=num_classes, in_chans=input_channels, pretrained=pretrained)
    
#     # -------------------------------------------------------------------#
#     #   【关键修改】计算静态指标 (FLOPs, Params)
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0
    
#     if local_rank == 0:
#         print("\n正在计算 TransUNet 的模型复杂度...")
        
#         # 使用标准 512x512 尺寸计算
#         flops_input_shape = [512, 512]
        
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, flops_input_shape)
        
#         print(f"计算基准尺寸: {flops_input_shape}")
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)
        
#     # -------------------------------------------------------------------#
#     #   加载自定义权重 (如果是 Resume 训练)
#     # -------------------------------------------------------------------#
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
        
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location = device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # 记录 Loss
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
#         else:
#             # 确保模型在 GPU 上
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     # -------------------------------------------------------------------#
#     #   读取数据集
#     # -------------------------------------------------------------------#
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     # -------------------------------------------------------------------#
#     #   初始化详细指标 CSV
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         # 写入表头
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch', 
#                     'Train Time (min)', 
#                     'Params Size (MB)', 
#                     'Total Params (10^6)', 
#                     'FLOPs (G)', 
#                     'mIoU (%)', 
#                     'mPA (%)', 
#                     'Accuracy (%)', 
#                     'F1 Score',
#                     'clDice (%)',
#                     'CC-F1 (%)'
#                 ]
#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     if True:
#         UnFreeze_flag = False
        
#         if Freeze_Train:
#             try:
#                 # TransUNet Backbone 冻结逻辑可能需要根据具体结构调整
#                 # 这里假设 encoder 属性存在
#                 for param in model.model.encoder.parameters():
#                     param.requires_grad = False
#             except:
#                 print("【注意】无法自动冻结 Backbone，将进行全参数训练。")
#                 pass

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                     drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#         gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                     drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#         if local_rank == 0:
#             eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
#                                             eval_flag=eval_flag, period=eval_period)
#         else:
#             eval_callback   = None
        
#         # ---------------------------------------#
#         #   Epoch 循环
#         # ---------------------------------------#
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             # 记录本轮开始时间
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size

#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#                 # 解冻 Backbone
#                 try:
#                     for param in model.backbone.parameters():
#                         param.requires_grad = True
#                 except:
#                     for param in model.parameters():
#                         param.requires_grad = True
                            
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#                 gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                             drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#                 gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                             drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#                 UnFreeze_flag   = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
#                 dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   每5轮记录一次详细表格并打印
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#                 # 获取最新计算的指标
#                 current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 # 1. 写入 CSV
#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",
#                         f"{current_ccf1 * 100:.2f}"
#                     ]
#                     writer.writerow(row_data)

#                 # 2. 打印漂亮的表格
#                 table_data = {
#                     "Metric": [
#                         "Epoch",
#                         "Train Time (min)",
#                         "Params Size (MB)",
#                         "Total Params (10^6)",
#                         "FLOPs (G)",
#                         "mIoU (%)",
#                         "mPA (%)",
#                         "Accuracy (%)",
#                         "F1 Score",
#                         "clDice (%)",
#                         "CC-F1 (%)"
#                     ],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1}")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()

# Swin-Unet

# import datetime
# import os
# import time
# import csv
# import pandas as pd
# import copy  # 【必须】用于深拷贝模型
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # -------------------------------------------------------------------#
# #   【关键修改】尝试导入 ptflops (替代 thop)
# # -------------------------------------------------------------------#
# try:
#     from ptflops import get_model_complexity_info
# except ImportError:
#     get_model_complexity_info = None
#     print("【提示】未安装 ptflops，请运行 pip install ptflops")

# # -------------------------------------------------------------------#
# #   导入 SwinUnet
# # -------------------------------------------------------------------#
# from nets.swin_unet import SwinUnet

# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr, weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config, worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   【修复版】使用 ptflops 计算 SwinUnet 复杂度
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if get_model_complexity_info is None:
#         return 0, 0, 0

#     print(f"【INFO】正在使用 ptflops 计算 FLOPs (输入尺寸: {input_shape})...")
    
#     # 1. 深拷贝模型，防止对训练产生任何副作用
#     try:
#         model_copy = copy.deepcopy(model)
#     except Exception as e:
#         print(f"【注意】模型深拷贝失败({e})，将直接使用原模型计算...")
#         model_copy = model

#     if torch.cuda.is_available():
#         model_copy = model_copy.cuda()
    
#     flops_g, params_m, params_mb = 0, 0, 0

#     try:
#         # ptflops 返回的是 MACs
#         macs, params = get_model_complexity_info(
#             model_copy, 
#             (input_channels, input_shape[0], input_shape[1]), 
#             as_strings=False, 
#             print_per_layer_stat=False, 
#             verbose=False
#         )
        
#         # 1 MAC ≈ 2 FLOPs
#         flops_g = (macs * 2) / 1e9
#         params_m = params / 1e6
#         params_mb = params * 4 / (1024 * 1024)
        
#     except Exception as e:
#         print(f"【警告】ptflops 计算失败: {e}")
#     finally:
#         if model_copy != model:
#             del model_copy
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     return flops_g, params_m, params_mb


# if __name__ == "__main__":
#     # ---------------------------------#
#     #   Cuda    是否使用Cuda
#     # ---------------------------------#
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   输入通道数 (8通道)
#     # -------------------------------------------------------------------#
#     input_channels  = 8  

#     # -------------------------------------------------------------------#
#     #   模型配置
#     # -------------------------------------------------------------------#
#     phi             = "swin_tiny" 
#     pretrained      = True   
#     model_path      = ""     
    
#     # -------------------------------------------------------------------#
#     #   SwinUnet 输入尺寸限制
#     # -------------------------------------------------------------------#
#     input_shape     = [448, 448]
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # 训练参数
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
    
#     save_dir        = 'logs_swin_unet' 
    
#     eval_flag       = True 
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   初始化 SwinUnet 模型
#     # -------------------------------------------------------------------#
#     model = SwinUnet(num_classes=num_classes, in_chans=input_channels, pretrained=pretrained)
    
#     # -------------------------------------------------------------------#
#     #   计算复杂度 (使用 ptflops)
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0
    
#     if local_rank == 0:
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
        
#         print("-" * 50)
#         print(f"SwinUnet 复杂度报告 (Input: {input_shape}):")
#         print(f"FLOPs:       {flops_g:.3f} G")
#         print(f"Params:      {params_m:.3f} M")
#         print(f"Weight Size: {params_mb:.2f} MB")
#         print("-" * 50 + "\n")

#     if not pretrained:
#         weights_init(model)
        
#     # -------------------------------------------------------------------#
#     #   加载权重
#     # -------------------------------------------------------------------#
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
        
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location = device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # 记录 Loss
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     # -------------------------------------------------------------------#
#     #   读取数据集
#     # -------------------------------------------------------------------#
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     # -------------------------------------------------------------------#
#     #   CSV 表格记录
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch',
#                     'Train Time (min)',
#                     'Params Size (MB)',
#                     'Total Params (10^6)', 
#                     'FLOPs (G)',
#                     'mIoU (%)',
#                     'mPA (%)',
#                     'Accuracy (%)',
#                     'F1 Score',
#                     'clDice (%)',
#                     'CC-F1 (%)'
#                 ]
#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     if True:
#         UnFreeze_flag = False
        
#         if Freeze_Train:
#             try:
#                 if hasattr(model, "swin_unet"):
#                     for param in model.swin_unet.parameters():
#                         param.requires_grad = False
#                 elif hasattr(model, "backbone"):
#                     for param in model.backbone.parameters():
#                         param.requires_grad = False
#             except:
#                 print("【注意】无法自动冻结 Backbone，将进行全参数训练。")
#                 pass

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                     drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#         gen_val         = DataLoader(val_dataset, shuffle = False, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                     drop_last = False, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#         if local_rank == 0:
#             eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
#                                             eval_flag=eval_flag, period=eval_period)
#         else:
#             eval_callback   = None
        
#         # ---------------------------------------#
#         #   Epoch 循环
#         # ---------------------------------------#
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size

#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#                 try:
#                     if hasattr(model, "swin_unet"):
#                         for param in model.swin_unet.parameters():
#                             param.requires_grad = True
#                     else:
#                         for param in model.parameters():
#                             param.requires_grad = True
#                 except:
#                     for param in model.parameters():
#                         param.requires_grad = True
                            
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#                 gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                             drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#                 gen_val         = DataLoader(val_dataset, shuffle = False, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                             drop_last = False, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#                 UnFreeze_flag   = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
#                 dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   每5轮记录一次详细表格并打印
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#                 current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 # 1. 写入 CSV
#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",
#                         f"{current_ccf1 * 100:.2f}"
#                     ]
#                     writer.writerow(row_data)

#                 # 2. 打印漂亮的表格
#                 table_data = {
#                     "Metric": [
#                         "Epoch",
#                         "Train Time (min)",
#                         "Params Size (MB)",
#                         "Total Params (10^6)",
#                         "FLOPs (G)",
#                         "mIoU (%)",
#                         "mPA (%)",
#                         "Accuracy (%)",
#                         "F1 Score",
#                         "clDice (%)",
#                         "CC-F1 (%)"
#                     ],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1}")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()

#ResUnet 
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# import copy  # 【必须】用于深拷贝模型
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # -------------------------------------------------------------------#
# #   【关键修改】尝试导入 ptflops (替代 thop)
# # -------------------------------------------------------------------#
# try:
#     from ptflops import get_model_complexity_info
# except ImportError:
#     get_model_complexity_info = None
#     print("【提示】未安装 ptflops，请运行 pip install ptflops")

# # -------------------------------------------------------------------#
# #   导入 SwinUnet
# # -------------------------------------------------------------------#
# from nets.swin_unet import SwinUnet

# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr, weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config, worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   【修复版】使用 ptflops 计算 SwinUnet 复杂度
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if get_model_complexity_info is None:
#         return 0, 0, 0

#     print(f"【INFO】正在使用 ptflops 计算 FLOPs (输入尺寸: {input_shape})...")
    
#     # 1. 深拷贝模型，防止对训练产生任何副作用
#     try:
#         model_copy = copy.deepcopy(model)
#     except Exception as e:
#         print(f"【注意】模型深拷贝失败({e})，将直接使用原模型计算...")
#         model_copy = model

#     if torch.cuda.is_available():
#         model_copy = model_copy.cuda()
    
#     flops_g, params_m, params_mb = 0, 0, 0

#     try:
#         # ptflops 返回的是 MACs
#         macs, params = get_model_complexity_info(
#             model_copy, 
#             (input_channels, input_shape[0], input_shape[1]), 
#             as_strings=False, 
#             print_per_layer_stat=False, 
#             verbose=False
#         )
        
#         # 1 MAC ≈ 2 FLOPs
#         flops_g = (macs * 2) / 1e9
#         params_m = params / 1e6
#         params_mb = params * 4 / (1024 * 1024)
        
#     except Exception as e:
#         print(f"【警告】ptflops 计算失败: {e}")
#     finally:
#         if model_copy != model:
#             del model_copy
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     return flops_g, params_m, params_mb


# if __name__ == "__main__":
#     # ---------------------------------#
#     #   Cuda    是否使用Cuda
#     # ---------------------------------#
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   输入通道数 (8通道)
#     # -------------------------------------------------------------------#
#     input_channels  = 8  

#     # -------------------------------------------------------------------#
#     #   模型配置
#     # -------------------------------------------------------------------#
#     phi             = "swin_tiny" 
#     pretrained      = True   
#     model_path      = ""     
    
#     # -------------------------------------------------------------------#
#     #   SwinUnet 输入尺寸限制
#     # -------------------------------------------------------------------#
#     input_shape     = [448, 448]
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # 训练参数
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
    
#     save_dir        = 'logs_swin_unet' 
    
#     eval_flag       = True 
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   初始化 SwinUnet 模型
#     # -------------------------------------------------------------------#
#     model = SwinUnet(num_classes=num_classes, in_chans=input_channels, pretrained=pretrained)
    
#     # -------------------------------------------------------------------#
#     #   计算复杂度 (使用 ptflops)
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0
    
#     if local_rank == 0:
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
        
#         print("-" * 50)
#         print(f"SwinUnet 复杂度报告 (Input: {input_shape}):")
#         print(f"FLOPs:       {flops_g:.3f} G")
#         print(f"Params:      {params_m:.3f} M")
#         print(f"Weight Size: {params_mb:.2f} MB")
#         print("-" * 50 + "\n")

#     if not pretrained:
#         weights_init(model)
        
#     # -------------------------------------------------------------------#
#     #   加载权重
#     # -------------------------------------------------------------------#
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
        
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location = device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # 记录 Loss
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     # -------------------------------------------------------------------#
#     #   读取数据集
#     # -------------------------------------------------------------------#
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     # -------------------------------------------------------------------#
#     #   CSV 表格记录
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch',
#                     'Train Time (min)',
#                     'Params Size (MB)',
#                     'Total Params (10^6)', 
#                     'FLOPs (G)',
#                     'mIoU (%)',
#                     'mPA (%)',
#                     'Accuracy (%)',
#                     'F1 Score',
#                     'clDice (%)',
#                     'CC-F1 (%)'
#                 ]
#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     if True:
#         UnFreeze_flag = False
        
#         if Freeze_Train:
#             try:
#                 if hasattr(model, "swin_unet"):
#                     for param in model.swin_unet.parameters():
#                         param.requires_grad = False
#                 elif hasattr(model, "backbone"):
#                     for param in model.backbone.parameters():
#                         param.requires_grad = False
#             except:
#                 print("【注意】无法自动冻结 Backbone，将进行全参数训练。")
#                 pass

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                     drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#         gen_val         = DataLoader(val_dataset, shuffle = False, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                     drop_last = False, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#         if local_rank == 0:
#             eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
#                                             eval_flag=eval_flag, period=eval_period)
#         else:
#             eval_callback   = None
        
#         # ---------------------------------------#
#         #   Epoch 循环
#         # ---------------------------------------#
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size

#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#                 try:
#                     if hasattr(model, "swin_unet"):
#                         for param in model.swin_unet.parameters():
#                             param.requires_grad = True
#                     else:
#                         for param in model.parameters():
#                             param.requires_grad = True
#                 except:
#                     for param in model.parameters():
#                         param.requires_grad = True
                            
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#                 gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
#                                             drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#                 gen_val         = DataLoader(val_dataset, shuffle = False, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
#                                             drop_last = False, collate_fn = seg_dataset_collate, sampler=val_sampler, 
#                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#                 UnFreeze_flag   = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
#                 dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   每5轮记录一次详细表格并打印
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#                 current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 # 1. 写入 CSV
#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",
#                         f"{current_ccf1 * 100:.2f}"
#                     ]
#                     writer.writerow(row_data)

#                 # 2. 打印漂亮的表格
#                 table_data = {
#                     "Metric": [
#                         "Epoch",
#                         "Train Time (min)",
#                         "Params Size (MB)",
#                         "Total Params (10^6)",
#                         "FLOPs (G)",
#                         "mIoU (%)",
#                         "mPA (%)",
#                         "Accuracy (%)",
#                         "F1 Score",
#                         "clDice (%)",
#                         "CC-F1 (%)"
#                     ],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1}")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()


# DeeplabV3+
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # -------------------------------------------------------------------#
# #   尝试导入 thop 用于计算 FLOPs
# # -------------------------------------------------------------------#
# try:
#     from thop import profile
# except ImportError:
#     profile = None

# # -------------------------------------------------------------------#
# #   导入 DeepLabV3+
# #   假设您的模型文件在 nets/deeplabv3_plus.py，类名为 DeepLab
# # -------------------------------------------------------------------#
# from nets.deeplabv3_plus import DeepLab

# # 直接借用现有的 segformer_training 里的通用函数
# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr, weights_init)

# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config, worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   辅助函数：计算 Params 和 FLOPs
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         print("【提示】未检测到 thop 库，跳过 FLOPs 计算 (pip install thop)")
#         return 0, 0, 0
    
#     model_eval = model.eval()
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1])
    
#     if torch.cuda.is_available():
#         dummy_input = dummy_input.cuda()
#         model_eval = model_eval.cuda()

#     try:
#         flops, params = profile(model_eval, inputs=(dummy_input, ), verbose=False)
#     except Exception as e:
#         print(f"【警告】FLOPs 计算失败: {e}")
#         return 0, 0, 0
    
#     model.train()
    
#     flops_g = flops / 1e9
#     params_m = params / 1e6
#     params_mb = params * 4 / (1024 * 1024)
    
#     return flops_g, params_m, params_mb

# if __name__ == "__main__":
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   输入通道数
#     # -------------------------------------------------------------------#
#     input_channels  = 8

#     # -------------------------------------------------------------------#
#     #   DeepLabV3+ 配置
#     # -------------------------------------------------------------------#
#     backbone          = "resnet50"
#     downsample_factor = 16
#     pretrained        = True
#     model_path        = ""
    
#     input_shape     = [512, 512]
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # 训练参数
#     Init_lr         = 7e-3
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "sgd"
#     momentum        = 0.9
#     weight_decay    = 1e-4
#     lr_decay_type   = 'cos'
#     save_period     = 5
    
#     # -------------------------------------------------------------------#
#     #   保存路径
#     # -------------------------------------------------------------------#
#     save_dir        = 'logs_deeplabv3plus_mobilenet'
    
#     eval_flag       = True
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   构建 DeepLabV3+ 模型
#     # -------------------------------------------------------------------#
#     # 注意：确保您的 DeepLab 类支持 input_channels 参数。
#     model = DeepLab(
#         num_classes=num_classes,
#         backbone=backbone,
#         downsample_factor=downsample_factor,
#         pretrained=pretrained,
#         input_channels=input_channels
#     )
    
#     # 计算复杂度
#     flops_g, params_m, params_mb = 0, 0, 0
#     if local_rank == 0:
#         print(f"\n正在计算 DeepLabV3+ ({backbone}) 模型复杂度...")
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)
        
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location=device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # Loss 记录
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(
#                 model_train, device_ids=[local_rank], find_unused_parameters=True
#             )
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = backbone, model_path = model_path, input_shape = input_shape, \
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     # -------------------------------------------------------------------#
#     #   CSV 表格初始化
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch',
#                     'Train Time (min)',
#                     'Params Size (MB)',
#                     'Total Params (10^6)',
#                     'FLOPs (G)',
#                     'mIoU (%)',
#                     'mPA (%)',
#                     'Accuracy (%)',
#                     'F1 Score',
#                     'clDice (%)',
#                     'CC-F1 (%)'
#                 ]
#                 writer.writerow(header)

#     if True:
#         UnFreeze_flag = False
        
#         if Freeze_Train:
#             try:
#                 # DeepLab 的 backbone 属性通常就叫 backbone
#                 for param in model.backbone.parameters():
#                     param.requires_grad = False
#             except:
#                 pass

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen = DataLoader(
#             train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#             drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
#             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#         )
#         gen_val = DataLoader(
#             val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#             drop_last=False, collate_fn=seg_dataset_collate, sampler=val_sampler,
#             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#         )

#         if local_rank == 0:
#             eval_callback = EvalCallback(
#                 model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
#                 eval_flag=eval_flag, period=eval_period
#             )
#         else:
#             eval_callback = None
        
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size
#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#                 try:
#                     for param in model.backbone.parameters():
#                         param.requires_grad = True
#                 except:
#                     for param in model.parameters():
#                         param.requires_grad = True
                            
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，请扩充数据集。")

#                 gen = DataLoader(
#                     train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#                     drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
#                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#                 )
#                 gen_val = DataLoader(
#                     val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#                     drop_last=False, collate_fn=seg_dataset_collate, sampler=val_sampler,
#                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#                 )

#                 UnFreeze_flag   = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(
#                 model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
#                 gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes,
#                 fp16, scaler, save_period, save_dir, local_rank
#             )

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   表格输出
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#                 current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",
#                         f"{current_ccf1 * 100:.2f}"
#                     ]
#                     writer.writerow(row_data)

#                 table_data = {
#                     "Metric": [
#                         "Epoch",
#                         "Train Time (min)",
#                         "Params Size (MB)",
#                         "Total Params (10^6)",
#                         "FLOPs (G)",
#                         "mIoU (%)",
#                         "mPA (%)",
#                         "Accuracy (%)",
#                         "F1 Score",
#                         "clDice (%)",
#                         "CC-F1 (%)"
#                     ],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1} (DeepLabV3+)")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()


# MCANet

# import datetime
# import os
# import time
# import csv
# import pandas as pd
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # -------------------------------------------------------------------#
# #   尝试导入 thop 用于计算 FLOPs
# # -------------------------------------------------------------------#
# try:
#     from thop import profile
# except ImportError:
#     profile = None

# # -------------------------------------------------------------------#
# #   导入 MCANet
# # -------------------------------------------------------------------#
# from nets.mcanet import MCANet

# # 使用通用的辅助函数
# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr, weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config, worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   辅助函数：计算 Params 和 FLOPs
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         print("【提示】未检测到 thop 库，跳过 FLOPs 计算 (pip install thop)")
#         return 0, 0, 0
    
#     model_eval = model.eval()
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1])
    
#     if torch.cuda.is_available():
#         dummy_input = dummy_input.cuda()
#         model_eval = model_eval.cuda()

#     try:
#         flops, params = profile(model_eval, inputs=(dummy_input,), verbose=False)
#     except Exception as e:
#         print(f"【警告】FLOPs 计算失败: {e}")
#         return 0, 0, 0
    
#     model.train()
    
#     flops_g = flops / 1e9
#     params_m = params / 1e6
#     params_mb = params * 4 / (1024 * 1024)
    
#     return flops_g, params_m, params_mb

# if __name__ == "__main__":
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   输入配置
#     # -------------------------------------------------------------------#
#     input_channels  = 8
#     input_shape     = [512, 512]

#     # -------------------------------------------------------------------#
#     #   MCANet 配置
#     # -------------------------------------------------------------------#
#     pretrained      = True
#     model_path      = ""  # 第一次训练设为空，自动下载 ResNet 权重
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # 训练参数
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
    
#     # -------------------------------------------------------------------#
#     #   保存路径
#     # -------------------------------------------------------------------#
#     save_dir        = 'logs_mcanet_resnet50'
    
#     eval_flag       = True
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   构建 MCANet
#     # -------------------------------------------------------------------#
#     model = MCANet(num_classes=num_classes, in_chans=input_channels, pretrained=pretrained)
    
#     # 计算复杂度
#     flops_g, params_m, params_mb = 0, 0, 0
#     if local_rank == 0:
#         print(f"\n正在计算 MCANet (ResNet50) 模型复杂度...")
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)
        
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location=device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # Loss 记录
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(
#                 model_train, device_ids=[local_rank], find_unused_parameters=True
#             )
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = "resnet50", model_path = model_path, input_shape = input_shape,
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch,
#             Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train,
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type,
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     # -------------------------------------------------------------------#
#     #   CSV 表格初始化
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch',
#                     'Train Time (min)',
#                     'Params Size (MB)',
#                     'Total Params (10^6)',
#                     'FLOPs (G)',
#                     'mIoU (%)',
#                     'mPA (%)',
#                     'Accuracy (%)',
#                     'F1 Score',
#                     'clDice (%)',
#                     'CC-F1 (%)'
#                 ]
#                 writer.writerow(header)

#     if True:
#         UnFreeze_flag = False
        
#         if Freeze_Train:
#             try:
#                 # 尝试冻结 backbone
#                 for param in model.model.encoder.parameters():
#                     param.requires_grad = False
#             except:
#                 pass

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen = DataLoader(
#             train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#             drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
#             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#         )
#         gen_val = DataLoader(
#             val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#             drop_last=False, collate_fn=seg_dataset_collate, sampler=val_sampler,
#             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#         )

#         if local_rank == 0:
#             eval_callback = EvalCallback(
#                 model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
#                 eval_flag=eval_flag, period=eval_period
#             )
#         else:
#             eval_callback = None
        
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size
#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#                 try:
#                     for param in model.model.encoder.parameters():
#                         param.requires_grad = True
#                 except:
#                     for param in model.parameters():
#                         param.requires_grad = True
                            
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，请扩充数据集。")

#                 gen = DataLoader(
#                     train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#                     drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
#                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#                 )
#                 gen_val = DataLoader(
#                     val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#                     drop_last=False, collate_fn=seg_dataset_collate, sampler=val_sampler,
#                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#                 )

#                 UnFreeze_flag = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(
#                 model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
#                 gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes,
#                 fp16, scaler, save_period, save_dir, local_rank
#             )

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   表格输出：MCANet
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#                 current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",
#                         f"{current_ccf1 * 100:.2f}"
#                     ]
#                     writer.writerow(row_data)

#                 table_data = {
#                     "Metric": [
#                         "Epoch",
#                         "Train Time (min)",
#                         "Params Size (MB)",
#                         "Total Params (10^6)",
#                         "FLOPs (G)",
#                         "mIoU (%)",
#                         "mPA (%)",
#                         "Accuracy (%)",
#                         "F1 Score",
#                         "clDice (%)",
#                         "CC-F1 (%)"
#                     ],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1} (MCANet)")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()
# maxvit
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # -------------------------------------------------------------------#
# #   尝试导入 thop 用于计算 FLOPs
# # -------------------------------------------------------------------#
# try:
#     from thop import profile
# except ImportError:
#     profile = None

# # -------------------------------------------------------------------#
# #   导入 MaxViT
# # -------------------------------------------------------------------#
# from nets.maxvit_seg import MaxViTSeg

# # 使用通用的辅助函数
# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr, weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config, worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   辅助函数：计算 Params 和 FLOPs
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         print("【提示】未检测到 thop 库，跳过 FLOPs 计算 (pip install thop)")
#         return 0, 0, 0
    
#     model_eval = model.eval()
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1])
    
#     if torch.cuda.is_available():
#         dummy_input = dummy_input.cuda()
#         model_eval = model_eval.cuda()

#     try:
#         flops, params = profile(model_eval, inputs=(dummy_input,), verbose=False)
#     except Exception as e:
#         print(f"【警告】FLOPs 计算失败: {e}")
#         return 0, 0, 0
    
#     model.train()
    
#     flops_g = flops / 1e9
#     params_m = params / 1e6
#     params_mb = params * 4 / (1024 * 1024)
    
#     return flops_g, params_m, params_mb

# if __name__ == "__main__":
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
#     num_classes     = 2
    
#     # -------------------------------------------------------------------#
#     #   输入配置
#     # -------------------------------------------------------------------#
#     input_channels  = 8
#     input_shape     = [512, 512]

#     # -------------------------------------------------------------------#
#     #   MaxViT 配置
#     # -------------------------------------------------------------------#
#     pretrained      = True
#     model_path      = ""  # 保持为空，会自动从 huggingface/timm 下载
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # 训练参数 (MaxViT 推荐使用 AdamW)
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
    
#     # -------------------------------------------------------------------#
#     #   保存路径
#     # -------------------------------------------------------------------#
#     save_dir        = 'logs_maxvit_unet'
    
#     eval_flag       = True
#     eval_period     = 5

#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   构建 MaxViT-Unet
#     # -------------------------------------------------------------------#
#     model = MaxViTSeg(num_classes=num_classes, in_chans=input_channels, pretrained=pretrained)
    
#     # 计算复杂度
#     flops_g, params_m, params_mb = 0, 0, 0
#     if local_rank == 0:
#         print(f"\n正在计算 MaxViT-Tiny 模型复杂度...")
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)
        
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location=device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
        
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # Loss 记录
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = "maxvit_tiny", model_path = model_path, input_shape = input_shape, \
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     # -------------------------------------------------------------------#
#     #   CSV 表格初始化
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch',
#                     'Train Time (min)',
#                     'Params Size (MB)',
#                     'Total Params (10^6)',
#                     'FLOPs (G)',
#                     'mIoU (%)',
#                     'mPA (%)',
#                     'Accuracy (%)',
#                     'F1 Score',
#                     'clDice (%)',
#                     'CC-F1 (%)'
#                 ]
#                 writer.writerow(header)

#     if True:
#         UnFreeze_flag = False
        
#         if Freeze_Train:
#             try:
#                 # 尝试冻结 backbone
#                 for param in model.model.encoder.parameters():
#                     param.requires_grad = False
#             except:
#                 pass

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen = DataLoader(
#             train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#             drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
#             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#         )
#         gen_val = DataLoader(
#             val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#             drop_last=False, collate_fn=seg_dataset_collate, sampler=val_sampler,
#             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#         )

#         if local_rank == 0:
#             eval_callback = EvalCallback(
#                 model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
#                 eval_flag=eval_flag, period=eval_period
#             )
#         else:
#             eval_callback = None
        
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size
#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
#                 try:
#                     for param in model.model.encoder.parameters():
#                         param.requires_grad = True
#                 except:
#                     for param in model.parameters():
#                         param.requires_grad = True
                            
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，请扩充数据集。")

#                 gen = DataLoader(
#                     train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#                     drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
#                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#                 )
#                 gen_val = DataLoader(
#                     val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
#                     drop_last=False, collate_fn=seg_dataset_collate, sampler=val_sampler,
#                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#                 )

#                 UnFreeze_flag = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(
#                 model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
#                 gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes,
#                 fp16, scaler, save_period, save_dir, local_rank
#             )

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   表格输出：MaxViT
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
#                 current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",
#                         f"{current_ccf1 * 100:.2f}"
#                     ]
#                     writer.writerow(row_data)

#                 table_data = {
#                     "Metric": [
#                         "Epoch",
#                         "Train Time (min)",
#                         "Params Size (MB)",
#                         "Total Params (10^6)",
#                         "FLOPs (G)",
#                         "mIoU (%)",
#                         "mPA (%)",
#                         "Accuracy (%)",
#                         "F1 Score",
#                         "clDice (%)",
#                         "CC-F1 (%)"
#                     ],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1} (MaxViT)")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()

# convnext_seg
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # -------------------------------------------------------------------#
# #   尝试导入 thop
# # -------------------------------------------------------------------#
# try:
#     from thop import profile
# except ImportError:
#     profile = None

# # -------------------------------------------------------------------#
# #   【修改点1】导入 ConvNeXtSeg
# # -------------------------------------------------------------------#
# from nets.convnext_seg import ConvNeXtSeg

# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr, weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config, worker_init_fn)
# from utils.utils_fit import fit_one_epoch

# # -------------------------------------------------------------------#
# #   辅助函数
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         print("【提示】未检测到 thop 库，跳过 FLOPs 计算")
#         return 0, 0, 0
#     model_eval = model.eval()
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1])
#     if torch.cuda.is_available():
#         dummy_input = dummy_input.cuda()
#         model_eval = model_eval.cuda()
#     try:
#         flops, params = profile(model_eval, inputs=(dummy_input, ), verbose=False)
#     except Exception as e:
#         print(f"【警告】FLOPs 计算失败: {e}")
#         return 0, 0, 0
#     model.train()
#     return flops / 1e9, params / 1e6, params * 4 / (1024 * 1024)

# if __name__ == "__main__":
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False
#     num_classes     = 2
    
#     input_channels  = 8   
#     input_shape     = [512, 512]

#     # -------------------------------------------------------------------#
#     #   【修改点2】ConvNeXt V2 配置
#     #   phi = "tiny"  -> 对标 SegFormer-B3 / MaxViT-Tiny (~28M)
#     #   phi = "base"  -> 对标 SegFormer-B5 / MaxViT-Base (~89M)
#     # -------------------------------------------------------------------#
#     phi             = "small"  
#     pretrained      = True   
#     model_path      = "" 
    
#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4
    
#     Freeze_Train    = False

#     # ConvNeXt 推荐使用 AdamW
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = 'cos'
#     save_period     = 5
    
#     # -------------------------------------------------------------------#
#     #   【修改点3】保存路径
#     # -------------------------------------------------------------------#
#     save_dir        = f'logs_convnext_{phi}_unet' 
    
#     eval_flag       = True 
#     eval_period     = 5
#     VOCdevkit_path  = 'VOCdevkit'
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()
    
#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   【修改点4】构建模型
#     # -------------------------------------------------------------------#
#     model = ConvNeXtSeg(num_classes=num_classes, in_chans=input_channels, phi=phi, pretrained=pretrained)
    
#     flops_g, params_m, params_mb = 0, 0, 0
#     if local_rank == 0:
#         print(f"\n正在计算 ConvNeXt V2 ({phi}) 模型复杂度...")
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     if not pretrained:
#         weights_init(model)
        
#     if model_path != '':
#         if local_rank == 0:
#             print('Load weights {}.'.format(model_path))
#         model_dict = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location = device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None
        
#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train     = model.train()
    
#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()
    
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
#         val_lines = f.readlines()
#     num_train   = len(train_lines)
#     num_val     = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes = num_classes, phi = f"convnext_{phi}", model_path = model_path, input_shape = input_shape, \
#             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
#             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
#             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
#         )

#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = ['Epoch', 'Train Time (min)', 'Params Size (MB)', 'Total Params (10^6)', 'FLOPs (G)', 'mIoU (%)', 'mPA (%)', 'Accuracy (%)', 'F1 Score']
#                 writer.writerow(header)

#     if True:
#         UnFreeze_flag = False
#         if Freeze_Train:
#             try:
#                 for param in model.model.encoder.parameters():
#                     param.requires_grad = False
#             except:
#                 pass

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
#         nbs             = 16
#         lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
#             'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
#         epoch_step      = num_train // batch_size
#         epoch_step_val  = num_val // batch_size
        
#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，请扩充数据集。")
        
#         train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
#         if distributed:
#             train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
#             val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
#             batch_size      = batch_size // ngpus_per_node
#             shuffle         = False
#         else:
#             train_sampler   = None
#             val_sampler     = None
#             shuffle         = True

#         gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#         gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

#         if local_rank == 0:
#             eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, eval_flag=eval_flag, period=eval_period)
#         else:
#             eval_callback   = None
        
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             epoch_start_time = time.time()
#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size
#                 nbs             = 16
#                 lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
#                 try:
#                     for param in model.model.encoder.parameters():
#                         param.requires_grad = True
#                 except:
#                     for param in model.parameters():
#                         param.requires_grad = True
#                 epoch_step      = num_train // batch_size
#                 epoch_step_val  = num_val // batch_size
#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，请扩充数据集。")
#                 gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#                 gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
#                 UnFreeze_flag   = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)
#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
#             fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)
#             if distributed:
#                 dist.barrier()

#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0
#                 current_miou = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa  = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc  = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1   = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0

#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [epoch + 1, f"{train_time_min:.2f}", f"{params_mb:.2f}", f"{params_m:.3f}", f"{flops_g:.3f}", f"{current_miou * 100:.2f}", f"{current_mpa * 100:.2f}", f"{current_acc * 100:.2f}", f"{current_f1:.4f}"]
#                     writer.writerow(row_data)
#                 table_data = {"Metric": ["Epoch", "Train Time (min)", "Params Size (MB)", "Total Params (10^6)", "FLOPs (G)", "mIoU (%)", "mPA (%)", "Accuracy (%)", "F1 Score"], "Value": row_data}
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "="*50)
#                 print(f"Performance Summary at Epoch {epoch + 1} (ConvNeXt V2)")
#                 print("="*50)
#                 print(df.to_string(index=False))
#                 print("="*50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()







# import datetime
# import math
# import os
# import time
# import csv
# import pandas as pd
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 引入 thop 计算 FLOPs (如果没有请 pip install thop)
# try:
#     from thop import profile
# except ImportError:
#     print("【警告】未安装 thop 库，无法自动计算 FLOPs。请运行 pip install thop")
#     profile = None

# from nets.segformer import SegFormer
# from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
#                                      weights_init)
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (download_weights, seed_everything, show_config,
#                          worker_init_fn)
# from utils.utils_fit import fit_one_epoch


# # -------------------------------------------------------------------#
# #   辅助函数：计算 Params 和 FLOPs
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         return 0, 0, 0

#     was_training = model.training
#     model_eval = model.eval()
#     dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1])

#     if torch.cuda.is_available():
#         dummy_input = dummy_input.cuda()
#         model_eval = model_eval.cuda()

#     try:
#         flops, params = profile(model_eval, inputs=(dummy_input,), verbose=False)
#     except Exception as e:
#         print(f"计算 FLOPs 失败: {e}")
#         if was_training:
#             model.train()
#         return 0, 0, 0

#     if was_training:
#         model.train()

#     flops_g = flops / 1e9
#     params_m = params / 1e6
#     params_mb = params * 4 / (1024 * 1024)
#     return flops_g, params_m, params_mb


# # -------------------------------------------------------------------#
# #   辅助函数：判断权重类型并加载
# # -------------------------------------------------------------------#
# def is_backbone_checkpoint(model_path: str) -> bool:
#     if not model_path:
#         return False
#     filename = os.path.basename(model_path).lower()
#     return "backbone" in filename


# def load_full_model_weights(model, model_path, device, local_rank=0):
#     if local_rank == 0:
#         print(f'Load full model weights from {model_path}.')

#     model_dict = model.state_dict()
#     pretrained_dict = torch.load(model_path, map_location=device)

#     if isinstance(pretrained_dict, dict):
#         if "state_dict" in pretrained_dict:
#             pretrained_dict = pretrained_dict["state_dict"]
#         elif "model" in pretrained_dict:
#             pretrained_dict = pretrained_dict["model"]

#     cleaned_dict = {}
#     for k, v in pretrained_dict.items():
#         nk = k
#         if nk.startswith("module."):
#             nk = nk[len("module."):]
#         cleaned_dict[nk] = v

#     load_key, no_load_key, temp_dict = [], [], {}
#     for k, v in cleaned_dict.items():
#         if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
#             temp_dict[k] = v
#             load_key.append(k)
#         else:
#             no_load_key.append(k)

#     model_dict.update(temp_dict)
#     msg = model.load_state_dict(model_dict, strict=False)

#     if local_rank == 0:
#         print("\nSuccessful Load Key Num:", len(load_key))
#         print("Fail To Load Key Num:", len(no_load_key))
#         if msg.missing_keys:
#             print("Missing keys after full-model load:", msg.missing_keys)
#         if msg.unexpected_keys:
#             print("Unexpected keys after full-model load:", msg.unexpected_keys)


# def build_dataloader(dataset, batch_size, shuffle, num_workers, rank, seed, sampler=None, drop_last=False):
#     return DataLoader(
#         dataset,
#         shuffle=shuffle if sampler is None else False,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#         drop_last=drop_last,
#         collate_fn=seg_dataset_collate,
#         sampler=sampler,
#         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
#         persistent_workers=True if num_workers > 0 else False,
#     )


# if __name__ == "__main__":
#     # ---------------------------------#
#     #   Cuda    是否使用Cuda
#     # ---------------------------------#
#     Cuda = True
#     seed = 11
#     distributed = False
#     sync_bn = False
#     fp16 = False

#     # -----------------------------------------------------#
#     #   num_classes     分类个数（背景+目标时，二分类填 2）
#     # -----------------------------------------------------#
#     num_classes = 2

#     # -------------------------------------------------------------------#
#     #   输入通道数 (双时相4波段 = 8)
#     # -------------------------------------------------------------------#
#     input_channels = 8

#     phi = "b5"
#     pretrained = True
#     model_path = "model_data/segformer_b5_backbone_weights.pth"

#     input_shape = [512, 512]

#     Init_Epoch = 0
#     Freeze_Epoch = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch = 200
#     Unfreeze_batch_size = 4

#     Freeze_Train = False

#     # ------------------------------------------------------------------#
#     #   训练参数
#     # ------------------------------------------------------------------#
#     Init_lr = 1e-4
#     Min_lr = Init_lr * 0.01
#     optimizer_type = "adamw"
#     momentum = 0.9
#     weight_decay = 1e-2
#     lr_decay_type = 'cos'
#     save_period = 5
#     save_dir = 'logs'

#     # ------------------------------------------------------------------#
#     #   评估设置
#     # ------------------------------------------------------------------#
#     eval_flag = True
#     eval_period = 5

#     VOCdevkit_path = 'VOCdevkit'
#     dice_loss = True
#     focal_loss = False
#     cls_weights = np.ones([num_classes], np.float32)
#     num_workers = 4

#     seed_everything(seed)
#     ngpus_per_node = torch.cuda.device_count()

#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank = int(os.environ["LOCAL_RANK"])
#         rank = int(os.environ["RANK"])
#         device = torch.device("cuda", local_rank)
#     else:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         local_rank = 0
#         rank = 0

#     backbone_ckpt = is_backbone_checkpoint(model_path)
#     use_internal_backbone_pretrain = pretrained and (model_path == "")

#     if pretrained and (use_internal_backbone_pretrain or backbone_ckpt):
#         if distributed:
#             if local_rank == 0:
#                 download_weights(phi)
#             dist.barrier()
#         else:
#             download_weights(phi)

#     # 构建模型
#     # 只有 model_path 为空且需要 backbone 预训练时，才走模型内部默认加载
#     model = SegFormer(
#         num_classes=num_classes,
#         phi=phi,
#         pretrained=use_internal_backbone_pretrain,
#         in_chans=input_channels
#     )

#     # 如果完全不使用预训练，才做统一初始化
#     if not pretrained and model_path == '':
#         weights_init(model)

#     # -------------------------------------------------------------------#
#     #   加载权重（避免“内部预训练 + 外部 model_path”双重加载）
#     # -------------------------------------------------------------------#
#     if model_path != '':
#         if backbone_ckpt:
#             if hasattr(model.backbone, "load_pretrained"):
#                 if local_rank == 0:
#                     print(f"Load backbone weights with remapping from {model_path}.")
#                 model.backbone.load_pretrained(model_path)
#             else:
#                 raise AttributeError("当前 backbone 不包含 load_pretrained()，无法正确映射 backbone 权重。")
#         else:
#             load_full_model_weights(model, model_path, device, local_rank)

#     # -------------------------------------------------------------------#
#     #   计算静态指标 (FLOPs, Params)
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0
#     if local_rank == 0:
#         print("\n正在计算模型复杂度 (FLOPs & Params)...")
#         flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
#         print(f"计算完成: FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

#     # 记录 Loss
#     if local_rank == 0:
#         time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
#         log_dir = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history = None

#     if fp16:
#         from torch.cuda.amp import GradScaler as GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train = model.train()

#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(
#                 model_train,
#                 device_ids=[local_rank],
#                 find_unused_parameters=True
#             )
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()

#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
#         val_lines = f.readlines()

#     num_train = len(train_lines)
#     num_val = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes=num_classes, phi=phi, model_path=model_path, input_shape=input_shape,
#             Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
#             Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
#             Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr,
#             optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
#             save_period=save_period, save_dir=save_dir, num_workers=num_workers,
#             num_train=num_train, num_val=num_val
#         )

#     # -------------------------------------------------------------------#
#     #   初始化详细指标 CSV
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""
#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")

#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 header = [
#                     'Epoch',
#                     'Train Time (min)',
#                     'Params Size (MB)',
#                     'Total Params (10^6)',
#                     'FLOPs (G)',
#                     'mIoU (%)',
#                     'mPA (%)',
#                     'Accuracy (%)',
#                     'F1 Score',
#                     'clDice (%)',
#                     'CC-F1 (%)'
#                 ]
#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     if True:
#         UnFreeze_flag = False

#         if Freeze_Train:
#             for param in model.backbone.parameters():
#                 param.requires_grad = False

#         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#         nbs = 16
#         lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#         lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#         Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#         Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#         optimizer = {
#             'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
#             'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
#             'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
#         }[optimizer_type]

#         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

#         epoch_step = num_train // batch_size
#         epoch_step_val = math.ceil(num_val / batch_size)

#         if epoch_step == 0 or epoch_step_val == 0:
#             raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#         train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
#         val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

#         if distributed:
#             train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
#             val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
#             batch_size = max(1, batch_size // ngpus_per_node)
#             train_shuffle = False
#             val_shuffle = False
#         else:
#             train_sampler = None
#             val_sampler = None
#             train_shuffle = True
#             val_shuffle = False

#         gen = build_dataloader(
#             train_dataset, batch_size, train_shuffle, num_workers, rank, seed,
#             sampler=train_sampler, drop_last=True
#         )
#         gen_val = build_dataloader(
#             val_dataset, batch_size, val_shuffle, num_workers, rank, seed,
#             sampler=val_sampler, drop_last=False
#         )

#         if local_rank == 0:
#             eval_callback = EvalCallback(
#                 model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
#                 eval_flag=eval_flag, period=eval_period
#             )
#         else:
#             eval_callback = None

#         # ---------------------------------------#
#         #   Epoch 循环
#         # ---------------------------------------#
#         for epoch in range(Init_Epoch, UnFreeze_Epoch):
#             epoch_start_time = time.time()

#             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#                 batch_size = Unfreeze_batch_size

#                 nbs = 16
#                 lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
#                 lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
#                 Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#                 Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
#                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

#                 for param in model.backbone.parameters():
#                     param.requires_grad = True

#                 if distributed:
#                     batch_size = max(1, batch_size // ngpus_per_node)

#                 epoch_step = num_train // batch_size
#                 epoch_step_val = math.ceil(num_val / batch_size)

#                 if epoch_step == 0 or epoch_step_val == 0:
#                     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#                 gen = build_dataloader(
#                     train_dataset, batch_size, train_shuffle, num_workers, rank, seed,
#                     sampler=train_sampler, drop_last=True
#                 )
#                 gen_val = build_dataloader(
#                     val_dataset, batch_size, val_shuffle, num_workers, rank, seed,
#                     sampler=val_sampler, drop_last=False
#                 )

#                 UnFreeze_flag = True

#             if distributed:
#                 train_sampler.set_epoch(epoch)

#             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

#             fit_one_epoch(
#                 model_train, model, loss_history, eval_callback, optimizer, epoch,
#                 epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
#                 dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
#                 save_period, save_dir, local_rank
#             )

#             if distributed:
#                 dist.barrier()

#             # -------------------------------------------------------------------#
#             #   每 eval_period 轮记录一次详细表格并打印
#             # -------------------------------------------------------------------#
#             if local_rank == 0 and (epoch + 1) % eval_period == 0:
#                 epoch_end_time = time.time()
#                 train_time_min = (epoch_end_time - epoch_start_time) / 60.0

#                 current_miou = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
#                 current_mpa = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
#                 current_acc = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
#                 current_f1 = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
#                 current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
#                 current_ccf1 = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

#                 with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     row_data = [
#                         epoch + 1,
#                         f"{train_time_min:.2f}",
#                         f"{params_mb:.2f}",
#                         f"{params_m:.3f}",
#                         f"{flops_g:.3f}",
#                         f"{current_miou * 100:.2f}",
#                         f"{current_mpa * 100:.2f}",
#                         f"{current_acc * 100:.2f}",
#                         f"{current_f1:.4f}",
#                         f"{current_cldice * 100:.2f}",
#                         f"{current_ccf1 * 100:.2f}"
#                     ]
#                     writer.writerow(row_data)

#                 table_data = {
#                     "Metric": [
#                         "Epoch", "Train Time (min)", "Params Size (MB)",
#                         "Total Params (10^6)", "FLOPs (G)", "mIoU (%)",
#                         "mPA (%)", "Accuracy (%)", "F1 Score",
#                         "clDice (%)", "CC-F1 (%)"
#                     ],
#                     "Value": row_data
#                 }
#                 df = pd.DataFrame(table_data)
#                 print("\n" + "=" * 50)
#                 print(f"Performance Summary at Epoch {epoch + 1}")
#                 print("=" * 50)
#                 print(df.to_string(index=False))
#                 print("=" * 50 + "\n")

#         if local_rank == 0:
#             loss_history.writer.close()




# Mask2Former
# import datetime
# import os
# import time
# import csv
# import pandas as pd
# import copy
# from functools import partial

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 引入 thop 计算 FLOPs
# try:
#     from thop import profile
# except ImportError:
#     print("【警告】未安装 thop 库，无法自动计算 FLOPs。请运行 pip install thop")
#     profile = None

# # -------------------------------------------------------------------#
# #   导入 Mask2Former
# # -------------------------------------------------------------------#
# from nets.mask2former import Mask2Former

# from nets.segformer_training import (
#     get_lr_scheduler,
#     set_optimizer_lr,
#     weights_init
# )

# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import SegmentationDataset, seg_dataset_collate
# from utils.utils import (
#     download_weights,
#     seed_everything,
#     show_config,
#     worker_init_fn
# )
# from utils.utils_fit import fit_one_epoch


# # -------------------------------------------------------------------#
# #   鲁棒的 FLOPs 计算函数 CPU模式
# # -------------------------------------------------------------------#
# def get_model_complexity(model, input_channels, input_shape):
#     if profile is None:
#         return 0, 0, 0

#     try:
#         model_copy = copy.deepcopy(model).to("cpu").eval()
#     except Exception as e:
#         print(f"【注意】模型深拷贝失败 ({e})，尝试直接使用原模型计算...")
#         model_copy = model.eval().cpu()

#     dummy_input = torch.randn(
#         1,
#         input_channels,
#         input_shape[0],
#         input_shape[1]
#     ).to("cpu")

#     try:
#         flops, params = profile(
#             model_copy,
#             inputs=(dummy_input,),
#             verbose=False
#         )
#     except Exception as e:
#         print(f"【错误】FLOPs 计算失败: {e}")
#         return 0, 0, 0
#     finally:
#         if model_copy is not model:
#             del model_copy
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     flops_g = flops / 1e9
#     params_m = params / 1e6
#     params_mb = params * 4 / (1024 * 1024)

#     if next(model.parameters()).is_cuda:
#         model.train().cuda()
#     else:
#         model.train()

#     return flops_g, params_m, params_mb


# if __name__ == "__main__":

#     # ---------------------------------#
#     #   Cuda 是否使用Cuda
#     # ---------------------------------#
#     Cuda            = True
#     seed            = 11
#     distributed     = False
#     sync_bn         = False
#     fp16            = False

#     # -----------------------------------------------------#
#     #   num_classes 分类个数
#     # -----------------------------------------------------#
#     num_classes     = 2

#     # -------------------------------------------------------------------#
#     #   输入通道数 8通道
#     # -------------------------------------------------------------------#
#     input_channels  = 8

#     # -------------------------------------------------------------------#
#     #   模型配置
#     # -------------------------------------------------------------------#
#     phi             = "b5"
#     pretrained      = True

#     # 注意：
#     # 这里不要填 DeepLabV3+ 或 SegFormer 的权重。
#     # 如果以后恢复 Mask2Former 自己训练出来的权重，再填路径。
#     model_path      = ""

#     input_shape     = [512, 512]

#     Init_Epoch      = 0
#     Freeze_Epoch    = 50
#     Freeze_batch_size = 8
#     UnFreeze_Epoch  = 200
#     Unfreeze_batch_size = 4

#     Freeze_Train    = False

#     # ------------------------------------------------------------------#
#     #   训练参数
#     # ------------------------------------------------------------------#
#     Init_lr         = 1e-4
#     Min_lr          = Init_lr * 0.01
#     optimizer_type  = "adamw"
#     momentum        = 0.9
#     weight_decay    = 1e-2
#     lr_decay_type   = "cos"
#     save_period     = 5

#     # Mask2Former 单独保存
#     save_dir        = "logs_mask2former"

#     # ------------------------------------------------------------------#
#     #   评估设置
#     # ------------------------------------------------------------------#
#     eval_flag       = True
#     eval_period     = 5

#     VOCdevkit_path  = "VOCdevkit"
#     dice_loss       = True
#     focal_loss      = False
#     cls_weights     = np.ones([num_classes], np.float32)
#     num_workers     = 4

#     seed_everything(seed)
#     ngpus_per_node  = torch.cuda.device_count()

#     if distributed:
#         dist.init_process_group(backend="nccl")
#         local_rank  = int(os.environ["LOCAL_RANK"])
#         rank        = int(os.environ["RANK"])
#         device      = torch.device("cuda", local_rank)
#     else:
#         device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         local_rank  = 0
#         rank        = 0

#     # -------------------------------------------------------------------#
#     #   下载 / 准备 MiT backbone 预训练权重
#     # -------------------------------------------------------------------#
#     if pretrained:
#         if distributed:
#             if local_rank == 0:
#                 download_weights(phi)
#             dist.barrier()
#         else:
#             download_weights(phi)

#     # -------------------------------------------------------------------#
#     #   构建 Mask2Former 模型
#     # -------------------------------------------------------------------#
#     model = Mask2Former(
#         num_classes=num_classes,
#         in_chans=input_channels,
#         phi=phi,
#         pretrained=pretrained,

#         # 默认 b5 版本，显存要求较高
#         hidden_dim=256,
#         mask_dim=256,
#         num_queries=100,
#         nheads=8,
#         dec_layers=6,
#         dim_feedforward=1024
#     )

#     # -------------------------------------------------------------------#
#     #   计算静态指标 FLOPs, Params
#     # -------------------------------------------------------------------#
#     flops_g, params_m, params_mb = 0, 0, 0

#     if local_rank == 0:
#         print("\n正在计算 Mask2Former 的复杂度...")
#         flops_input_shape = [512, 512]
#         flops_g, params_m, params_mb = get_model_complexity(
#             model,
#             input_channels,
#             flops_input_shape
#         )
#         print(f"计算基准尺寸: {flops_input_shape}")
#         print(
#             f"计算完成: FLOPs={flops_g:.3f}G, "
#             f"Params={params_m:.3f}M ({params_mb:.2f}MB)\n"
#         )

#     if not pretrained:
#         weights_init(model)

#     # -------------------------------------------------------------------#
#     #   加载自定义权重 Resume
#     # -------------------------------------------------------------------#
#     if model_path != "":
#         if local_rank == 0:
#             print("Load weights {}.".format(model_path))

#         model_dict      = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location=device)
#         load_key, no_load_key, temp_dict = [], [], {}

#         if "state_dict" in pretrained_dict:
#             pretrained_dict = pretrained_dict["state_dict"]

#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)

#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)

#         if local_rank == 0:
#             print("\nSuccessful Load Key Num:", len(load_key))
#             print("Fail To Load Key Num:", len(no_load_key))

#     # -------------------------------------------------------------------#
#     #   记录 Loss
#     # -------------------------------------------------------------------#
#     if local_rank == 0:
#         time_str        = datetime.datetime.strftime(
#             datetime.datetime.now(),
#             "%Y_%m_%d_%H_%M_%S"
#         )
#         log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
#         loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
#     else:
#         loss_history    = None

#     if fp16:
#         from torch.cuda.amp import GradScaler
#         scaler = GradScaler()
#     else:
#         scaler = None

#     model_train = model.train()

#     if sync_bn and ngpus_per_node > 1 and distributed:
#         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
#     elif sync_bn:
#         print("Sync_bn is not support in one gpu or not distributed.")

#     if Cuda:
#         if distributed:
#             model_train = model_train.cuda(local_rank)
#             model_train = torch.nn.parallel.DistributedDataParallel(
#                 model_train,
#                 device_ids=[local_rank],
#                 find_unused_parameters=True
#             )
#         else:
#             model_train = torch.nn.DataParallel(model)
#             cudnn.benchmark = True
#             model_train = model_train.cuda()

#     # -------------------------------------------------------------------#
#     #   读取数据集
#     # -------------------------------------------------------------------#
#     with open(
#         os.path.join(
#             VOCdevkit_path,
#             "VOC2007/ImageSets/Segmentation/train.txt"
#         ),
#         "r"
#     ) as f:
#         train_lines = f.readlines()

#     with open(
#         os.path.join(
#             VOCdevkit_path,
#             "VOC2007/ImageSets/Segmentation/val.txt"
#         ),
#         "r"
#     ) as f:
#         val_lines = f.readlines()

#     num_train = len(train_lines)
#     num_val   = len(val_lines)

#     if local_rank == 0:
#         show_config(
#             num_classes=num_classes,
#             phi=phi,
#             model_path=model_path,
#             input_shape=input_shape,
#             Init_Epoch=Init_Epoch,
#             Freeze_Epoch=Freeze_Epoch,
#             UnFreeze_Epoch=UnFreeze_Epoch,
#             Freeze_batch_size=Freeze_batch_size,
#             Unfreeze_batch_size=Unfreeze_batch_size,
#             Freeze_Train=Freeze_Train,
#             Init_lr=Init_lr,
#             Min_lr=Min_lr,
#             optimizer_type=optimizer_type,
#             momentum=momentum,
#             lr_decay_type=lr_decay_type,
#             save_period=save_period,
#             save_dir=save_dir,
#             num_workers=num_workers,
#             num_train=num_train,
#             num_val=num_val
#         )

#     # -------------------------------------------------------------------#
#     #   初始化详细指标 CSV
#     # -------------------------------------------------------------------#
#     metrics_csv_path = ""

#     if local_rank == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         metrics_csv_path = os.path.join(
#             save_dir,
#             "training_metrics_every_5_epochs.csv"
#         )

#         if not os.path.exists(metrics_csv_path):
#             with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)

#                 header = [
#                     "Epoch",
#                     "Train Time (min)",
#                     "Params Size (MB)",
#                     "Total Params (10^6)",
#                     "FLOPs (G)",
#                     "mIoU (%)",
#                     "mPA (%)",
#                     "Accuracy (%)",
#                     "F1 Score",
#                     "clDice (%)",
#                     "CC-F1 (%)"
#                 ]

#                 writer.writerow(header)

#     # ------------------------------------------------------#
#     #   开始训练
#     # ------------------------------------------------------#
#     UnFreeze_flag = False

#     if Freeze_Train:
#         try:
#             for param in model.backbone.parameters():
#                 param.requires_grad = False
#         except Exception:
#             print("【注意】无法自动冻结 Backbone，将进行全参数训练。")

#     batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

#     nbs             = 16
#     lr_limit_max    = 1e-4 if optimizer_type in ["adam", "adamw"] else 5e-2
#     lr_limit_min    = 3e-5 if optimizer_type in ["adam", "adamw"] else 5e-4
#     Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#     Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#     optimizer = {
#         "adam": optim.Adam(
#             model.parameters(),
#             Init_lr_fit,
#             betas=(momentum, 0.999),
#             weight_decay=weight_decay
#         ),
#         "adamw": optim.AdamW(
#             model.parameters(),
#             Init_lr_fit,
#             betas=(momentum, 0.999),
#             weight_decay=weight_decay
#         ),
#         "sgd": optim.SGD(
#             model.parameters(),
#             Init_lr_fit,
#             momentum=momentum,
#             nesterov=True,
#             weight_decay=weight_decay
#         )
#     }[optimizer_type]

#     lr_scheduler_func = get_lr_scheduler(
#         lr_decay_type,
#         Init_lr_fit,
#         Min_lr_fit,
#         UnFreeze_Epoch
#     )

#     epoch_step      = num_train // batch_size
#     epoch_step_val  = max(1, (num_val + batch_size - 1) // batch_size)

#     if epoch_step == 0 or epoch_step_val == 0:
#         raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#     train_dataset = SegmentationDataset(
#         train_lines,
#         input_shape,
#         num_classes,
#         True,
#         VOCdevkit_path
#     )

#     val_dataset = SegmentationDataset(
#         val_lines,
#         input_shape,
#         num_classes,
#         False,
#         VOCdevkit_path
#     )

#     if distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(
#             train_dataset,
#             shuffle=True
#         )
#         val_sampler = torch.utils.data.distributed.DistributedSampler(
#             val_dataset,
#             shuffle=False
#         )
#         batch_size = batch_size // ngpus_per_node
#         train_shuffle = False
#         val_shuffle = False
#     else:
#         train_sampler = None
#         val_sampler = None
#         train_shuffle = True
#         val_shuffle = False

#     gen = DataLoader(
#         train_dataset,
#         shuffle=train_shuffle,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#         collate_fn=seg_dataset_collate,
#         sampler=train_sampler,
#         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#     )

#     gen_val = DataLoader(
#         val_dataset,
#         shuffle=val_shuffle,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False,
#         collate_fn=seg_dataset_collate,
#         sampler=val_sampler,
#         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#     )

#     if local_rank == 0:
#         eval_callback = EvalCallback(
#             model,
#             input_shape,
#             num_classes,
#             val_lines,
#             VOCdevkit_path,
#             log_dir,
#             Cuda,
#             eval_flag=eval_flag,
#             period=eval_period
#         )
#     else:
#         eval_callback = None

#     # ---------------------------------------#
#     #   Epoch 循环
#     # ---------------------------------------#
#     for epoch in range(Init_Epoch, UnFreeze_Epoch):
#         epoch_start_time = time.time()

#         if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
#             batch_size = Unfreeze_batch_size

#             nbs             = 16
#             lr_limit_max    = 1e-4 if optimizer_type in ["adam", "adamw"] else 5e-2
#             lr_limit_min    = 3e-5 if optimizer_type in ["adam", "adamw"] else 5e-4
#             Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
#             Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

#             lr_scheduler_func = get_lr_scheduler(
#                 lr_decay_type,
#                 Init_lr_fit,
#                 Min_lr_fit,
#                 UnFreeze_Epoch
#             )

#             try:
#                 for param in model.backbone.parameters():
#                     param.requires_grad = True
#             except Exception:
#                 for param in model.parameters():
#                     param.requires_grad = True

#             epoch_step      = num_train // batch_size
#             epoch_step_val  = max(1, (num_val + batch_size - 1) // batch_size)

#             if epoch_step == 0 or epoch_step_val == 0:
#                 raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#             gen = DataLoader(
#                 train_dataset,
#                 shuffle=train_shuffle,
#                 batch_size=batch_size,
#                 num_workers=num_workers,
#                 pin_memory=True,
#                 drop_last=True,
#                 collate_fn=seg_dataset_collate,
#                 sampler=train_sampler,
#                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#             )

#             gen_val = DataLoader(
#                 val_dataset,
#                 shuffle=val_shuffle,
#                 batch_size=batch_size,
#                 num_workers=num_workers,
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=seg_dataset_collate,
#                 sampler=val_sampler,
#                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
#             )

#             UnFreeze_flag = True

#         if distributed:
#             train_sampler.set_epoch(epoch)

#         set_optimizer_lr(
#             optimizer,
#             lr_scheduler_func,
#             epoch
#         )

#         fit_one_epoch(
#             model_train,
#             model,
#             loss_history,
#             eval_callback,
#             optimizer,
#             epoch,
#             epoch_step,
#             epoch_step_val,
#             gen,
#             gen_val,
#             UnFreeze_Epoch,
#             Cuda,
#             dice_loss,
#             focal_loss,
#             cls_weights,
#             num_classes,
#             fp16,
#             scaler,
#             save_period,
#             save_dir,
#             local_rank
#         )

#         if distributed:
#             dist.barrier()

#         # -------------------------------------------------------------------#
#         #   每5轮记录一次详细表格并打印
#         # -------------------------------------------------------------------#
#         if local_rank == 0 and (epoch + 1) % eval_period == 0:
#             epoch_end_time = time.time()
#             train_time_min = (epoch_end_time - epoch_start_time) / 60.0

#             current_miou = (
#                 eval_callback.mious[-1]
#                 if hasattr(eval_callback, "mious") and len(eval_callback.mious) > 0
#                 else 0
#             )

#             current_mpa = (
#                 eval_callback.mPAs[-1]
#                 if hasattr(eval_callback, "mPAs") and len(eval_callback.mPAs) > 0
#                 else 0
#             )

#             current_acc = (
#                 eval_callback.Accs[-1]
#                 if hasattr(eval_callback, "Accs") and len(eval_callback.Accs) > 0
#                 else 0
#             )

#             current_f1 = (
#                 eval_callback.f1s[-1]
#                 if hasattr(eval_callback, "f1s") and len(eval_callback.f1s) > 0
#                 else 0
#             )

#             current_cldice = (
#                 eval_callback.cldices[-1]
#                 if hasattr(eval_callback, "cldices") and len(eval_callback.cldices) > 0
#                 else 0
#             )

#             current_ccf1 = (
#                 eval_callback.cc_f1s[-1]
#                 if hasattr(eval_callback, "cc_f1s") and len(eval_callback.cc_f1s) > 0
#                 else 0
#             )

#             row_data = [
#                 epoch + 1,
#                 f"{train_time_min:.2f}",
#                 f"{params_mb:.2f}",
#                 f"{params_m:.3f}",
#                 f"{flops_g:.3f}",
#                 f"{current_miou * 100:.2f}",
#                 f"{current_mpa * 100:.2f}",
#                 f"{current_acc * 100:.2f}",
#                 f"{current_f1:.4f}",
#                 f"{current_cldice * 100:.2f}",
#                 f"{current_ccf1 * 100:.2f}"
#             ]

#             with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(row_data)

#             table_data = {
#                 "Metric": [
#                     "Epoch",
#                     "Train Time (min)",
#                     "Params Size (MB)",
#                     "Total Params (10^6)",
#                     "FLOPs (G)",
#                     "mIoU (%)",
#                     "mPA (%)",
#                     "Accuracy (%)",
#                     "F1 Score",
#                     "clDice (%)",
#                     "CC-F1 (%)"
#                 ],
#                 "Value": row_data
#             }

#             df = pd.DataFrame(table_data)

#             print("\n" + "=" * 50)
#             print(f"Mask2Former Performance Summary at Epoch {epoch + 1}")
#             print("=" * 50)
#             print(df.to_string(index=False))
#             print("=" * 50 + "\n")

#     if local_rank == 0 and loss_history is not None:
#         loss_history.writer.close()