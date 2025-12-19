# import _init_paths
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = [8, 8]

# from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
# from lib.test.evaluation import get_dataset, trackerlist

# trackers = []
# dataset_name = 'trackingnet'  # 改为你测试的数据集
# # choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot', 'otb99_lang', 'tnl2k'

# trackers.extend(trackerlist(name='sutrack', parameter_name='sutrack_t224', dataset_name=dataset_name,
#                             run_ids=None, display_name='SUTrack-T224'))

# dataset = get_dataset(dataset_name)

# print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
#               force_evaluation=True)

# # === FPS Information ===
# import os
# print("\n" + "="*60)
# print("FPS Information:")
# print("="*60)

# for tracker in trackers:
#     print(f"\nTracker: {tracker.display_name}")
    
#     fps_list = []
#     for seq in dataset:
#         try:
#             times_file = os.path.join(tracker.results_dir, dataset_name, seq.name + '_time.txt')
            
#             if os.path.exists(times_file):
#                 with open(times_file, 'r') as f:
#                     times = [float(line.strip()) for line in f if line.strip()]
#                     if times:
#                         avg_time = sum(times) / len(times)
#                         fps = 1.0 / avg_time if avg_time > 0 else 0
#                         fps_list.append(fps)
#         except Exception:
#             continue
    
#     if fps_list:
#         avg_fps = sum(fps_list) / len(fps_list)
#         print(f"  Average FPS: {avg_fps:.2f}")
#         print(f"  Min FPS: {min(fps_list):.2f}")
#         print(f"  Max FPS: {max(fps_list):.2f}")
#         print(f"  Total sequences: {len(fps_list)}")
#     else:
#         print("  ⚠️  FPS data not available")
#         print("  Tip: Time files (*_time.txt) are generated during testing")

# print("="*60)

import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'uav'
# choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot', 'otb99_lang', 'tnl2k', 'got10k_test'

trackers.extend(trackerlist(name='sutrack', parameter_name='sutrack_t224', dataset_name=dataset_name,
                            run_ids=None, display_name='sutrack_t224'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)

# === FPS Information ===
import os
print("\n" + "="*60)
print("FPS Information:")
print("="*60)

for tracker in trackers:
    print(f"\nTracker: {tracker.display_name}")
    
    fps_list = []
    for seq in dataset:
        try:
            times_file = os.path.join(tracker.results_dir, dataset_name, seq.name + '_time.txt')
            
            if os.path.exists(times_file):
                with open(times_file, 'r') as f:
                    times = [float(line.strip()) for line in f if line.strip()]
                    if times:
                        avg_time = sum(times) / len(times)
                        fps = 1.0 / avg_time if avg_time > 0 else 0
                        fps_list.append(fps)
        except Exception:
            continue
    
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Min FPS: {min(fps_list):.2f}")
        print(f"  Max FPS: {max(fps_list):.2f}")
        print(f"  Total sequences: {len(fps_list)}")
    else:
        print("  ⚠️  FPS data not available")
        print("  Tip: Time files (*_time.txt) are generated during testing")

print("="*60)