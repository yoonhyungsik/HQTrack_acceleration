import os
import json
import time
import glob
from pathlib import Path
from depth_optical_flow_final import MotionDepthAnalyzer, HQTrack, count_total_frames
from configs.ytb_vip_dav_deaot_internT import EngineConfig

def run_tracking_experiment(use_ci, use_iqr, use_ewma):
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        demo_video = 'p_09'
        img_files = sorted(glob.glob(os.path.join(base_path, demo_video, '*.jp*')))
        total_frames = count_total_frames(img_files)
        save_dir = os.path.join(base_path, "demo/output")
        os.makedirs(save_dir, exist_ok=True)

        config = {
            'exp_name': 'default',
            'model': 'internT_msdeaotl_v2',
            'pretrain_model_path': 'result/default_InternT_MSDeAOTL_V2/YTB_DAV_VIP/ckpt/save_step_42000.pth',
            'gpu_id': 0,
        }

        cfg = EngineConfig(config['exp_name'], config['model'])
        cfg.TEST_CKPT_PATH = os.path.join(base_path, '..', config['pretrain_model_path'])

        # Create tracker and analyzer
        tracker = HQTrack(cfg, config, local_track=True, sam_refine=True, sam_refine_iou=0.1)
        analyzer = MotionDepthAnalyzer(use_ci=use_ci, use_iqr=use_iqr, use_ewma=use_ewma)

        # Run full tracking process using built-in execution logic
        results = tracker.run_all(img_files, analyzer)

        return results

    except Exception as e:
        print(f"[ERROR] run_tracking_experiment 실패: {e}")
        return None

if __name__ == "__main__":
    use_ci = True
    use_iqr = False
    use_ewma = True

    print(f"\n[수동 실험 실행] CI={use_ci}, IQR={use_iqr}, EWMA={use_ewma}")
    start_time = time.time()
    results = run_tracking_experiment(use_ci, use_iqr, use_ewma)
    elapsed_time = time.time() - start_time

    if results is None:
        print("[실패] 결과 없음 (에러 발생 가능성)")
        exit(1)

    print("\n[실험 결과]")
    for key, value in results.items():
        print(f"{key}: {value}")

    save_dir = "ablation_results"
    os.makedirs(save_dir, exist_ok=True)
    experiment_name = f"CI_{use_ci}_IQR_{use_iqr}_EWMA_{use_ewma}"
    results["experiment"] = experiment_name
    results["elapsed_time"] = round(elapsed_time, 2)

    with open(os.path.join(save_dir, f"metrics_{experiment_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[저장 완료] {experiment_name} → metrics_{experiment_name}.json")
