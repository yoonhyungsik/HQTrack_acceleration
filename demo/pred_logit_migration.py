import os
import glob
import torch
import cv2
import numpy as np
import random
import sys
import torch.nn.functional as F
from PIL import Image
import importlib
from torchvision import transforms
from os.path import join
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
cnt = 0 #for pred_logit in track function
torch.cuda.empty_cache

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.image import flip_tensor
AOT_PATH = os.path.join(os.path.dirname(__file__), '..')
import dataloaders.video_transforms as tr
from networks.engines import build_engine
from utils.checkpoint import load_network
from networks.models import build_vos_model
from utils.metric import pytorch_iou
from pathlib import Path
base_path =os.path.dirname(os.path.abspath(__file__))
#base_path = 'C:/Users/ye761/HQTrack/demo/your_video/bolt'
# video for test
demo_video = 'p_09'
img_files = sorted(glob.glob(join(base_path, demo_video, '*.jp*'))) 
print(f"diretory: {base_path}")
print(f"Image files: {img_files}")
point_box_prompts=[]


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)

cur_colors = [(0, 255, 255), # yellow b g r
              (255, 0, 0), # blue
              (0, 255, 0), # green
              (0, 0, 255), # red
              (255, 255, 255), # white
              (0, 0, 0), # black
              (255, 255, 0), # Cyan
              (225, 228, 255), # MistyRose
              (180, 105, 255), # HotPink
              (255, 0, 255), # Magenta
              ]*100


class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.saved_pred_logits={} #pred_logit 값 저장할 딕셔너리
        self.tracked_frames=[]
        self.with_crop = False
        self.EXPAND_SCALE = None
        self.small_ratio = 12
        self.mid_ratio = 100
        self.large_ratio = 0.5
        self.AOT_INPUT_SIZE = (465, 465)
        self.cnt = 2
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        self.model.cuda(gpu_id)
        self.model.eval()
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.aug_nums = len(cfg.TEST_MULTISCALE)
        if cfg.TEST_FLIP:
            self.aug_nums *= 2
        self.engine = []
        for aug_idx in range(self.aug_nums):
            self.engine.append(build_engine(cfg.MODEL_ENGINE,
                                            phase='eval',
                                            aot_model=self.model,
                                            gpu_id=gpu_id,
                                            short_term_mem_skip=cfg.TEST_SHORT_TERM_MEM_SKIP,
                                            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
                                            ))
            self.engine[-1].eval()
        self.transform = transforms.Compose([
            tr.MultiRestrictSize_(cfg.TEST_MAX_SHORT_EDGE,
                                  cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
                                  cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

    def add_first_frame(self, frame, mask):
        sample = {
            'current_img': frame,
            'current_label': mask,
            'height': frame.shape[0],
            'weight': frame.shape[1]
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
            torch.cuda.empty_cache()
        for aug_idx in range(self.aug_nums):
            frame = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = sample[aug_idx]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")
            self.engine[aug_idx].add_reference_frame(frame, mask, frame_step=0, obj_nums=int(mask.max()))
    
    def track(self, image):
        print(f"Track called, i: {cnt}")
        height = image.shape[0]
        width = image.shape[1]
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
            'flip': False
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
            torch.cuda.empty_cache()
        all_preds = []
        for aug_idx in range(self.aug_nums):
            output_height = sample[aug_idx]['meta']['height']
            output_width = sample[aug_idx]['meta']['width']
            image = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            #image = image.cuda(self.gpu_id, non_blocking=True)
            try:
                self.engine[aug_idx].match_propogate_one_frame(image) #network->engine->aot_engine.py->aotengine
            except Exception as e:
                print(f"match_propogate_one_frame fail: {e}")
                raise
            is_flipped = sample[aug_idx]['meta']['flip']
            pred_logit = self.engine[aug_idx].decode_current_logits((output_height, output_width))
            if is_flipped:
                pred_logit = flip_tensor(pred_logit, 3)
            all_preds.append(pred_logit)
        
        if len(all_preds) > 1:
            combined_pred_logit = torch.mean(torch.stack(all_preds),dim=0)
        else:
            combined_pred_logit = all_preds[0] 
           
        def load_pred_logit(directory_path):
                pred_logits = []  # 로드된 모든 텐서를 저장할 리스트
                try:
                # 디렉토리 내 파일 리스트 확인
                    file_list = os.listdir(directory_path)
                    pt_files = [f for f in file_list if f.endswith('.pt')]

                    for pt_file in pt_files:
                        file_path = os.path.join(directory_path, pt_file)
                        #print(f"로딩 중: {file_path}")
                        pred_logits.append(torch.load(file_path))
        
                    print(f"총 {len(pred_logits)}개의 파일 로드 완료 {file_path} & {pt_files}")
                    return pred_logits
                except Exception as e:
                    print(f"디렉토리 로드 실패: {e}")
                    raise
    
        def compute_weighted_logit(logits, reference_camera=9, camera_ids=[2, 8, 26, 30, 31, 36, 9],device='cuda'):
            global cnt
            """
            주어진 카메라 로짓들을 기준 카메라와의 거리 기반으로 가중합 계산.

            Args:
            logits (dict): {카메라 ID: 로짓 값} 형식의 딕셔너리
            reference_camera (int): 기준 카메라 ID (기본값: 22)
            camera_ids (list): 포함할 카메라 ID 리스트

            Returns:
            torch.Tensor: 가중합된 로짓 값
            """
            
            # 카메라 간 거리 계산
            distances = []

            for camera in camera_ids:
                if camera == reference_camera:
                    distances.append(0)  # reference_camera는 거리를 0으로 설정
                else:
                    distances.append(abs(camera - reference_camera))
            # 거리 기반 가중치 계산
            weights = [1 / (distance + 1) for distance in distances]
            # 가중치 정규화
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
        
            # 각 카메라 ID에 맞는 로짓을 가져옴 (리스트로 로딩된 값을 사용)
            logit_02 = torch.tensor(logits[2][cnt], dtype=torch.float32).to(device)
            logit_08 = torch.tensor(logits[8][cnt], dtype=torch.float32).to(device)
            logit_31 = torch.tensor(logits[31][cnt], dtype=torch.float32).to(device)    
            logit_26 = torch.tensor(logits[26][cnt], dtype=torch.float32).to(device)
            logit_30 = torch.tensor(logits[30][cnt], dtype=torch.float32).to(device)   
            logit_36 = torch.tensor(logits[36][cnt], dtype=torch.float32).to(device)  
            cnt+=1
            # 가중합 계산
            print("succes loading logit files, loading compute_weighted_logit")
            combined_logit = sum(w * logit for w, logit in zip(normalized_weights, [logit_02, logit_08, logit_31, logit_26, logit_30, logit_36, pred_logit]))
            print("success compute weighted logit")
            
            return combined_logit
        
    
        #로짓 파일 경로
        logit_path_02 = r"C:\Users\ye761\HQTrack\demo\p_02_pred_logit.path"
        logit_path_08 = r"C:\Users\ye761\HQTrack\demo\p_08_pred_logit.path"
        logit_path_26 = r"C:\Users\ye761\HQTrack\demo\p_26_pred_logit.path"
        logit_path_31 = r"C:\Users\ye761\HQTrack\demo\p_31_pred_logit.path"
        logit_path_30 = r"C:\Users\ye761\HQTrack\demo\p_30_pred_logit.path"
        logit_path_36 = r"C:\Users\ye761\HQTrack\demo\p_36_pred_logit.path"
        
        # 로짓 읽기
        logit_02 = load_pred_logit(logit_path_02)
        logit_08 = load_pred_logit(logit_path_08)
        logit_26 = load_pred_logit(logit_path_26)
        logit_31 = load_pred_logit(logit_path_31)
        logit_30 = load_pred_logit(logit_path_30)
        logit_36 = load_pred_logit(logit_path_36)

        logit_09 = combined_pred_logit  # 의 로짓을 로드
        # logits 딕셔너리 정의
        logits = {
                2: logit_02,
                8: logit_08,
                26: logit_26,
                9: logit_09,
                31: logit_31,
                36: logit_36,
                30: logit_30
                
            }
        # 예시로 가중합 로짓 계산
        combined_logit = compute_weighted_logit(logits, reference_camera=9, camera_ids=[2, 8, 31, 26, 30, 36, 9])                        
        # 병합된 로짓으로 마스크 생성
        pred_prob = torch.softmax(combined_logit, dim=1)
        conf = torch.max(pred_prob).item()
        #all_preds.append(pred_prob)
        #cat_all_preds = torch.cat(all_preds, dim=0)
        #pred_prob = torch.mean(pred_prob, dim=0, keepdim=True)
        pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()
        _pred_label = F.interpolate(pred_label,
                                        size=self.engine[aug_idx].input_size_2d,
                                        mode="nearest")
        self.engine[aug_idx].update_memory(_pred_label)
        # 결과 마스크
        mask = pred_label[0, 0].detach().cpu().numpy().astype(np.uint8)
#가우시안이나 거리 관련 알고리즘 사용도괜찮을 것 같고 confidence 값 사용도 고려해볼만 한듯   ##sfm 공부해서 적용가능한지 알아볼 것     
        '''
            pred_prob = torch.softmax(pred_logit, dim=1)
            all_preds.append(pred_prob)
            cat_all_preds = torch.cat(all_preds, dim=0)
            pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)
            pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()
            _pred_label = F.interpolate(pred_label,
                                        size=self.engine[aug_idx].input_size_2d,
                                        mode="nearest")
            self.engine[aug_idx].update_memory(_pred_label)
            mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)
        '''
        conf = 0
        return mask, conf


def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class HQTrack(object):
    def __init__(self, cfg, config, local_track=False,sam_refine=False,sam_refine_iou=0):
        self.mask_size = None
        self.local_track = local_track
        self.aot_tracker = AOTTracker(cfg, config['gpu_id'])
        # SAM
        self.sam_refine=sam_refine
        if self.sam_refine:
            model_type = 'vit_h' #'vit_h'
            sam_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'segment_anything_hq/pretrained_model/sam_hq_vit_h.pth')
            output_mode = "binary_mask"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=torch.device('cuda'))
            self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
            self.mask_prompt = SamPredictor(sam)
        self.sam_refine_iou=sam_refine_iou

    def get_box(self, label):
        thre = np.max(label) * 0.5
        label[label > thre] = 1
        label[label <= thre] = 0
        a = np.where(label != 0)
        height, width = label.shape
        ratio = 0.0

        if len(a[0]) != 0:
            bbox1 = np.stack([np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])])
            w, h = np.max(a[1]) - np.min(a[1]), np.max(a[0]) - np.min(a[0])
            x1 = max(bbox1[0] - w * ratio, 0)
            y1 = max(bbox1[1] - h * ratio, 0)
            x2 = min(bbox1[2] + w * ratio, width)
            y2 = min(bbox1[3] + h * ratio, height)
            bbox = np.array([x1, y1, x2, y2])
        else:
            bbox = np.array([0, 0, 0, 0])
        return bbox

    def initialize(self, image, mask):
        self.tracker = self.aot_tracker
        self.tracker.add_first_frame(image, mask)
        self.aot_mix_tracker = None
        self.mask_size = mask.shape

    def track(self, image):
        m, confidence = self.tracker.track(image)
        m = F.interpolate(torch.tensor(m)[None, None, :, :],
                          size=self.mask_size, mode="nearest").numpy().astype(np.uint8)[0][0]

        if self.sam_refine:
            obj_list = np.unique(m)
            mask_ = np.zeros_like(m)
            mask_2 = np.zeros_like(m)
            masks_ls = []
            for i in obj_list:
                mask = (m == i).astype(np.uint8)
                if i == 0 or mask.sum() == 0:
                    masks_ls.append(mask_)
                    continue
                bbox = self.get_box(mask)
                # box prompt
                self.mask_prompt.set_image(image)  #set_image의 input_image 부분에서 처리
                masks_, iou_predictions, _ = self.mask_prompt.predict(box=bbox)

                select_index = list(iou_predictions).index(max(iou_predictions))
                output = masks_[select_index].astype(np.uint8)
                iou = pytorch_iou(torch.from_numpy(output).cuda().unsqueeze(0),
                                  torch.from_numpy(mask).cuda().unsqueeze(0), [1])
                iou = iou.cpu().numpy()
                if iou < self.sam_refine_iou:
                    output = mask
                masks_ls.append(output)
                mask_2 = mask_2 + output * i
            masks_ls = np.stack(masks_ls)
            masks_ls_ = masks_ls.sum(0)
            masks_ls_argmax = np.argmax(masks_ls, axis=0)
            rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)
            rs = np.array(rs).astype(np.uint8)

            return rs, confidence
        return m, confidence


def OnMouse_box(event,x,y,flags,param):
    global x0, y0, img4show, img
    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 =x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        x_temp, y_temp = x, y
        img4show=img.copy()
        cv2.rectangle(img4show, (x0, y0), (x_temp, y_temp), (255, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        x1, y1 = x, y
        cv2.rectangle(img4show, (x0, y0), (x, y), (255, 255, 0), 2)
        img=img4show
        point_box_prompts.append([x0, y0, x1, y1])

def OnMouse_point(event,x,y,flags,param):
    global x0, y0, img4show, img
    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 =x,y
        # print(x0,y0)
        point_box_prompts.append([x0,y0])
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img4show, (x0, y0), 4, (0, 255, 0), 6)
        img=img4show


# SAM
print("SAM init ...")
model_type = 'vit_h'
sam_checkpoint = os.path.join(base_path, '..', 'segment_anything_hq/pretrained_model/sam_hq_vit_h.pth')
output_mode = "binary_mask"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=torch.device('cuda'))
mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
mask_prompt = SamPredictor(sam)

# HQTrack config
# choose point or box prompt for SAM
SAM_prompt = 'Box' #'Point
set_Tracker = 'HQTrack'
sam_refine = True
sam_refine_iou = 0.1
muti_object = True
epoch_num=42000
config = {
        'exp_name': 'default',
        'model': 'internT_msdeaotl_v2',
        'pretrain_model_path': 'result/default_InternT_MSDeAOTL_V2/YTB_DAV_VIP/ckpt/save_step_{}.pth'.format(epoch_num),
        'gpu_id': 0,}
# set cfg
print('VMOS init ...')
if set_Tracker in ['HQTrack']:
    engine_config = importlib.import_module('configs.' + 'ytb_vip_dav_deaot_internT')
cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
cfg.TEST_CKPT_PATH = os.path.join(AOT_PATH, config['pretrain_model_path'])
palette_template = Image.open(os.path.join(os.path.dirname(__file__), '..', 'my_tools/mask_palette.png')).getpalette()
tracker = HQTrack(cfg, config, True, sam_refine,sam_refine_iou)
save_dir = 'C:/Users/ye761/HQTrack/demo/output'
#save_dir = './output'
print('starting prompt')

iou_list = []
frame_numbers = []

for idx,img_file in enumerate(img_files):
    print(f"Processing image: {img_file}")

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    #debug tool
    if img is None:
        print(f"Error loading image: {img_file}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ori=img.copy()
    print("Image load success") #debug tool
    # Select ROI
    if idx == 0:
        img4show = img.copy()
        while (1):
            cv2.namedWindow("demo")
            cv2.imshow('demo', cv2.cvtColor(img4show, cv2.COLOR_RGB2BGR))
            if SAM_prompt == 'Box':
                OnMouse = OnMouse_box
            elif SAM_prompt == 'Point':
                OnMouse = OnMouse_point

            cv2.setMouseCallback('demo', OnMouse)
            k = cv2.waitKey(1)
            if k == ord('r'):
                break
        # point prompt
        #유저 로이 선택후 이미지 가공
        print("ROI selected. Processing image...")
        masks_ls = []
        mask_2 = np.zeros_like(img[:,:,0])
        masks_ls.append(mask_2)
        for obj_idx, prompt in enumerate(point_box_prompts):
            mask_prompt.set_image(img_ori)
            if SAM_prompt == 'Box':
                masks_, iou_predictions, _ = mask_prompt.predict(box=np.array(prompt).astype(float))
            elif SAM_prompt == 'Point':
                masks_, iou_predictions, _ = mask_prompt.predict(point_labels=np.asarray([1]), point_coords=np.asarray([prompt]))
            select_index = list(iou_predictions).index(max(iou_predictions))   
            init_mask = masks_[select_index].astype(np.uint8)
            masks_ls.append(init_mask)
            mask_2 = mask_2 + init_mask * (obj_idx+1)
        masks_ls = np.stack(masks_ls)  #
        masks_ls_ = masks_ls.sum(0)
        masks_ls_argmax = np.argmax(masks_ls, axis=0)
        rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)   #최종 마스크 만드는 단계
        rs = np.array(rs).astype(np.uint8)
         #rs정보 디버깅
        print("Initial mask shape:", rs.shape)
        print("Unique values in initial mask:", np.unique(rs))
        init_masks = []
        #마스크 사용하여 원본 이미지 시각적 표시 및 객체의 경계 그리는 로직
        for i in range(len(masks_ls)):
            m_temp = rs.copy()
            m_temp[m_temp!=i+1]=0
            m_temp[m_temp!=0]=1
            init_masks.append(m_temp)
        # img+mask for vis
        img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_RGB2BGR)
        for idx, m in enumerate(init_masks):
            img[:, :, 1] += 127.0 * m
            img[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
        im_m = im_m.clip(0, 255).astype(np.uint8)
        cv2.putText(im_m, 'Init', (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
        cv2.imshow('demo', im_m)
        k = cv2.waitKey(1)
        # HQtrack init
        print('init target objects ...')
        tracker.initialize(img_ori, rs)
        obj_num = len(init_masks)
        print('HQTrack running ...')
    else:
        m, confidence = tracker.track(img_ori)
        print('Running frame ', idx)
        pred_masks = []   ####이거 봐
        for i in range(obj_num):
            m_temp = m.copy()
            m_temp[m_temp != i + 1] = 0
            m_temp[m_temp != 0] = 1
            pred_masks.append(m_temp)
        img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_RGB2BGR)
        for idx, m in enumerate(pred_masks):
            img[:, :, 1] += 127.0 * m
            img[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
        im_m = im_m.clip(0, 255).astype(np.uint8)
        save_path = os.path.join(save_dir, img_file.split('/')[-1])
        cv2.imwrite(save_path, im_m)

    
