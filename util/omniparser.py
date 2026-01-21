from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
import io
import base64
from typing import Dict
class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(model_name=config['caption_model_name'], model_name_or_path=config['caption_model_path'], device=device)
        print('Omniparser initialized!!!')

    def parse(self, image_base64: str):
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print('image size:', image.size)
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        (text, ocr_bbox), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold': 0.8}, use_paddleocr=False)
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD = self.config['BOX_TRESHOLD'], output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

        return dino_labled_img, parsed_content_list
    def parse_screen_with_omniparser(screenshot_path):
        """
        Wrapper function for compatibility with omniparser_executor
    
        Args:
            screenshot_path: Path to screenshot
    
        Returns:
            {
                'elements': [...],
                'resolution': 'WxH'
            }
        """
        try:
            from PIL import Image
            import torch
            from utils import get_yolo_model
        
            # Load image
            image = Image.open(screenshot_path)
            width, height = image.size
        
            # Load model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = get_yolo_model(model_path='weights/icon_detect/best.pt')
        
            # Run detection
            results = model.predict(image, conf=0.15, device=device, verbose=False)
        
            elements = []
            for idx, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy.tolist()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                conf = float(box.conf)
            
                elements.append({
                'id': idx + 1,
                'label': f'UI Element {idx + 1}',
                'x': center_x,
                'y': center_y,
                'confidence': conf,
                'type': 'clickable',
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
            return {
            'elements': elements,
            'resolution': f'{width}x{height}'
            }
    
        except Exception as e:
            print(f"Error in parse_screen_with_omniparser: {e}")
            return {'elements': [], 'resolution': '0x0'}
