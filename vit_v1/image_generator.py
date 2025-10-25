import os
import random
from PIL import Image
import collections

NUM_OUTPUT_IMAGES = 100000

INPUT_DIR_NAME = 'tools'
MESSY_DIR_NAME = 'messy'
NEAT_DIR_NAME = 'neat'
MAX_COPIES_PER_OBJECT = 4
MAX_PLACEMENT_ATTEMPTS = 1000000
CANVAS_SIZE = (256, 256)
RULER_SPACING = 3
HORIZONTAL_SPACING = 3
SPECIAL_TOOL_OVERLAPS = {
    'wrench.png': 1,
    'hammer.png': 6,
    'drill.png': 19
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, INPUT_DIR_NAME)
MESSY_DIR = os.path.join(SCRIPT_DIR, MESSY_DIR_NAME)
NEAT_DIR = os.path.join(SCRIPT_DIR, NEAT_DIR_NAME)

os.makedirs(MESSY_DIR, exist_ok=True)
os.makedirs(NEAT_DIR, exist_ok=True)

def get_image_paths():
    image_paths = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.png'):
            image_paths.append(os.path.join(INPUT_DIR, filename))
    return image_paths

def check_overlap(box1, box2):
    x1, y1, x1_end, y1_end = box1
    x2, y2, x2_end, y2_end = box2
    return not (x1_end <= x2 or x2_end <= x1 or y1_end <= y2 or y2_end <= y1)

def generate_messy_image(objects_to_place, image_id):
    canvas = Image.new('RGBA', CANVAS_SIZE, (255, 255, 255, 255))
    placed_bboxes = []
    placed_objects = []
    
    random.shuffle(objects_to_place)
    
    for obj_path in objects_to_place:
        obj_image = Image.open(obj_path).convert("RGBA")
        rotation = random.randint(0, 359)
        rotated_obj = obj_image.rotate(rotation, expand=True)
        
        tight_bbox = rotated_obj.getbbox()
            
        tight_width = tight_bbox[2] - tight_bbox[0]
        tight_height = tight_bbox[3] - tight_bbox[1]
        
        for attempt in range(MAX_PLACEMENT_ATTEMPTS):
            x = random.randint(0, CANVAS_SIZE[0] - tight_width)
            y = random.randint(0, CANVAS_SIZE[1] - tight_height)
            
            new_bbox = (x, y, x + tight_width, y + tight_height)
            is_overlapping = False
            for placed_bbox in placed_bboxes:
                if check_overlap(new_bbox, placed_bbox):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                paste_x = x - tight_bbox[0]
                paste_y = y - tight_bbox[1]
                
                canvas.paste(rotated_obj, (paste_x, paste_y), mask=rotated_obj)
                placed_bboxes.append(new_bbox)
                placed_objects.append(obj_path)
                break
    
    canvas.save(os.path.join(MESSY_DIR, f'messy_{image_id:06d}.png'))
    return placed_objects

def generate_neat_image(objects_to_place, image_id):
    canvas = Image.new('RGBA', CANVAS_SIZE, (255, 255, 255, 255))
    
    groups = collections.defaultdict(list)
    for obj_path in objects_to_place:
        groups[os.path.basename(obj_path)].append(Image.open(obj_path).convert("RGBA"))
    
    grouped_images_to_place = []

    filenames = list(groups.keys())
    random.shuffle(filenames)

    for filename in filenames:
        obj_list = groups[filename]
        if not obj_list:
            continue
            
        if 'ruler' in filename:
            for i in range(0, len(obj_list), 4):
                ruler_group = obj_list[i:i+4]
                if not ruler_group:
                    continue
                
                max_width = max(img.width for img in ruler_group) if ruler_group else 0
                total_height = sum(img.height for img in ruler_group) + (len(ruler_group) - 1) * RULER_SPACING
                
                group_img = Image.new('RGBA', (max_width, total_height), (0, 0, 0, 0))
                y_offset = 0
                for img in ruler_group:
                    group_img.paste(img, (0, y_offset), mask=img)
                    y_offset += img.height + RULER_SPACING
                grouped_images_to_place.append(group_img)
        
        elif filename in ['wrench.png', 'hammer.png']:
            overlap_amount = SPECIAL_TOOL_OVERLAPS.get(filename, 0)
            
            combined_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            
            for i in range(len(obj_list)):
                img = obj_list[i]
                rotated_img = img.rotate(180) if i % 2 != 0 else img

                new_width = combined_img.width + rotated_img.width - (overlap_amount if i > 0 else 0)
                new_height = max(combined_img.height, rotated_img.height)

                temp_combined = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
                temp_combined.paste(combined_img, (0, new_height - combined_img.height), mask=combined_img)
                
                paste_x = combined_img.width - (overlap_amount if i > 0 else 0)
                paste_y = new_height - rotated_img.height
                
                temp_combined.paste(rotated_img, (paste_x, paste_y), mask=rotated_img)
                combined_img = temp_combined
                
            grouped_images_to_place.append(combined_img)

        elif filename == 'drill.png':
            for i in range(0, len(obj_list), 2):
                pair = obj_list[i:i+2]
                if not pair:
                    continue
                
                img1 = pair[0]
                img2 = pair[1].rotate(180) if len(pair) > 1 else None
                
                overlap_amount = SPECIAL_TOOL_OVERLAPS.get(filename, 0)

                pair_width = img1.width + (img2.width - overlap_amount if img2 else 0)
                pair_height = max(img1.height, img2.height if img2 else 0)

                combined_img = Image.new('RGBA', (pair_width, pair_height), (0, 0, 0, 0))
                
                y1 = pair_height - img1.height
                combined_img.paste(img1, (0, y1), mask=img1)
                
                if img2:
                    y2 = pair_height - img2.height
                    combined_img.paste(img2, (img1.width - overlap_amount, y2), mask=img2)
                
                grouped_images_to_place.append(combined_img)
                
        else:
            combined_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            
            for i in range(len(obj_list)):
                img = obj_list[i]

                new_width = combined_img.width + img.width + (HORIZONTAL_SPACING if i > 0 else 0)
                new_height = max(combined_img.height, img.height)

                temp_combined = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
                temp_combined.paste(combined_img, (0, new_height - combined_img.height), mask=combined_img)
                
                paste_x = combined_img.width + (HORIZONTAL_SPACING if i > 0 else 0)
                paste_y = new_height - img.height
                
                temp_combined.paste(img, (paste_x, paste_y), mask=img)
                combined_img = temp_combined
                
            grouped_images_to_place.append(combined_img)
    
    rows = []
    current_row = []
    current_row_width = 0
    row_height = 0
    
    for group_img in grouped_images_to_place:
        group_width, group_height = group_img.size
        
        if current_row_width + group_width > CANVAS_SIZE[0] and current_row:
            rows.append({'images': current_row, 'height': row_height})
            current_row = []
            current_row_width = 0
            row_height = 0
        
        current_row.append({'image': group_img, 'x': current_row_width, 'height': group_height})
        current_row_width += group_width + HORIZONTAL_SPACING
        row_height = max(row_height, group_height)

    if current_row:
        rows.append({'images': current_row, 'height': row_height})

    y_cursor = 0
    for row in rows:
        max_row_height = row['height']
        for item in row['images']:
            img = item['image']
            x = item['x']
            y_offset = y_cursor
            canvas.paste(img, (x, y_offset), mask=img)
        
        y_cursor += max_row_height + HORIZONTAL_SPACING

    canvas.save(os.path.join(NEAT_DIR, f'neat_{image_id:06d}.png'))
        
def main():
    image_paths = get_image_paths()

    for i in range(NUM_OUTPUT_IMAGES):
        objects_to_place = []
        for obj_path in image_paths:
            num_copies = random.randint(0, MAX_COPIES_PER_OBJECT)
            for _ in range(num_copies):
                objects_to_place.append(obj_path)

        messy_list = objects_to_place[:]
        random.shuffle(messy_list)
        placed_objects = generate_messy_image(messy_list, i)
        
        neat_list = placed_objects[:]
        neat_list.sort()
        generate_neat_image(neat_list, i)

if __name__ == "__main__":
    main()